from __future__ import annotations

from datasets import load_dataset
import random
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.decomposition import PCA
from typing import Optional, Sequence

import sys
import re


RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# INGEST DATA
# ---------------------------------------------------------------------------

def ingest_data(
    artists: set[str],
    hf_dataset: str,
    hf_split: str = "train",
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Load and normalize lyric data.

    Returns a dataframe with canonical columns: artist, song, lyric.
    """

    if cache_path:
        cache_file = Path(cache_path)
        if cache_file.exists():
            print(f"Loading cached dataset from: {cache_file}")
            return pd.read_csv(cache_file, keep_default_na=False)

    ds = load_dataset(hf_dataset, split=hf_split, streaming=True)
    
    artists_lower = {artist.lower() for artist in artists}
    
    records = []
    for record in ds:
        artist = record.get('artist', '')
        if artist.lower() in artists_lower:
            records.append({
                'artist': artist,
                'song': record.get('title', ''),
                'lyric': re.sub(r'\[.*?\]', '', record.get('lyrics', '')),
            })

            sys.stdout.write(f"\rMatches found: {len(records)}")
            sys.stdout.flush()
    print("\nIngestion done")
    df = pd.DataFrame(records)

    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
        print(f"Saved cached dataset to: {cache_file}")

    return df



def chunk_lyric_dataframe(
    df: pd.DataFrame,
    tokenizer: BertTokenizer,
    min_tokens: int = 64,
    max_tokens: int = 128,
    overlap: int = 16,
) -> tuple[list[str], list[str], list[str]]:
    """
    Groups a lyric dataframe by song, aggregates the lines, and splits them into 
    overlapping chunks based on token counts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'artist', 'song', and 'lyric' columns.
    tokenizer : BertTokenizer
        The Hugging Face tokenizer to use.
    min_tokens : int
        The minimum number of tokens a chunk must have to be kept (prevents tiny trailing chunks).
    max_tokens : int
        The maximum window size for each chunk.
    overlap : int
        The number of tokens to overlap between consecutive chunks.
    """
    print("Chunking lyrics into overlapping token-based chunks...")
    stride = max_tokens - overlap
    
    if stride <= 0:
        raise ValueError("Overlap must be strictly less than max_tokens.")
        
    chunked_records = []
    
    for (artist, song), group in df.groupby(['artist', 'song']):
        # Combine all lines for this song into a single string, space-separated
        full_song_text = " ".join(group['lyric'].astype(str).str.strip().tolist())
        
        # Tokenize the entire song into token IDs (without padding/truncation yet)
        input_ids = tokenizer.encode(full_song_text, add_special_tokens=False)
        
        total_tokens = len(input_ids)
        
        # If the whole song is shorter than the minimum token limit, keep it as one chunk
        if total_tokens <= max_tokens:
            if total_tokens >= min_tokens:
                chunk_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                chunked_records.append(
                    {'artist': artist, 'song_id': f"{artist}::{song}", 'chunk_lyric': chunk_text}
                )
            continue
            
        # Sliding window logic
        start_idx = 0
        while start_idx < total_tokens:
            end_idx = start_idx + max_tokens
            chunk_ids = input_ids[start_idx:end_idx]
            
            # Check if the trailing chunk meets the minimum token size requirement
            if len(chunk_ids) < min_tokens:
                break
                
            # Decode token IDs back into actual text strings
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunked_records.append(
                {'artist': artist, 'song_id': f"{artist}::{song}", 'chunk_lyric': chunk_text}
            )
            
            # Slide the window forward
            start_idx += stride

    artists_out = [record['artist'] for record in chunked_records]
    lyrics_out = [record['chunk_lyric'] for record in chunked_records]
    song_ids_out = [record['song_id'] for record in chunked_records]
    return artists_out, lyrics_out, song_ids_out


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------

class _TextClassificationDataset(Dataset):
    """Tokenizes a list of raw texts and stores BERT-ready tensors."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: BertTokenizer,
        max_length: int = 128,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) and labels ({len(labels)}) must have the same length."
            )
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
            "labels":         self.labels[idx],
        }


def preprocess(
    texts: list[str],
    labels: list[int],
    tokenizer: BertTokenizer,
    max_length: int = 128,
    val_split: float = 0.15,
    batch_size: int = 16,
) -> tuple[DataLoader, DataLoader]:
    """
    Tokenize texts and split into train / validation DataLoaders.

    Args:
        texts:       Raw input strings.
        labels:      Integer class indices (0-based).
        tokenizer:   Pre-loaded BertTokenizer.
        max_length:  Maximum token length (sequences are padded/truncated).
        val_split:   Fraction of data reserved for validation.
        batch_size:  Mini-batch size for both loaders.

    Returns:
        (train_loader, val_loader)
    """
    dataset = _TextClassificationDataset(texts, labels, tokenizer, max_length)

    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(RANDOM_SEED) if RANDOM_SEED is not None else None
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    print(
        f"[Preprocessing] Total: {len(dataset)} samples  "
        f"| Train: {train_size}  | Val: {val_size}"
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate(
    model: BertForSequenceClassification,
    data_loader: DataLoader,
    device: torch.device,
    label_names: Optional[list[str]] = None,
) -> dict:
    """
    Evaluate the model and print a full diagnostics report.

    Args:
        model:        Fine-tuned classification model.
        data_loader:  DataLoader for the evaluation split.
        device:       torch.device.
        label_names:  Human-readable class names (optional).

    Returns:
        Dictionary with keys: accuracy, f1_macro, f1_weighted,
        confusion_matrix, classification_report.
    """
    model.eval()
    model.to(device)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc         = accuracy_score(all_labels, all_preds)
    f1_macro    = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    cm          = confusion_matrix(all_labels, all_preds)
    cr          = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        zero_division=0,
    )

    print("=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  F1 (macro)       : {f1_macro:.4f}")
    print(f"  F1 (weighted)    : {f1_weighted:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    print("=" * 55)

    return {
        "accuracy":               acc,
        "f1_macro":               f1_macro,
        "f1_weighted":            f1_weighted,
        "confusion_matrix":       cm,
        "classification_report":  cr,
    }


# ---------------------------------------------------------------------------
# Visualize embeddings
# ---------------------------------------------------------------------------

def _extract_cls_embeddings(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    device: torch.device,
    texts: Sequence[str],
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Extract [CLS] embeddings for each input text."""
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = list(texts[i : i + batch_size])
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model.bert(
                input_ids=encodings["input_ids"].to(device),
                attention_mask=encodings["attention_mask"].to(device),
                token_type_ids=encodings["token_type_ids"].to(device),
            )
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


def _aggregate_embeddings_by_group(
    embeddings: np.ndarray,
    group_ids: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> tuple[list[str], np.ndarray, Optional[list[str]]]:
    """Average chunk-level embeddings into one embedding per group ID."""
    if len(embeddings) != len(group_ids):
        raise ValueError("embeddings and group_ids must have the same length")
    if labels is not None and len(labels) != len(group_ids):
        raise ValueError("labels and group_ids must have the same length")

    group_order: list[str] = []
    group_to_vectors: dict[str, list[np.ndarray]] = {}
    group_to_label: dict[str, str] = {}

    for idx, group_id in enumerate(group_ids):
        if group_id not in group_to_vectors:
            group_order.append(group_id)
            group_to_vectors[group_id] = []
        group_to_vectors[group_id].append(embeddings[idx])

        if labels is not None and group_id not in group_to_label:
            group_to_label[group_id] = str(labels[idx])

    group_embeddings = np.vstack(
        [np.mean(np.vstack(group_to_vectors[group_id]), axis=0) for group_id in group_order]
    )
    group_labels = [group_to_label[group_id] for group_id in group_order] if labels is not None else None
    return group_order, group_embeddings, group_labels


def pca_visualize_embeddings(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    device: torch.device,
    texts: list[str],
    song_ids: list[str],
    labels: Optional[list[str]] = None,
    max_length: int = 128,
    batch_size: int = 16,
):
    model.eval()
    model.to(device)
    chunk_embeddings = _extract_cls_embeddings(
        model,
        tokenizer,
        device,
        texts,
        max_length=max_length,
        batch_size=batch_size,
    )
    song_order, song_embeddings, song_labels = _aggregate_embeddings_by_group(
        chunk_embeddings,
        song_ids,
        labels=labels,
    )

    print(
        f"Generated embeddings for {len(texts)} chunks and aggregated to "
        f"{len(song_embeddings)} songs, performing PCA..."
    )
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    embeddings_2d = pca.fit_transform(song_embeddings)

    plt.figure(figsize=(10, 10))
    if song_labels is not None:
        for label in set(song_labels):
            idxs = [i for i, l in enumerate(song_labels) if l == label]
            plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], label=label)
        for i, song_id in enumerate(song_order):
            plt.annotate(song_id, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    plt.show()

