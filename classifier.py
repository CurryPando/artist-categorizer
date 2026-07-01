"""
BERT Text Classification Pipeline
==================================
Covers three stages:
  1. Preprocessing  - tokenize raw texts into BERT-ready tensors
  2. Fine-tuning    - train a BertForSequenceClassification model
  3. Evaluation     - accuracy, F1, confusion matrix, classification report

Dependencies:
    pip install torch transformers scikit-learn numpy
"""

# TODO:
# Use DistilBERT or ALBERT for faster training if computational resources are limited.
# Apply techniques like early stopping or learning rate scheduling to prevent overfitting.

from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Optional, Sequence

from index import ingest_data, get_lyric_data_by_artist, chunk_lyric_dataframe

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
# 1. PREPROCESSING
# ---------------------------------------------------------------------------

class TextClassificationDataset(Dataset):
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
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length)

    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    print(
        f"[Preprocessing] Total: {len(dataset)} samples  "
        f"| Train: {train_size}  | Val: {val_size}"
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 2. FINE-TUNING
# ---------------------------------------------------------------------------

def fine_tune(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights_tensor: torch.Tensor | None = None,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    save_path: Optional[str] = None,
) -> tuple[BertForSequenceClassification, list[float], list[float]]:
    """
    Fine-tune a BertForSequenceClassification model.

    Args:
        model:          HuggingFace BERT classification model.
        train_loader:   DataLoader for training data.
        val_loader:     DataLoader for validation data.
        class_weights_tensor:  Tensor of class weights for CrossEntropyLoss.
        device:         torch.device ('cuda' or 'cpu').
        epochs:         Number of full passes over the training set.
        learning_rate:  Peak LR for AdamW optimizer.
        warmup_ratio:   Fraction of total steps used for LR warm-up.
        weight_decay:   L2 regularization coefficient.
        save_path:      If provided, save the best checkpoint here.

    Returns:
        The fine-tuned model (best checkpoint by validation loss).
    """
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Optimizer — separate weight decay from bias / LayerNorm parameters
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_state    = None

    train_loss_history, val_loss_history = [], []

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            if step % max(1, len(train_loader) // 5) == 0:
                avg = total_train_loss / step
                print(f"  Epoch {epoch} | Step {step:>4}/{len(train_loader)} | Train Loss: {avg:.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels         = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                total_val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc      = accuracy_score(all_labels, all_preds)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        print(
            f"\nEpoch {epoch}/{epochs} Summary\n"
            f"  Train Loss : {avg_train_loss:.4f}\n"
            f"  Val   Loss : {avg_val_loss:.4f}\n"
            f"  Val   Acc  : {val_acc:.4f}\n"
        )

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            if save_path:
                torch.save(best_state, save_path)
                print(f"  ✓ Best checkpoint saved → {save_path}\n")

    # Restore best weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_loss_history, val_loss_history

def load_saved_model(save_path: str, model: BertForSequenceClassification) -> BertForSequenceClassification:
    """
    Load a saved model checkpoint.

    Args:
        save_path: Path to the saved model checkpoint.
        model:     Model instance to load the checkpoint into.

    Returns:
        Model with loaded weights.
    """
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)
    return model

# ---------------------------------------------------------------------------
# 3. EVALUATION
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
# Inference helper
# ---------------------------------------------------------------------------

def predict(
    texts: list[str],
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 16,
    label_names: Optional[list[str]] = None,
) -> list[str | int]:
    """
    Run inference on arbitrary texts.

    Returns:
        List of predicted label names (if provided) or integer indices.
    """
    model.eval()
    model.to(device)
    predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings   = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(
                input_ids=encodings["input_ids"].to(device),
                attention_mask=encodings["attention_mask"].to(device),
                token_type_ids=encodings["token_type_ids"].to(device),
            )
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        if label_names:
            predictions.extend([label_names[p] for p in preds])
        else:
            predictions.extend(preds.tolist())

    return predictions


# ---------------------------------------------------------------------------
# Visualize embeddings
# ---------------------------------------------------------------------------

def tsne_visualize_embeddings(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    device: torch.device,
    texts: list[str],
    labels: list[str],
    max_length: int = 128,
    batch_size: int = 16,
):
    """
    Visualize embeddings of the given texts using the model.

    Args:
        model: Pretrained BERT model.
        tokenizer: BERT tokenizer.
        device: Torch device.
        texts: List of texts to visualize.
        labels: Optional list of labels for coloring the embeddings.
        max_length: Maximum token length for the tokenizer.
        batch_size: Batch size for processing texts.
    """
    model.eval()
    model.to(device)
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
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

    embeddings = np.vstack(embeddings)

    print(f"Generated embeddings for {len(texts)} texts, performing TSNE...")

    tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    if labels is not None:
        for label in set(labels):
            idxs = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], label=label)
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    plt.show()

def pca_visualize_embeddings(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    device: torch.device,
    texts: list[str],
    labels: list[str],
    max_length: int = 128,
    batch_size: int = 16,
):
    model.eval()
    model.to(device)
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
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

    embeddings = np.vstack(embeddings)

    print(f"Generated embeddings for {len(texts)} texts, performing PCA...")
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    if labels is not None:
        for label in set(labels):
            idxs = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], label=label)
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    plt.show()


if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    DATA_PATH = "updated_rappers.csv"
    ARTISTS = {'Drake', 'Eminem', 'Future', 'XXXTentacion', 'Nas', '21 Savage', 'Kendrick Lamar'}

    # Configuration
    MODEL_NAME  = "bert-base-uncased"
    NUM_LABELS  = len(ARTISTS)
    MAX_LENGTH  = 64
    BATCH_SIZE  = 8
    EPOCHS      = 3
    LR          = 2e-5
    SAVE_PATH   = "best_bert_classifier.pt"  # set to None to skip saving

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── 1. Load tokenizer ─────────────────────────────────────────────────
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    print("\n[Stage 1] Preprocessing...")
    df = ingest_data(DATA_PATH, ARTISTS)

    # Training Data
    artists, lyrics = chunk_lyric_dataframe(df, tokenizer, min_tokens=32, max_tokens=64, overlap=8)
    artist_to_index = {artist: idx for idx, artist in enumerate(ARTISTS)}
    label_nums = [artist_to_index[artist] for artist in artists]

    print(lyrics[:5])

    train_loader, val_loader = preprocess(
        lyrics, label_nums, tokenizer,
        max_length=MAX_LENGTH,
        val_split=0.2,
        batch_size=BATCH_SIZE,
    )

    # ── 3. Load model ─────────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_NAME} with {NUM_LABELS} labels")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    class_counts = [label_nums.count(i) for i in range(NUM_LABELS)]
    total_samples = sum(class_counts)
    class_weights = [total_samples / (NUM_LABELS * class_count) for class_count in class_counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # ── 4. Fine-tune ──────────────────────────────────────────────────────
    print("\n[Stage 2] Fine-tuning...\n")
    model, train_loss_history, val_loss_history = fine_tune(
        model, train_loader, val_loader, device,
        class_weights_tensor=class_weights_tensor,
        epochs=EPOCHS,
        learning_rate=LR,
        save_path=SAVE_PATH,
    )
    
    # model = load_saved_model(SAVE_PATH, model)

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    print("\n[Stage 3] Evaluating on validation set...\n")
    metrics = evaluate(model, val_loader, device, label_names=list(ARTISTS))

    # # ── 6. Inference example ──────────────────────────────────────────────
    # test_texts = [
    #     "I like beating women",
    #     "I love to rap and spit fire like Eminem.",
    # ]
    # print("\n[Inference] Predicting on new texts...")
    # predictions = predict(
    #     test_texts, model, tokenizer, device,
    #     max_length=MAX_LENGTH,
    #     label_names=list(ARTISTS),
    # )
    # for text, label in zip(test_texts, predictions):
    #     print(f"  '{text}'  →  {label}")

    # Visualize embeddings with PCA
    print("\n[Stage 4] Visualizing embeddings with PCA...\n")
    pca_visualize_embeddings(model, tokenizer, device, lyrics, artists)