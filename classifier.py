# TODO:
# Use DistilBERT or ALBERT for faster training if computational resources are limited.
# Clustering (use raw semantic BERT embeddings), other things that were suggested
# Interact with data per song, most like an artist, least like an artist

from __future__ import annotations

from process_eval_funcs import RANDOM_SEED, set_seed, ingest_data, chunk_lyric_dataframe, preprocess, evaluate, pca_visualize_embeddings, load_dataset

import json
from pathlib import Path
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)

from typing import Optional



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




if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    DATASET_NAME = "theelderemo/genius-lyrics-cleaned"
    DATASET_SPLIT = "train"
    ARTISTS = {'Kendrick Lamar', 'Kanye West'}
    CACHE_PATH = Path("saved_model") / "cached_dataset.csv"

    # Configuration
    MODEL_NAME     = "bert-base-uncased"
    NUM_LABELS     = len(ARTISTS)
    MAX_LENGTH     = 128
    BATCH_SIZE     = 16
    SAVE_PATH_BERT = "saved_model/best_bert_classifier.pt"
    SAVE_PATH_LBLS = "saved_model/label_map.json"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── 1. Load tokenizer ─────────────────────────────────────────────────
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    print("\n[Stage 1] Preprocessing...")
    df = ingest_data(artists=ARTISTS,
        hf_dataset=DATASET_NAME,
        hf_split=DATASET_SPLIT,
        cache_path=str(CACHE_PATH),
    )

    # Training Data
    artists, lyrics, song_ids = chunk_lyric_dataframe(
        df,
        tokenizer,
        min_tokens=MAX_LENGTH//2,
        max_tokens=MAX_LENGTH,
        overlap=MAX_LENGTH//8,
    )
    with open(SAVE_PATH_LBLS, "r") as f:
        artist_to_index = json.load(f)
    label_nums = [artist_to_index[artist] for artist in artists]

    # ── 3. Class weights calculations ─────────────────────────────────────
    class_counts = [label_nums.count(i) for i in range(NUM_LABELS)]
    total_samples = sum(class_counts)
    class_weights = [total_samples / (NUM_LABELS * class_count) for class_count in class_counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Prepare loaders
    train_loader, val_loader = preprocess(
        lyrics, label_nums, tokenizer,
        max_length=MAX_LENGTH,
        val_split=0.2,
        batch_size=BATCH_SIZE,
    )

    print(f"Loading saved model from: {SAVE_PATH_BERT}")
    final_model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    final_model = load_saved_model(SAVE_PATH_BERT, final_model)

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    # print("\n[Stage 2] Evaluating on validation set...\n")
    # metrics = evaluate(final_model, val_loader, device, label_names=list(ARTISTS))

    # # ── 7. Inference example ──────────────────────────────────────────────
    test_texts = [
"""Know you wonder where the F he been (Where he been)
But I'm back to life like an Epi-Pen
And she still in the leopard skin
And I check me out, then check me in
See this coat, nigga?

Bye-bye to my old self (Old self)
Wake up to the new me (It's a new me)
I used to be on Worldstar (Worldstar)
Now I'm making Newsweek (Newsweek)
I used to hang on the 9 (On the 9)
Now I bought two streets (Two streets)
Cottage Grove to King Drive (King Drive)
Yeah, this life is a movie (Movie)
Bye-bye to my old self (Old self)
Wake up to the new me (It's a new me)
I used to be on Worldstar (Worldstar)
Now I'm making Newsweek (Newsweek)
I used to hang on the 9 (On the 9)
Now I bought two streets (Two streets)
Cottage Grove to King Drive (King Drive)
Yeah, this life is a movie (Movie)""",
    ]
    print("\n[Inference] Predicting on new texts...")
    predictions = predict(
        test_texts, final_model, tokenizer, device,
        max_length=MAX_LENGTH,
        label_names=list(ARTISTS),
    )
    for text, label in zip(test_texts, predictions):
        print(f"  '{text}'  →  {label}")

    # Visualize embeddings with PCA
    # print("\n[Stage 3] Visualizing embeddings with PCA...\n")
    bert_base = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    # pca_visualize_embeddings(final_model, tokenizer, device, lyrics, song_ids, artists)
    pca_visualize_embeddings(bert_base, tokenizer, device, lyrics, song_ids, artists)