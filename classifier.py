# TODO:
# Use DistilBERT or ALBERT for faster training if computational resources are limited.
# Clustering (use raw semantic BERT embeddings), other things that were suggested
# Interact with data per song, most like an artist, least like an artist

from __future__ import annotations

from process_eval_funcs import RANDOM_SEED, set_seed, ingest_data, chunk_lyric_dataframe, preprocess, evaluate, pca_visualize_embeddings

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from typing import Optional



def load_saved_model(save_path: str, model: PreTrainedModel) -> PreTrainedModel:
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
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
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
            )
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        if label_names:
            predictions.extend([label_names[p] for p in preds])
        else:
            predictions.extend(preds.tolist())

    return predictions




if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    TRAIN_CSV_PATH = "train_df.csv"
    ARTISTS = {
        "Drake",
        "Eminem",
        "Kanye West",
        "Kendrick Lamar",
        "J. Cole",
        "Travis Scott",
        "Lil Wayne",
        "JAY-Z",
        "Juice WRLD",
        "Future",
        "Nicki Minaj",
        "Tyler, The Creator",
        "Lil Uzi Vert",
        "Migos",
        "2Pac",
        "Young Thug",
        "Mac Miller",
        "YoungBoy Never Broke Again",
        "Chief Keef",
        "Nas",
        "Playboi Carti",
        "Kid Cudi",
        "50 Cent",
        "A Tribe Called Quest",
        "Common",
        "Gucci Mane",
        "Pop Smoke",
    }

    # Configuration
    MODEL_NAME     = "answerdotai/ModernBERT-base"
    NUM_LABELS     = len(ARTISTS)
    MAX_LENGTH     = 128
    BATCH_SIZE     = 16
    SAVE_PATH_BERT = "saved_model/best_bert_classifier.pt"
    SAVE_PATH_LBLS = "saved_model/label_map.json"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── 1. Load tokenizer ─────────────────────────────────────────────────
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    print("\n[Stage 1] Preprocessing...")
    df = ingest_data(
        artists=ARTISTS,
        csv_path=TRAIN_CSV_PATH,
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
    # Order label names by index, not set iteration order (which is hash-seed dependent across processes)
    label_names = [artist for artist, _ in sorted(artist_to_index.items(), key=lambda kv: kv[1])]

    # ── 3. Class weights calculations ─────────────────────────────────────
    class_counts = [label_nums.count(i) for i in range(NUM_LABELS)]
    total_samples = sum(class_counts)
    class_weights = [total_samples / (NUM_LABELS * class_count) for class_count in class_counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Prepare loaders
    train_loader, val_loader = preprocess(
        lyrics, label_nums, song_ids, tokenizer,
        max_length=MAX_LENGTH,
        val_split=0.2,
        batch_size=BATCH_SIZE,
    )

    print(f"Loading saved model from: {SAVE_PATH_BERT}")
    final_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, attn_implementation="flash_attention_2", dtype=torch.bfloat16, num_labels=NUM_LABELS
    )
    final_model = load_saved_model(SAVE_PATH_BERT, final_model)

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    print("\n[Stage 2] Evaluating on validation set...\n")
    metrics = evaluate(final_model, val_loader, device, label_names=label_names)

    confusion_matrix = metrics["confusion_matrix"]
    # get most confused pairs of artists
    most_confused = []
    for i in range(NUM_LABELS):
        for j in range(NUM_LABELS):
            if i != j:
                most_confused.append((confusion_matrix[i][j], label_names[i], label_names[j]))
    most_confused.sort(reverse=True)
    print("\nMost confused pairs of artists:")
    for count, artist1, artist2 in most_confused[:10]:
        print(f"{artist1} vs {artist2}: {count}")

    # ── 7. Inference example ──────────────────────────────────────────────
#     test_texts = [
# """Know you wonder where the F he been (Where he been)
# But I'm back to life like an Epi-Pen
# And she still in the leopard skin
# And I check me out, then check me in
# See this coat, nigga?

# Bye-bye to my old self (Old self)
# Wake up to the new me (It's a new me)
# I used to be on Worldstar (Worldstar)
# Now I'm making Newsweek (Newsweek)
# I used to hang on the 9 (On the 9)
# Now I bought two streets (Two streets)
# Cottage Grove to King Drive (King Drive)
# Yeah, this life is a movie (Movie)
# Bye-bye to my old self (Old self)
# Wake up to the new me (It's a new me)
# I used to be on Worldstar (Worldstar)
# Now I'm making Newsweek (Newsweek)
# I used to hang on the 9 (On the 9)
# Now I bought two streets (Two streets)
# Cottage Grove to King Drive (King Drive)
# Yeah, this life is a movie (Movie)""",
#     ]
#     print("\n[Inference] Predicting on new texts...")
#     predictions = predict(
#         test_texts, final_model, tokenizer, device,
#         max_length=MAX_LENGTH,
#         label_names=label_names,
#     )
#     for text, label in zip(test_texts, predictions):
#         print(f"  '{text}'  →  {label}")