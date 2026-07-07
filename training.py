"""
BERT Text Classification Pipeline
==================================
Three stages:
  1. Preprocessing  - tokenize raw texts into BERT-ready tensors
  2. Fine-tuning    - train a BertForSequenceClassification model
  3. Evaluation     - accuracy, F1, confusion matrix, classification report

Dependencies:
    pip install torch transformers scikit-learn numpy
"""

from __future__ import annotations

import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
)
from typing import Optional, Any
import optuna

from process_eval_funcs import RANDOM_SEED, set_seed, ingest_data, chunk_lyric_dataframe, preprocess, evaluate, pca_visualize_embeddings


# ---------------------------------------------------------------------------
# FINE-TUNING
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
    trial: Optional[optuna.Trial] = None,
    early_stopping_patience: Optional[int] = None,
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
        trial:          Optional Optuna trial context.
        early_stopping_patience: Is the number of epochs to wait for validation loss improvement before stopping.

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
    early_stop_counter = 0

    train_loss_history, val_loss_history = [], []

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0

        try:
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
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("  ✗ Out of Memory detected. Clearing CUDA cache and pruning trial.")
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            else:
                raise e

        avg_train_loss = total_train_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        all_preds, all_labels = [], []

        try:
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
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("  ✗ Out of Memory detected during validation. Clearing CUDA cache and pruning trial.")
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            else:
                raise e

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
            early_stop_counter = 0
            if save_path:
                torch.save(best_state, save_path)
                print(f"  ✓ Best checkpoint saved → {save_path}\n")
        else:
            if early_stopping_patience is not None:
                early_stop_counter += 1
                if early_stop_counter >= early_stopping_patience:
                    print(f"  ⚠ Early stopping triggered: validation loss did not improve for {early_stopping_patience} epochs.\n")
                    break

        if trial is not None:
            trial.report(avg_val_loss, step=epoch)
            if trial.should_prune():
                print(f"  ✗ Trial pruned by Optuna at epoch {epoch}\n")
                raise optuna.TrialPruned()

    # Restore best weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_loss_history, val_loss_history


# ---------------------------------------------------------------------------
# Hyperparameter Optimization with Optuna
# ---------------------------------------------------------------------------

def optimize_hyperparameters(
    lyrics: list[str],
    label_nums: list[int],
    tokenizer: BertTokenizer,
    device: torch.device,
    num_labels: int,
    class_weights_tensor: torch.Tensor | None = None,
    n_trials: int = 3,
    epochs: int = 3,
    max_length: int = 128,
) -> dict[str, Any]:
    """
    Run Bayesian Optimization with Optuna to find the optimal hyperparameters:
    learning_rate, weight_decay, warmup_ratio, and batch_size.
    Uses Optuna's MedianPruner and validation-based early stopping to save compute.
    """
    # Prepare loaders
    batch_size = 32
    train_loader, val_loader = preprocess(
        lyrics, label_nums, tokenizer,
        max_length=max_length,
        val_split=0.2,
        batch_size=batch_size,
    )

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        #batch_size = trial.suggest_categorical("batch_size", [32])

        print(
            f"\n[Optuna Trial {trial.number}] Suggested hyperparameters: "
            f"learning_rate={lr:.6e}, weight_decay={weight_decay:.6e}, warmup_ratio={warmup_ratio:.4f}, batch_size={batch_size}"
        )

        # Initialize fresh model each trial configuration
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )

        try:
            # We tune for 'epochs' but early stop model weight updates inside fine_tune (early_stopping_patience=1)
            # and report validation performance metrics to Optuna after each epoch.
            _, _, val_loss_history = fine_tune(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                class_weights_tensor=class_weights_tensor,
                epochs=epochs,
                learning_rate=lr,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                save_path=None,  # skip saving intermediate trial checkpoints
                trial=trial,
                early_stopping_patience=1,  # Stop trial model epoch training if val loss doesn't improve
            )
            # Metric to minimize is the best validation loss achieved in this trial
            best_val_loss = min(val_loss_history) if val_loss_history else float("inf")
            return best_val_loss
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Exception during trial {trial.number}: {e}")
            return float("inf")

    # Set up Optuna study with MedianPruner for early stopping pruning of trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 55)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("=" * 55)
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"  Best Validation Loss: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for param_name, param_val in study.best_params.items():
        print(f"  {param_name}: {param_val}")
    print("=" * 55)

    return study.best_params





if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    DATA_PATH = "updated_rappers.csv"
    ARTISTS = {'Drake', 'Eminem', 'Future', 'Kendrick Lamar', 'Kanye West'}

    # Configuration
    MODEL_NAME     = "bert-base-uncased"
    NUM_LABELS     = len(ARTISTS)
    MAX_LENGTH     = 128
    EPOCHS         = 4
    OPTUNA_TRIALS  = 3
    BATCH_SIZE     = 32
    SAVE_PATH_BERT = "saved_model/best_bert_classifier.pt"
    SAVE_PATH_LBLS = "saved_model/label_map.json"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── 1. Load tokenizer ─────────────────────────────────────────────────
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    print("\n[Stage 1] Preprocessing...")
    df = ingest_data(DATA_PATH, ARTISTS)

    # Training Data
    artists, lyrics, song_ids = chunk_lyric_dataframe(
        df,
        tokenizer,
        min_tokens=MAX_LENGTH//2,
        max_tokens=MAX_LENGTH,
        overlap=MAX_LENGTH//8,
    )
    artist_to_index = {artist: idx for idx, artist in enumerate(ARTISTS)}
    with open(SAVE_PATH_LBLS, "w") as f:
        json.dump(artist_to_index, f)
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

    # ── 4. Hyperparameter Optimization with Optuna ────────────────────────
    print("\n[Stage 2] Running Bayesian Optimization with Optuna...")
    best_params = optimize_hyperparameters(
        lyrics=lyrics,
        label_nums=label_nums,
        tokenizer=tokenizer,
        device=device,
        num_labels=NUM_LABELS,
        class_weights_tensor=class_weights_tensor,
        n_trials=OPTUNA_TRIALS,
        epochs=EPOCHS,
        max_length=MAX_LENGTH,
    )

    # Extract best hyperparams
    best_lr = best_params["learning_rate"]
    best_weight_decay = best_params["weight_decay"]
    best_warmup_ratio = best_params["warmup_ratio"]

    # ── 5. Fine-tune Final Model with Optimal Hyperparameters ──────────────
    print("\n[Stage 3] Fine-tuning final model with optimal hyperparameters...\n")
    
    # Load fresh model for final training
    final_model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    final_model, train_loss_history, val_loss_history = fine_tune(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights_tensor=class_weights_tensor,
        epochs=EPOCHS,
        learning_rate=best_lr,
        weight_decay=best_weight_decay,
        warmup_ratio=best_warmup_ratio,
        save_path=SAVE_PATH_BERT,
        early_stopping_patience=1,  # Early stop the final run too if it starts overfitting
    )

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    print("\n[Stage 4] Evaluating on validation set...\n")
    metrics = evaluate(final_model, val_loader, device, label_names=list(ARTISTS))

    # Visualize embeddings with PCA
    print("\n[Stage 5] Visualizing embeddings with PCA...\n")
    pca_visualize_embeddings(final_model, tokenizer, device, lyrics, song_ids, artists)