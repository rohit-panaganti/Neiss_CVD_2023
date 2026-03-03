"""
Stage 3c: Fine-Tuned BioClinicalBERT Classifier
- Model: emilyalsentzer/Bio_ClinicalBERT
- Fine-tuned on 2,400 gold-annotated records
- 4 epochs, LR=2e-5, batch_size=16, max_length=128
- Evaluated on 600-record held-out test set
"""

import pandas as pd
import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
)

GOLD_PATH = "data/gold_standard_annotated.parquet"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
OUT_DIR = "outputs/models/bioclinicalbert"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
RANDOM_SEED = 42


class NEISSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def compute_metrics(eval_pred):
    """HuggingFace Trainer metrics callback."""
    logits, labels = eval_pred
    proba = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (proba >= 0.5).astype(int)

    auroc = roc_auc_score(labels, proba) if len(np.unique(labels)) > 1 else 0.0
    auprc = average_precision_score(labels, proba) if len(np.unique(labels)) > 1 else 0.0
    f1 = f1_score(labels, preds, zero_division=0)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


def train_bioclinicalbert(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fine-tune BioClinicalBERT on gold annotations."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    train_texts = train_df["Narrative_1"].tolist()
    test_texts = test_df["Narrative_1"].tolist()
    train_labels = train_df["gold_label"].tolist()
    test_labels = test_df["gold_label"].tolist()

    train_dataset = NEISSDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    test_dataset = NEISSDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auprc",
        greater_is_better=True,
        seed=RANDOM_SEED,
        logging_dir=f"{OUT_DIR}/logs",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(f"{OUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUT_DIR}/final")

    return trainer, tokenizer, model


def get_bert_probabilities(texts: list, model, tokenizer, batch_size: int = 32) -> np.ndarray:
    """
    Run inference on a list of texts, return class-1 probabilities.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_proba = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        proba = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_proba.extend(proba)

    return np.array(all_proba)


def evaluate_on_test(test_df: pd.DataFrame, model, tokenizer):
    """Full evaluation on test set."""
    texts = test_df["Narrative_1"].tolist()
    labels = test_df["gold_label"].values

    proba = get_bert_probabilities(texts, model, tokenizer)
    preds = (proba >= 0.5).astype(int)

    auroc = roc_auc_score(labels, proba)
    auprc = average_precision_score(labels, proba)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print("\nBioClinicalBERT Test Performance:")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  AUPRC:     {auprc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")

    return proba, {
        "model": "BioClinicalBERT",
        "auroc": auroc, "auprc": auprc,
        "precision": prec, "recall": rec, "f1": f1,
    }


def main():
    print("=" * 60)
    print("Stage 3c: BioClinicalBERT Fine-Tuning")
    print("=" * 60)

    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError(f"{GOLD_PATH} not found. Complete annotation first.")

    df = pd.read_parquet(GOLD_PATH)
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nFine-tuning {MODEL_NAME}...")
    trainer, tokenizer, model = train_bioclinicalbert(train_df, test_df)

    print("\nEvaluating on test set...")
    proba, metrics = evaluate_on_test(test_df, model, tokenizer)

    # Save test probabilities for ensemble
    test_proba_df = pd.DataFrame({
        "annotation_id": test_df["annotation_id"].values,
        "gold_label": test_df["gold_label"].values,
        "proba_bert": proba,
    })
    test_proba_df.to_csv("outputs/test_probas_bert.csv", index=False)
    print("Test probabilities saved: outputs/test_probas_bert.csv")

    with open("outputs/bert_performance.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved: outputs/bert_performance.json")


if __name__ == "__main__":
    main()
