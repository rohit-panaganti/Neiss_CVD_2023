"""
Stage 3: Gold Standard Annotation Sample Construction
Constructs a stratified 3,000-record sample (3:1 oversampling of rule-based positives)
for annotation in Label Studio. Exports in Label Studio JSON format.
"""

import pandas as pd
import numpy as np
import json
import os

IN_PATH = "data/neiss2023_rule_based.parquet"
SAMPLE_OUT = "data/annotation_sample.csv"
LABEL_STUDIO_OUT = "data/label_studio_tasks.json"

TOTAL_SAMPLE = 3_000
N_POSITIVE = 750   # rule-based positives (oversampled)
N_NEGATIVE = 2_250  # rule-based negatives

RANDOM_SEED = 42


def build_annotation_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratified sample: 750 rule-based CVD positive, 2250 rule-based negative.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    positives = df[df["CVD_Rule_Label"] == 1]
    negatives = df[df["CVD_Rule_Label"] == 0]

    n_pos_available = len(positives)
    n_pos_sample = min(N_POSITIVE, n_pos_available)

    print(f"  Rule-based positives available: {n_pos_available:,}")
    print(f"  Sampling positives: {n_pos_sample:,}")
    print(f"  Sampling negatives: {N_NEGATIVE:,}")

    pos_sample = positives.sample(n=n_pos_sample, random_state=RANDOM_SEED)
    neg_sample = negatives.sample(n=N_NEGATIVE, random_state=RANDOM_SEED)

    sample = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=RANDOM_SEED)
    sample = sample.reset_index(drop=True)
    sample["annotation_id"] = [f"ANNOT_{i:05d}" for i in range(len(sample))]

    return sample[["annotation_id", "CPSC_Case_Number", "Narrative_1",
                   "CVD_Rule_Label", "Age_Years", "Sex_Numeric"]]


def export_label_studio(sample: pd.DataFrame, path: str):
    """
    Export to Label Studio import format (JSON).
    Annotators label: cvd_present, cvd_absent, uncertain.
    """
    tasks = []
    for _, row in sample.iterrows():
        task = {
            "id": row["annotation_id"],
            "data": {
                "text": row["Narrative_1"],
                "case_number": str(row["CPSC_Case_Number"]),
                "rule_based_label": int(row["CVD_Rule_Label"]),
                "age": float(row["Age_Years"]) if pd.notna(row["Age_Years"]) else None,
            }
        }
        tasks.append(task)

    with open(path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"  Label Studio tasks exported: {path} ({len(tasks):,} tasks)")


def split_train_test(sample: pd.DataFrame) -> tuple:
    """
    80/20 stratified split. Returns (train_df, test_df).
    Call after annotation is complete and gold labels are added.
    """
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        sample,
        test_size=0.20,
        stratify=sample["gold_label"],
        random_state=RANDOM_SEED
    )
    print(f"  Train: {len(train):,} | Test: {len(test):,}")
    return train, test


def load_gold_standard(annotation_export_path: str, sample: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Label Studio annotations back onto the sample.
    Expects a CSV with columns: annotation_id, gold_label (0/1), annotator_1, annotator_2.
    """
    annotations = pd.read_csv(annotation_export_path)
    merged = sample.merge(annotations[["annotation_id", "gold_label"]], on="annotation_id")
    print(f"  Gold standard records merged: {len(merged):,}")
    print(f"  Gold CVD positive: {merged['gold_label'].sum():,} "
          f"({merged['gold_label'].mean()*100:.1f}%)")
    return merged


def main():
    print("=" * 60)
    print("Stage 2: Annotation Sample Construction")
    print("=" * 60)

    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df):,} records")

    print("\nBuilding stratified annotation sample...")
    sample = build_annotation_sample(df)

    os.makedirs("data", exist_ok=True)
    sample.to_csv(SAMPLE_OUT, index=False)
    print(f"  Sample CSV saved: {SAMPLE_OUT}")

    print("\nExporting to Label Studio format...")
    export_label_studio(sample, LABEL_STUDIO_OUT)

    print("\n[ACTION REQUIRED]")
    print("  1. Import label_studio_tasks.json into Label Studio")
    print("  2. Have two annotators label each record (cvd_present/cvd_absent/uncertain)")
    print("  3. Resolve disagreements by consensus adjudication")
    print("  4. Export gold labels as 'data/gold_standard_labels.csv'")
    print("     (columns: annotation_id, gold_label)")
    print("  5. Re-run with gold labels for train/test split:")
    print("     gold_df = load_gold_standard('data/gold_standard_labels.csv', sample)")
    print("     train, test = split_train_test(gold_df)")


if __name__ == "__main__":
    main()
