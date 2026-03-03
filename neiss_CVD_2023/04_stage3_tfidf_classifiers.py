"""
Stage 3a/3b: TF-IDF Logistic Regression & Gradient Boosting Classifiers
- TF-IDF vectorization (unigrams + bigrams, 15k features, sublinear TF)
- Age + Sex concatenated as dense features
- Logistic Regression (L2, C=1.0, balanced class weights)
- Gradient Boosting (200 estimators, depth=4, lr=0.05, subsample=0.8)
- 5-fold stratified cross-validation for LR
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from scipy.sparse import hstack, csr_matrix

GOLD_PATH = "data/gold_standard_annotated.parquet"
OUT_DIR = "outputs/models"
RANDOM_SEED = 42


def load_annotated_data(path: str):
    """Load gold-annotated sample with train/test split flags."""
    df = pd.read_parquet(path)
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    print(f"Train: {len(train):,} | Test: {len(test):,}")
    print(f"Train CVD+: {train['gold_label'].sum():,} | Test CVD+: {test['gold_label'].sum():,}")
    return train, test


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """
    Build TF-IDF + structured feature matrices.
    """
    print("Building TF-IDF features...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15_000,
        min_df=2,
        sublinear_tf=True,
        analyzer="word",
    )
    X_train_tfidf = tfidf.fit_transform(train["Narrative_1"])
    X_test_tfidf = tfidf.transform(test["Narrative_1"])

    # Structured covariates: age (standardized) + sex
    scaler = StandardScaler()
    struct_cols = ["Age_Years", "Female"]
    train[struct_cols] = train[struct_cols].fillna(train[struct_cols].median())
    test[struct_cols] = test[struct_cols].fillna(train[struct_cols].median())

    X_train_struct = scaler.fit_transform(train[struct_cols].values)
    X_test_struct = scaler.transform(test[struct_cols].values)

    X_train = hstack([X_train_tfidf, csr_matrix(X_train_struct)])
    X_test = hstack([X_test_tfidf, csr_matrix(X_test_struct)])

    return X_train, X_test, tfidf, scaler


def train_logistic_regression(X_train, y_train):
    """L2 logistic regression with balanced class weights + 5-fold CV."""
    print("Training Logistic Regression (5-fold CV)...")
    lr = LogisticRegression(
        C=1.0,
        penalty="l2",
        class_weight="balanced",
        max_iter=500,
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_results = cross_validate(
        lr, X_train, y_train,
        cv=cv,
        scoring=["roc_auc", "average_precision"],
        return_train_score=False,
        n_jobs=-1,
    )
    print(f"  CV AUROC: {cv_results['test_roc_auc'].mean():.4f} ± "
          f"{cv_results['test_roc_auc'].std():.4f}")
    print(f"  CV AUPRC: {cv_results['test_average_precision'].mean():.4f} ± "
          f"{cv_results['test_average_precision'].std():.4f}")

    lr.fit(X_train, y_train)
    return lr


def train_gradient_boosting(X_train, y_train):
    """Gradient Boosting with 200 estimators."""
    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )
    gb.fit(X_train, y_train)
    return gb


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate on test set."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auroc = roc_auc_score(y_test, proba)
    auprc = average_precision_score(y_test, proba)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    cm = confusion_matrix(y_test, pred)

    print(f"\n{model_name} Test Performance:")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return {
        "model": model_name,
        "auroc": auroc,
        "auprc": auprc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def predict_full_corpus(model, tfidf, scaler, df_full: pd.DataFrame,
                        model_name: str) -> np.ndarray:
    """Apply trained model to full 338k NEISS corpus."""
    X_tfidf = tfidf.transform(df_full["Narrative_1"])
    struct_cols = ["Age_Years", "Female"]
    X_struct = scaler.transform(df_full[struct_cols].fillna(0).values)
    X = hstack([X_tfidf, csr_matrix(X_struct)])
    proba = model.predict_proba(X)[:, 1]
    return proba


def main():
    print("=" * 60)
    print("Stage 3: TF-IDF Classifiers (LR + Gradient Boosting)")
    print("=" * 60)

    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError(
            f"{GOLD_PATH} not found.\n"
            "Complete annotation in Label Studio first (see Stage 2)."
        )

    train, test = load_annotated_data(GOLD_PATH)
    y_train = train["gold_label"].values
    y_test = test["gold_label"].values

    X_train, X_test, tfidf, scaler = build_features(train, test)

    lr = train_logistic_regression(X_train, y_train)
    gb = train_gradient_boosting(X_train, y_train)

    results_lr = evaluate_model(lr, X_test, y_test, "TF-IDF Logistic Regression")
    results_gb = evaluate_model(gb, X_test, y_test, "TF-IDF Gradient Boosting")

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(lr, f"{OUT_DIR}/tfidf_lr.pkl")
    joblib.dump(gb, f"{OUT_DIR}/tfidf_gb.pkl")
    joblib.dump(tfidf, f"{OUT_DIR}/tfidf_vectorizer.pkl")
    joblib.dump(scaler, f"{OUT_DIR}/struct_scaler.pkl")

    # Save test probabilities for ensemble
    test_proba = pd.DataFrame({
        "annotation_id": test["annotation_id"].values,
        "gold_label": y_test,
        "proba_lr": lr.predict_proba(X_test)[:, 1],
        "proba_gb": gb.predict_proba(X_test)[:, 1],
    })
    test_proba.to_csv("outputs/test_probas_tfidf.csv", index=False)
    print(f"\nTest probabilities saved: outputs/test_probas_tfidf.csv")

    pd.DataFrame([results_lr, results_gb]).to_csv(
        "outputs/tfidf_performance.csv", index=False
    )
    print("Performance metrics saved: outputs/tfidf_performance.csv")


if __name__ == "__main__":
    main()
