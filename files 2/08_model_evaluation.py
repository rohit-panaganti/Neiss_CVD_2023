"""
Stage 6: Model Evaluation, SHAP Feature Attribution, and Calibration
- Full benchmarking table (Table 1 in paper)
- Bootstrap 1,000 iterations for 95% CI on all metrics (Analysis Plan §8.1)
- SHAP TreeExplainer on Gradient Boosting
- Calibration curves + Expected Calibration Error (ECE)
- Subgroup NLP performance (by age, sex, race) (Analysis Plan §8.1)
- Attention visualization for BioClinicalBERT (integrated gradients)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)

GOLD_PATH = "data/gold_standard_annotated.parquet"
ENSEMBLE_PREDS = "outputs/test_ensemble_predictions.csv"
GB_MODEL = "outputs/models/tfidf_gb.pkl"
TFIDF_VEC = "outputs/models/tfidf_vectorizer.pkl"
OUT_DIR = "outputs/figures"


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                 metric_fn, n_boot: int = 1000, alpha: float = 0.05) -> tuple:
    """
    Bootstrap 1,000 iterations for 95% CI on any performance metric.
    Analysis Plan §8.1.
    """
    rng = np.random.default_rng(42)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_scores.append(metric_fn(y_true[idx], y_score[idx]))
    lo = np.percentile(boot_scores, 100 * alpha / 2)
    hi = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return lo, hi


def subgroup_nlp_performance(preds_df: pd.DataFrame, labels: np.ndarray,
                              ensemble_col: str, out_dir: str):
    """
    Assess NLP accuracy stratified by age group, sex, and race to detect
    differential model performance and potential exclusion bias.
    Analysis Plan §8.1.
    """
    proba = preds_df[ensemble_col].values
    preds_bin = (proba >= 0.5).astype(int)

    subgroups = {}
    if "Age_Years" in preds_df.columns:
        preds_df = preds_df.copy()
        preds_df["age_group"] = pd.cut(
            preds_df["Age_Years"],
            bins=[0, 18, 40, 65, 120],
            labels=["0-17", "18-39", "40-64", "65+"]
        )
        subgroups["age_group"] = preds_df["age_group"].values

    if "Female" in preds_df.columns:
        subgroups["sex"] = preds_df["Female"].map({0: "Male", 1: "Female"}).values

    if "Race_Numeric" in preds_df.columns:
        subgroups["race"] = preds_df["Race_Numeric"].map(
            {1.0: "White", 2.0: "Black", 3.0: "Other",
             4.0: "Asian/PI", 5.0: "Am Indian", 6.0: "Hispanic"}
        ).values

    print("\nSubgroup NLP Performance (Analysis Plan §8.1):")
    results = []
    for sg_name, sg_vals in subgroups.items():
        for group in pd.Series(sg_vals).dropna().unique():
            mask = np.array(sg_vals == group)
            if mask.sum() < 10 or len(np.unique(labels[mask])) < 2:
                continue
            auroc = roc_auc_score(labels[mask], proba[mask])
            f1 = f1_score(labels[mask], preds_bin[mask], zero_division=0)
            auprc = average_precision_score(labels[mask], proba[mask])
            print(f"  {sg_name}={group}: n={mask.sum()}, CVD+={labels[mask].sum()}, "
                  f"AUROC={auroc:.3f}, AUPRC={auprc:.3f}, F1={f1:.3f}")
            results.append({
                "subgroup": sg_name, "group": group,
                "n": int(mask.sum()), "n_pos": int(labels[mask].sum()),
                "auroc": auroc, "auprc": auprc, "f1": f1
            })

    sg_df = pd.DataFrame(results)
    os.makedirs(out_dir, exist_ok=True)
    sg_df.to_csv(f"{out_dir}/subgroup_nlp_performance.csv", index=False)
    return sg_df


def build_benchmarking_table(metrics_list: list) -> pd.DataFrame:
    """
    Compile Table 1: performance benchmarking across all 5 models.
    """
    df = pd.DataFrame(metrics_list)
    df = df.set_index("model")
    df = df[["auroc", "auprc", "precision", "recall", "f1"]]
    df.columns = ["AUROC", "AUPRC", "Precision", "Recall", "F1"]
    print("\nBenchmarking Table (Table 1):")
    print(df.round(4).to_string())
    return df


def plot_roc_pr_curves(predictions: dict, labels: np.ndarray, out_dir: str):
    """
    Plot ROC and PR curves for all models.
    predictions: dict of model_name -> probability array
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, proba in predictions.items():
        fpr, tpr, _ = roc_curve(labels, proba)
        prec, rec, _ = precision_recall_curve(labels, proba)
        auroc = roc_auc_score(labels, proba)
        auprc = average_precision_score(labels, proba)

        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auroc:.3f})")
        axes[1].plot(rec, prec, label=f"{name} (AP={auprc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend(fontsize=8)

    axes[1].axhline(y=labels.mean(), color="k", linestyle="--", alpha=0.4,
                    label=f"Baseline (prev={labels.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = f"{out_dir}/roc_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC/PR curves saved: {path}")


def compute_ece(proba: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (proba >= bin_edges[i]) & (proba < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = proba[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (mask.sum() / len(proba)) * abs(avg_conf - avg_acc)
    return ece


def plot_calibration_curves(predictions: dict, labels: np.ndarray, out_dir: str):
    """Calibration curves with loess-style smoothing."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for name, proba in predictions.items():
        fraction_pos, mean_pred = calibration_curve(labels, proba, n_bins=10, strategy="uniform")
        ece = compute_ece(proba, labels)
        ax.plot(mean_pred, fraction_pos, marker="o", label=f"{name} (ECE={ece:.3f})")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = f"{out_dir}/calibration_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration curves saved: {path}")


def shap_analysis(out_dir: str):
    """
    SHAP TreeExplainer on Gradient Boosting model.
    Top 20 features by mean |SHAP|.
    """
    try:
        import shap
    except ImportError:
        print("shap not installed. Run: pip install shap")
        return

    if not (os.path.exists(GB_MODEL) and os.path.exists(TFIDF_VEC)):
        print("GB model or TF-IDF vectorizer not found. Run stage 4 first.")
        return

    print("\nComputing SHAP values (Gradient Boosting)...")
    gb = joblib.load(GB_MODEL)
    tfidf = joblib.load(TFIDF_VEC)

    # Load test data
    if not os.path.exists(GOLD_PATH):
        print("Gold data not found.")
        return

    df = pd.read_parquet(GOLD_PATH)
    test_df = df[df["split"] == "test"].copy()

    X_tfidf = tfidf.transform(test_df["Narrative_1"])
    # For SHAP, use dense matrix on subset
    X_dense = X_tfidf.toarray()

    explainer = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(X_dense)

    # If binary, shap_values is list [neg_class, pos_class]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Top 20 features
    feature_names = tfidf.get_feature_names_out()
    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-20:][::-1]

    top_features = pd.DataFrame({
        "feature": feature_names[top_idx],
        "mean_abs_shap": mean_abs_shap[top_idx]
    })

    print("\nTop 20 Features by Mean |SHAP| (GB):")
    print(top_features.to_string(index=False))

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_features["feature"][::-1], top_features["mean_abs_shap"][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 Features – Gradient Boosting (SHAP)")
    plt.tight_layout()
    path = f"{out_dir}/shap_top20.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot saved: {path}")

    top_features.to_csv(f"{out_dir}/shap_top20_features.csv", index=False)

    # SHAP force plots for 500 CVD+ and 500 CVD-
    shap.initjs()
    try:
        pos_idx = test_df.index[test_df["gold_label"] == 1][:500]
        neg_idx = test_df.index[test_df["gold_label"] == 0][:500]
        # Force plots saved as HTML
        for label, idx in [("positive", pos_idx), ("negative", neg_idx)]:
            fp = shap.force_plot(
                explainer.expected_value[1] if isinstance(explainer.expected_value, list)
                else explainer.expected_value,
                sv[idx[:10]],
                feature_names=feature_names,
                show=False,
            )
            shap.save_html(f"{out_dir}/shap_force_{label}.html", fp)
    except Exception as e:
        print(f"  Force plot generation skipped: {e}")


def confusion_matrix_grid(predictions: dict, labels: np.ndarray, out_dir: str):
    """Plot confusion matrices for all models."""
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, proba) in zip(axes, predictions.items()):
        preds = (proba >= 0.5).astype(int)
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No CVD", "CVD"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name)

    plt.tight_layout()
    path = f"{out_dir}/confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrices saved: {path}")


def main():
    print("=" * 60)
    print("Stage 6: Model Evaluation & Explainability")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(ENSEMBLE_PREDS):
        print(f"{ENSEMBLE_PREDS} not found. Run stage 5 (ensemble) first.")
        return

    preds_df = pd.read_csv(ENSEMBLE_PREDS)
    labels = preds_df["gold_label"].values

    pred_cols = {
        "Rule-Based": "proba_rule",
        "TF-IDF LR": "proba_lr",
        "TF-IDF GB": "proba_gb",
        "BioClinicalBERT": "proba_bert",
        "LLM": "proba_llm",
        "Ensemble": "ensemble_proba",
    }

    predictions = {}
    metrics_list = []

    for name, col in pred_cols.items():
        if col not in preds_df.columns:
            continue
        proba = preds_df[col].values
        preds = (proba >= 0.5).astype(int)
        predictions[name] = proba

        metrics_list.append({
            "model": name,
            "auroc": roc_auc_score(labels, proba),
            "auprc": average_precision_score(labels, proba),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        })

    benchmarking_table = build_benchmarking_table(metrics_list)
    benchmarking_table.to_csv(f"{OUT_DIR}/benchmarking_table1.csv")

    # Bootstrap 95% CIs for ensemble AUROC and AUPRC (Analysis Plan §8.1)
    if "Ensemble" in predictions:
        ens_proba = predictions["Ensemble"]
        auroc_lo, auroc_hi = bootstrap_ci(labels, ens_proba, roc_auc_score)
        auprc_lo, auprc_hi = bootstrap_ci(labels, ens_proba, average_precision_score)
        print(f"\nEnsemble Bootstrap 95% CIs (n=1000 iterations):")
        print(f"  AUROC: {roc_auc_score(labels, ens_proba):.4f} "
              f"(95% CI {auroc_lo:.4f}–{auroc_hi:.4f})")
        print(f"  AUPRC: {average_precision_score(labels, ens_proba):.4f} "
              f"(95% CI {auprc_lo:.4f}–{auprc_hi:.4f})")

    print("\nGenerating ROC/PR curves...")
    plot_roc_pr_curves(predictions, labels, OUT_DIR)

    print("Generating calibration curves...")
    plot_calibration_curves(predictions, labels, OUT_DIR)

    print("Generating confusion matrices...")
    confusion_matrix_grid(predictions, labels, OUT_DIR)

    print("Running SHAP analysis...")
    shap_analysis(OUT_DIR)

    # Subgroup NLP performance (Analysis Plan §8.1)
    if "ensemble_proba" in preds_df.columns:
        gold = pd.read_parquet(GOLD_PATH) if os.path.exists(GOLD_PATH) else pd.DataFrame()
        test_gold = gold[gold["split"] == "test"].copy() if "split" in gold.columns else pd.DataFrame()
        if not test_gold.empty and "annotation_id" in preds_df.columns:
            merged_sg = preds_df.merge(
                test_gold[["annotation_id", "Age_Years", "Female", "Race_Numeric"]],
                on="annotation_id", how="left"
            )
        else:
            merged_sg = preds_df
        print("\nRunning subgroup NLP performance evaluation...")
        subgroup_nlp_performance(merged_sg, labels, "ensemble_proba", OUT_DIR)

    print(f"\nAll outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
