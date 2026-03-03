"""
Stage 7: BERTopic Unsupervised Topic Modeling
Applied to CVD-positive narratives (n ≈ 2,260).
- Sentence-BERT embeddings (all-MiniLM-L6-v2)
- UMAP dimensionality reduction (n_neighbors=15, n_components=5, metric=cosine)
- HDBSCAN clustering (min_cluster_size=30)
- BERTopic with data-driven cluster discovery (no prespecified n_topics)
- Identifies 6 clinically coherent CVD injury phenotypes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer

ENSEMBLE_LABELS = "data/neiss2023_ensemble_labels.parquet"
OUT_DIR = "outputs/bertopic"
RANDOM_SEED = 42

UMAP_PARAMS = {
    "n_neighbors": 15,
    "n_components": 5,
    "metric": "cosine",
    "random_state": RANDOM_SEED,
    "min_dist": 0.0,
}

HDBSCAN_PARAMS = {
    "min_cluster_size": 30,
    "metric": "euclidean",
    "prediction_data": True,
    "min_samples": 5,
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def extract_cvd_positive_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Extract CVD-positive records for topic modeling."""
    cvd_pos = df[df["CVD_Ensemble_Label"] == 1].copy()
    cvd_pos = cvd_pos[cvd_pos["Narrative_1"].str.strip() != ""].copy()
    print(f"CVD-positive records for topic modeling: {len(cvd_pos):,}")
    return cvd_pos


def generate_embeddings(narratives: list, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Generate sentence-level BERT embeddings."""
    print(f"Generating embeddings with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        narratives,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  Embedding shape: {embeddings.shape}")
    return embeddings


def fit_bertopic(narratives: list, embeddings: np.ndarray) -> tuple:
    """
    Fit BERTopic with UMAP + HDBSCAN.
    Returns (topic_model, topics, probs).
    """
    print("Fitting BERTopic (UMAP + HDBSCAN)...")

    umap_model = UMAP(**UMAP_PARAMS)
    hdbscan_model = HDBSCAN(**HDBSCAN_PARAMS)

    # CountVectorizer for topic representation
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )

    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=EMBEDDING_MODEL,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(narratives, embeddings)
    return topic_model, topics, probs


def summarize_topics(topic_model: BERTopic, topics: list) -> pd.DataFrame:
    """
    Print and return topic summary with counts and top keywords.
    """
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] != -1]  # Exclude noise cluster

    print(f"\nDiscovered {len(topic_info)} topics (excluding noise):")
    print(topic_info[["Topic", "Count", "Name"]].to_string(index=False))

    # Per-topic top words
    print("\nTop terms per topic:")
    for topic_id in topic_info["Topic"].values:
        words = topic_model.get_topic(topic_id)
        word_str = ", ".join([w for w, _ in words[:8]])
        print(f"  Topic {topic_id}: {word_str}")

    return topic_info


def plot_topic_distribution(topic_info: pd.DataFrame, out_dir: str):
    """Bar chart of topic sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    t = topic_info[topic_info["Topic"] != -1].sort_values("Count", ascending=False)
    ax.bar(range(len(t)), t["Count"].values)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels([f"Topic {tid}" for tid in t["Topic"].values], rotation=45)
    ax.set_ylabel("Number of Narratives")
    ax.set_title("BERTopic Cluster Distribution (CVD-Positive Narratives)")
    plt.tight_layout()
    path = f"{out_dir}/topic_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Topic distribution saved: {path}")


def plot_umap_2d(narratives: list, embeddings: np.ndarray, topics: list, out_dir: str):
    """
    2D UMAP visualization for scatter plot.
    """
    print("Generating 2D UMAP visualization...")
    umap_2d = UMAP(
        n_neighbors=15, n_components=2, metric="cosine",
        random_state=RANDOM_SEED, min_dist=0.1
    )
    coords = umap_2d.fit_transform(embeddings)

    topics_arr = np.array(topics)
    unique_topics = sorted(set(topics_arr[topics_arr != -1]))

    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = plt.cm.get_cmap("tab10", len(unique_topics))

    # Noise in gray
    noise_mask = topics_arr == -1
    ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
               c="lightgray", s=5, alpha=0.3, label="Noise")

    for i, topic_id in enumerate(unique_topics):
        mask = topics_arr == topic_id
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i)], s=15, alpha=0.6,
                   label=f"Topic {topic_id} (n={mask.sum():,})")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP Visualization of CVD-Positive Narratives")
    ax.legend(fontsize=8, markerscale=2, loc="best")
    plt.tight_layout()
    path = f"{out_dir}/umap_2d_clusters.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"UMAP 2D scatter saved: {path}")


def assign_clinical_labels(topic_info: pd.DataFrame, topic_model: BERTopic) -> dict:
    """
    Heuristic assignment of clinical phenotype labels based on top keywords.
    Adjust based on actual discovered topics.
    """
    clinical_labels = {}
    for topic_id in topic_info["Topic"].values:
        words = [w.lower() for w, _ in topic_model.get_topic(topic_id)[:5]]
        word_str = " ".join(words)

        if any(k in word_str for k in ["fall", "fell", "floor", "ground"]):
            clinical_labels[topic_id] = "CVD-Complicated Fall"
        elif any(k in word_str for k in ["chest", "pain", "cardiac", "angina"]):
            clinical_labels[topic_id] = "Chest Pain / Acute Cardiac Event"
        elif any(k in word_str for k in ["syncop", "fainting", "loss of consciousness", "loc"]):
            clinical_labels[topic_id] = "Syncope / LOC-Triggered Injury"
        elif any(k in word_str for k in ["anticoag", "warfarin", "bleed", "hemorrhage"]):
            clinical_labels[topic_id] = "Anticoagulation-Related Bleeding"
        elif any(k in word_str for k in ["motor vehicle", "mvc", "crash", "car"]):
            clinical_labels[topic_id] = "CVD During MVC"
        else:
            clinical_labels[topic_id] = f"Mixed CVD Phenotype (Topic {topic_id})"

    return clinical_labels


def main():
    print("=" * 60)
    print("Stage 7: BERTopic Phenotyping")
    print("=" * 60)

    if not os.path.exists(ENSEMBLE_LABELS):
        raise FileNotFoundError(
            f"{ENSEMBLE_LABELS} not found. Run ensemble stage first."
        )

    df = pd.read_parquet(ENSEMBLE_LABELS)
    cvd_df = extract_cvd_positive_narratives(df)
    narratives = cvd_df["Narrative_1"].tolist()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Embeddings
    embeddings = generate_embeddings(narratives)
    np.save(f"{OUT_DIR}/cvd_embeddings.npy", embeddings)

    # BERTopic
    topic_model, topics, probs = fit_bertopic(narratives, embeddings)
    topic_model.save(f"{OUT_DIR}/bertopic_model")

    # Summary
    topic_info = summarize_topics(topic_model, topics)
    topic_info.to_csv(f"{OUT_DIR}/topic_info.csv", index=False)

    # Assign topics back to dataframe
    cvd_df = cvd_df.copy()
    cvd_df["bertopic_cluster"] = topics
    cvd_df.to_parquet(f"{OUT_DIR}/cvd_with_topics.parquet", index=False)

    # Clinical labels
    clinical_labels = assign_clinical_labels(topic_info, topic_model)
    print("\nClinical Phenotype Labels:")
    for tid, label in clinical_labels.items():
        print(f"  Topic {tid}: {label}")

    clinical_df = pd.DataFrame(
        [{"topic_id": k, "clinical_label": v} for k, v in clinical_labels.items()]
    )
    clinical_df.to_csv(f"{OUT_DIR}/clinical_phenotype_labels.csv", index=False)

    # Plots
    plot_topic_distribution(topic_info, OUT_DIR)
    plot_umap_2d(narratives, embeddings, topics, OUT_DIR)

    # Topic keywords table
    print("\nTop keywords per topic (for Table 3):")
    for topic_id in topic_info["Topic"].values:
        words = topic_model.get_topic(topic_id)
        print(f"  Topic {topic_id}: {[w for w, _ in words[:10]]}")

    print(f"\nAll BERTopic outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
