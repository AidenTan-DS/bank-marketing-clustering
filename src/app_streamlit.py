# src/app_streamlit.py
# Interactive clustering demo for the Bank Marketing project

import os
import sys

# Ensure src folder is importable
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.append(current_dir)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from kmeans import KMeansScratch
from bisecting_kmeans import BisectingKMeansScratch

# ---------- Streamlit page config ----------
st.set_page_config(
    page_title="Bank Marketing Clustering",
    layout="wide",
)

st.title("📊 Bank Marketing Clustering Dashboard")
st.write(
    "Explore customer segments in the bank marketing dataset using "
    "**K-Means** and **Bisecting K-Means** implemented from scratch."
)

# ---------- Sidebar controls ----------
st.sidebar.header("Controls")

k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=4, step=1)
algo = st.sidebar.selectbox(
    "Clustering algorithm",
    ["K-Means (scratch)", "Bisecting K-Means (scratch)"],
)

max_samples_sil = st.sidebar.slider(
    "Max samples for Silhouette (to avoid memory issues)",
    min_value=300,
    max_value=2000,
    value=1200,
    step=100,
)

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])


# ---------- Helper functions ----------
def load_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select numeric columns, then drop 'id' / 'cluster' style columns
    from the feature set used for clustering.

    Returns:
        num_all: all numeric columns
        features: numeric columns actually used for clustering
    """
    num_all = df.select_dtypes(include=[np.number]).copy()

    # columns we do NOT want to use as features
    block_names = {"cluster", "clusters", "label", "id", "index"}
    drop_cols = [c for c in num_all.columns if c.lower() in block_names]

    features = num_all.drop(columns=drop_cols, errors="ignore")

    return num_all, features


def safe_silhouette(X_embedded: np.ndarray, labels: np.ndarray, max_samples: int = 1500) -> float:
    """Compute silhouette score with optional subsampling to avoid memory blowup."""
    n = X_embedded.shape[0]
    if n <= 1 or len(np.unique(labels)) < 2:
        return float("nan")

    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        X_sub = X_embedded[idx]
        labels_sub = labels[idx]
        return silhouette_score(X_sub, labels_sub)
    else:
        return silhouette_score(X_embedded, labels)


# ---------- Main app ----------
if uploaded:
    # 1. Load data
    df = pd.read_csv(uploaded)

    st.subheader("📁 Raw Data Preview")
    st.dataframe(df.head())

    # 2. Select numeric & feature columns
    num_all, features = load_numeric(df)

    if features.empty:
        st.error(
            "No usable numeric feature columns found. "
            "Please upload a file with numeric variables."
        )
        st.stop()

    st.subheader("🔢 Numeric columns detected")
    st.write(list(num_all.columns))

    st.subheader("✅ Features used for clustering (numeric, excluding 'id' / 'cluster')")
    st.write(list(features.columns))

    st.write("First rows of numeric features used:")
    st.dataframe(features.head())

    # 3. Standardize
    X = features.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 4. PCA to 3 components for visualization & silhouette
    n_components = min(3, Xs.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    Xp = pca.fit_transform(Xs)

    # 5. Clustering on PCA space
    if algo.startswith("K-Means"):
        model = KMeansScratch(n_clusters=k)
        model.fit(Xp)
    else:
        model = BisectingKMeansScratch(n_clusters=k)
        model.fit(Xp)

    labels = model.labels_
    sse = getattr(model, "inertia_", np.nan)

    sil = safe_silhouette(Xp, labels, max_samples=max_samples_sil)

    # 6. Layout: metrics + plots + cluster summary
    st.subheader("📈 Clustering Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", algo.split(" ")[0])
    c2.metric("Clusters (k)", k)
    c3.metric("Silhouette", f"{sil:.3f}" if not np.isnan(sil) else "nan")

    st.caption(
        "Silhouette is computed on PCA-transformed features. "
        f"When sample size > {max_samples_sil}, a random subset is used "
        "to avoid memory issues."
    )

    st.subheader("📊 Cluster Summary")
    cluster_sizes = (
        pd.Series(labels, name="cluster")
        .value_counts()
        .sort_index()
        .rename("count")
        .to_frame()
    )
    st.dataframe(cluster_sizes)

    # Two columns for plots
    p1, p2 = st.columns(2)

    # 7. PCA scatter
    with p1:
        st.markdown("#### PCA Scatter Plot (PC1 vs PC2)")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        if Xp.shape[1] >= 2:
            scatter = ax.scatter(Xp[:, 0], Xp[:, 1], s=10, c=labels, cmap="tab10")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        else:
            scatter = ax.scatter(Xp[:, 0], np.zeros_like(Xp[:, 0]), s=10, c=labels, cmap="tab10")
            ax.set_xlabel("PC1")
            ax.set_ylabel("0")

        st.pyplot(fig)
        st.caption("Each point is a customer in PCA space; color indicates cluster membership.")

    # 8. Cluster size bar chart
    with p2:
        st.markdown("#### Cluster Size Distribution")

        fig2, ax2 = plt.subplots()
        ax2.bar(cluster_sizes.index.astype(str), cluster_sizes["count"])
        ax2.set_xlabel("Cluster label")
        ax2.set_ylabel("Number of samples")
        st.pyplot(fig2)

        st.caption("Use this chart to see whether clusters are balanced or highly skewed.")

else:
    st.info(
        "⬅ Please upload a CSV file in the sidebar to begin "
        "(for example: `data/2024_bank_marketing_with_clusters.csv`)."
    )

