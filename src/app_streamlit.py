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

st.set_page_config(page_title="Bank Marketing Clustering", layout="wide")
st.title("📊 Bank Marketing Clustering (K-Means & Bisecting K-Means)")

st.write(
    "Upload the bank marketing dataset (numeric features only or mixed), "
    "then experiment with different clustering algorithms and values of **k**."
)

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

k = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=4, step=1)
algo = st.selectbox("Clustering algorithm", ["K-Means (scratch)", "Bisecting K-Means (scratch)"])


def load_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Select only numeric columns and drop rows with NaNs."""
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.dropna()
    return num


def safe_silhouette(X_embedded: np.ndarray, labels: np.ndarray, max_samples: int = 1500) -> float:
    """
    Compute silhouette score with optional subsampling to avoid memory blowup.
    """
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


if uploaded:
    # Load data
    raw = pd.read_csv(uploaded)
    st.subheader("Preview of Raw Data")
    st.dataframe(raw.head())

    # Extract numeric features
    num = load_numeric(raw)

    if num.empty:
        st.error("No numeric columns found. Please upload a file with numeric features.")
        st.stop()

    st.subheader("Numeric Features Used for Clustering")
    st.write(list(num.columns))
    st.dataframe(num.head())

    # Standardize
    X = num.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA to 3 components for visualization & silhouette
    n_components = min(3, Xs.shape[1])
    pca = PCA(n_components=n_components)
    Xp = pca.fit_transform(Xs)

    # Clustering on PCA space
    if algo.startswith("K-Means"):
        model = KMeansScratch(n_clusters=k)
        model.fit(Xp)
    else:
        model = BisectingKMeansScratch(n_clusters=k)
        model.fit(Xp)

    labels = model.labels_
    sse = model.inertia_
    sil = safe_silhouette(Xp, labels)

    # Display metrics
    st.subheader("Clustering Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters (k)", k)
    c2.metric("SSE", f"{sse:,.0f}")
    c3.metric("Silhouette", f"{sil:.3f}" if not np.isnan(sil) else "nan")

    # PCA Scatter Plot
    st.subheader("PCA Scatter Plot (PC1 vs PC2)")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if Xp.shape[1] >= 2:
        scatter = ax.scatter(Xp[:, 0], Xp[:, 1], s=15, c=labels)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    else:
        scatter = ax.scatter(Xp[:, 0], np.zeros_like(Xp[:, 0]), s=15, c=labels)
        ax.set_xlabel("PC1")
        ax.set_ylabel("0")

    st.pyplot(fig)

    st.write(
        "Each point represents a customer projected into PCA space. "
        "Colors correspond to different clusters."
    )

else:
    st.info("Please upload a CSV file to begin (e.g., `data/2024_bank_marketing_with_clusters.csv`).")

