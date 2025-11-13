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

from kmeans import KMeansScratch
from bisecting_kmeans import BisectingKMeansScratch
from pca_utils import pca_fit_transform

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

    # Standardize + PCA (3 components)
    X = num.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca, Xp = pca_fit_transform(Xs, n_components=3)

    # Clustering
    if algo.startswith("K-Means"):
        model = KMeansScratch(n_clusters=k)
        model.fit(Xp)
    else:
        model = BisectingKMeansScratch(n_clusters=k)
        model.fit(Xp)

    labels = model.labels_
    sse = model.inertia_
    sil = silhouette_score(Xp, labels) if k > 1 else float("nan")

    # Display metrics
    st.subheader("Clustering Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters (k)", k)
    c2.metric("SSE", f"{sse:,.0f}")
    c3.metric("Silhouette", f"{sil:.3f}")

    # PCA Scatter Plot
    st.subheader("PCA Scatter Plot (PC1 vs PC2)")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    scatter = ax.scatter(Xp[:, 0], Xp[:, 1], s=15, c=labels)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

    st.write(
        "Each point represents a customer projected into PCA space. "
        "Colors correspond to different clusters."
    )

else:
    st.info("Please upload a CSV file to begin (e.g., `data/2024_bank_marketing_with_clusters.csv`).")
)
