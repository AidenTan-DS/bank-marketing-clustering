# src/app_streamlit.py
# Interactive clustering: upload CSV -> numeric-only -> standardize -> PCA -> cluster -> plot

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from kmeans import KMeansScratch, BisectingKMeansScratch
from pca_utils import pca_fit_transform

st.set_page_config(page_title="Bank Marketing Clustering", layout="wide")
st.title("📊 Bank Marketing Clustering (K-Means & Bisecting K-Means)")

st.write("Upload a CSV with **numeric features** (e.g., the 10D subset in your project).")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

k = st.slider("Number of clusters (k)", min_value=2, max_value=12, value=4, step=1)
algo = st.selectbox("Algorithm", ["K-Means (scratch)", "Bisecting K-Means (scratch)"])

def load_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.dropna()
    return num

if uploaded:
    raw = pd.read_csv(uploaded)
    num = load_numeric(raw)

    if num.empty:
        st.error("No numeric columns found. Please upload a CSV with numeric features.")
        st.stop()

    # Standardize -> PCA(3)
    Xs = StandardScaler().fit_transform(num.values)
    pca, Xp = pca_fit_transform(Xs, n_components=3)

    # Fit chosen algorithm
    if algo.startswith("K-Means"):
        model = KMeansScratch(n_clusters=k).fit(Xp)
    else:
        model = BisectingKMeansScratch(n_clusters=k).fit(Xp)

    labels = model.labels_
    sse = model.inertia_
    sil = silhouette_score(Xp, labels) if k > 1 else float("nan")

    # Metrics
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Clusters (k)", k)
    with c2: st.metric("SSE", f"{sse:,.0f}")
    with c3: st.metric("Silhouette", f"{sil:.3f}")

    # Plot PCA scatter (PC1 vs PC2)
    st.subheader("PCA Scatter (PC1 vs PC2)")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(Xp[:, 0], Xp[:, 1], c=labels, s=12)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

    st.write("**First rows of numeric features used:**")
    st.dataframe(num.head())
else:
    st.info("Upload your CSV to start (e.g., `data/2024_bank_marketing_with_clusters.csv`).")
