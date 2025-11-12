# src/train_eval.py
# Load CSV -> select numeric -> Standardize -> PCA(3) -> Cluster (3 variants) -> Save metrics.csv

import argparse, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans as SKKMeans

from kmeans import KMeansScratch, BisectingKMeansScratch
from pca_utils import pca_fit_transform, variance_retained

def load_numeric_csv(path):
    df = pd.read_csv(path)
    # Only numeric columns (fits your project spec: 10D numeric subset)
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.dropna()
    return num

def main(args):
    os.makedirs("results", exist_ok=True)

    df_num = load_numeric_csv(args.data)
    X = df_num.values

    # Standardize
    Xs = StandardScaler().fit_transform(X)

    # PCA(3)
    pca, Xp = pca_fit_transform(Xs, n_components=3)
    var_ret = variance_retained(pca)

    rows = []
    for k in range(args.kmin, args.kmax + 1):
        # KMeans (scratch)
        km = KMeansScratch(n_clusters=k).fit(Xp)
        sil_k = silhouette_score(Xp, km.labels_) if k > 1 else np.nan
        rows.append(["kmeans_scratch", k, km.inertia_, sil_k, var_ret])

        # Bisecting KMeans (scratch)
        bk = BisectingKMeansScratch(n_clusters=k).fit(Xp)
        sil_b = silhouette_score(Xp, bk.labels_) if k > 1 else np.nan
        rows.append(["bisecting_scratch", k, bk.inertia_, sil_b, var_ret])

        # Baseline: sklearn k-means++
        sk = SKKMeans(n_clusters=k, n_init=10, random_state=0).fit(Xp)  # n_init=10 for compatibility
        sil_s = silhouette_score(Xp, sk.labels_) if k > 1 else np.nan
        rows.append(["sklearn_kmeanspp", k, float(sk.inertia_), sil_s, var_ret])

    out = pd.DataFrame(rows, columns=["model", "k", "SSE", "Silhouette", "PCA_VarianceRetained"])
    out.to_csv("results/metrics.csv", index=False)

    print("\nSaved: results/metrics.csv")
    print("\nAveraged metrics by model:")
    print(out.groupby("model")[["SSE", "Silhouette"]].mean())
    print(f"\nPCA variance retained (3 comps): {var_ret:.2%}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV (numeric features).")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=10)
    main(ap.parse_args())
