# 📊 Bank Marketing Data Clustering and Analysis

Unsupervised customer segmentation on the **UCI Bank Marketing** dataset using:
- **K-Means (from scratch)**
- **Bisecting K-Means (largest-SSE split)**
- **PCA (3 comps)** for dimensionality reduction and visualization

> Implementation and results are based on my original course project report (included in `reports/`). :contentReference[oaicite:0]{index=0}

## ✨ Highlights
- From-scratch **K-Means** & **Bisecting K-Means** in NumPy
- **PCA → clustering** workflow; standardized features with `StandardScaler`
- Evaluation with **SSE (inertia)** and **Silhouette score**
- Optional **Streamlit** app for interactive exploration

## 🗂 Dataset & Features
UCI **Bank Marketing** dataset; this project focuses on the numerical subset:
`age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed`. :contentReference[oaicite:1]{index=1}

- Standardization: z-score (mean=0, std=1) before PCA & clustering. :contentReference[oaicite:2]{index=2}
- PCA(3) gives a compact representation for clustering & plotting. :contentReference[oaicite:3]{index=3}

> See `data/README.md` for how to obtain the dataset.

## 🚀 Quickstart

```bash
# 1) install deps
pip install -r requirements.txt

# 2) run pipeline (PCA -> clustering -> metrics)
python src/pipeline_pca_kmeans.py \
  --data data/2024_bank_marketing_with_clusters.csv \
  --kmin 2 --kmax 10

# 3) open the interactive app
streamlit run src/app_streamlit.py
