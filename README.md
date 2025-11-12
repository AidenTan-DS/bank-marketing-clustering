# 📊 Bank Marketing Data Clustering and Analysis

Unsupervised customer segmentation on the **UCI Bank Marketing** dataset using **K-Means** and **Bisecting K-Means (from scratch)**, with **PCA** for dimensionality reduction and a **Streamlit** app for interactive visualization.

## ✨ Highlights
- Implemented **K-Means** & **Bisecting K-Means** from scratch (NumPy core).
- **PCA (3 comps)** for compact representation (report your actual variance retained).
- Evaluated with **SSE** and **Silhouette**; compared to **scikit-learn k-means++**.
- **Streamlit** app for interactive cluster exploration.

## 🚀 Quickstart
```bash
# 1) create env & install
pip install -r requirements.txt

# 2) download dataset to data/bank_marketing.csv (see data/README.md)

# 3) train & evaluate (SSE / Silhouette)
python src/train_eval.py --data data/bank_marketing.csv --kmin 2 --kmax 10

# 4) open interactive app
streamlit run src/app_streamlit.py
