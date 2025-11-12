# src/pca_utils.py
from sklearn.decomposition import PCA

def pca_fit_transform(X, n_components=3, random_state=0):
    """
    Fit PCA on X and return (pca_model, X_transformed).
    """
    p = PCA(n_components=n_components, random_state=random_state)
    Xp = p.fit_transform(X)
    return p, Xp

def variance_retained(pca):
    """
    Sum of explained_variance_ratio_.
    """
    return float(pca.explained_variance_ratio_.sum())
