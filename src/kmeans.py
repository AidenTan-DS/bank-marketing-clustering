
# src/kmeans.py
# From-scratch implementations of K-Means and Bisecting K-Means (NumPy only)

import numpy as np

def _euclid(a, b):
    """Euclidean distance between each row of a and single point b."""
    return np.sqrt(((a - b) ** 2).sum(axis=1))

class KMeansScratch:
    """
    NumPy-only K-Means.
    - random centroid init
    - assign -> update until shift < tol or max_iter
    """
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=42, verbose=False):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X):
        idx = self.rng.choice(len(X), size=self.k, replace=False)
        return X[idx].copy()

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        centroids = self._init_centroids(X)

        for it in range(self.max_iter):
            # assign
            dists = np.stack([_euclid(X, c) for c in centroids], axis=1)
            labels = dists.argmin(axis=1)

            # update
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                for j in range(self.k)
            ])

            # optional: warn empty clusters
            empties = [j for j in range(self.k) if not np.any(labels == j)]
            if self.verbose and empties:
                print(f"[warn] Empty clusters: {empties}")

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = labels
        # SSE (inertia)
        self.inertia_ = float(sum(((X[labels == j] - centroids[j]) ** 2).sum()
                                  for j in range(self.k)))
        return self

    def predict(self, X):
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        dists = np.stack([_euclid(X, c) for c in self.centroids_], axis=1)
        return dists.argmin(axis=1)


class BisectingKMeansScratch:
    """
    Greedy bisecting algorithm:
    - Start with one cluster containing all points
    - Repeatedly split the cluster with largest SSE via 2-means (with few restarts)
    """
    def __init__(self, n_clusters=5, max_iter=20, kmeans_iter=100, restarts=5, random_state=42, verbose=False):
        self.k = n_clusters
        self.max_iter = max_iter
        self.kmeans_iter = kmeans_iter
        self.restarts = restarts
        self.random_state = random_state
        self.verbose = verbose

        self.labels_ = None
        self.centroids_ = None
        self.inertia_ = None

    @staticmethod
    def _sse_for_indices(X, idx):
        cx = X[idx]
        if len(cx) == 0:
            return 0.0
        c = cx.mean(axis=0, keepdims=True)
        return float(((cx - c) ** 2).sum())

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        clusters = {0: np.arange(len(X))}  # cluster_id -> indices

        def run_2means_on_indices(idx, seed):

            km = KMeansScratch(n_clusters=2, max_iter=self.kmeans_iter, random_state=seed)
            km.fit(X[idx])
            return km.labels_, km.inertia_

        while len(clusters) < self.k:
            splittable = {cid: idx for cid, idx in clusters.items() if len(idx) > 1}
            if not splittable:
                break
            cid, idx = max(splittable.items(), key=lambda kv: self._sse_for_indices(X, kv[1]))
            
            # split with multiple restarts of 2-means, keep best (no empty child)
            best_labels = None
            best_inertia = None
            for r in range(self.restarts):
                labels_local, inertia_local = run_2means_on_indices(idx, self.random_state + r)
            
                
                if (labels_local == 0).sum() == 0 or (labels_local == 1).sum() == 0:
                    continue
            
                if best_inertia is None or inertia_local < best_inertia:
                    best_inertia = inertia_local
                    best_labels = labels_local
            
           
            if best_labels is None:
                if self.verbose:
                    print(f"[warn] Unable to split cluster {cid} without empty child, stop splitting it.")
               
                break
            
         
                
            left = idx[best[1] == 0]
            right = idx[best[1] == 1]

            # replace old with two new clusters
            del clusters[cid]
            new_id = max(clusters.keys(), default=-1) + 1
            clusters[new_id] = left
            clusters[new_id + 1] = right

            if self.verbose:
                print(f"[info] Split cluster {cid} -> ({new_id}, {new_id+1}); total={len(clusters)}")

            if len(clusters) >= self.k:
                break

        # finalize labels/centroids
        labels = np.empty(len(X), dtype=int)
        for new_id, idx in enumerate(clusters.values()):
            labels[idx] = new_id
        cents = np.array([X[labels == j].mean(axis=0) for j in range(len(clusters))])

        self.labels_ = labels
        self.centroids_ = cents
        self.inertia_ = float(sum(((X[labels == j] - cents[j]) ** 2).sum()
                                  for j in range(len(clusters))))
        return self
