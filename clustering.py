"""
Clustering suite + evaluation metrics.

run_clustering_suite(X_encoded, y_true, dr_name, dr_time)
  -> appends rows to a results DataFrame and returns it together with
     the MiniBatchKMeans label array (used for visualisation).
"""
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    adjusted_rand_score,
)

warnings.filterwarnings('ignore')
from config import RANDOM_STATE, N_CLUSTERS

COLUMNS = [
    "DimReduction", "Clustering",
    "DR_TrainTime_s", "Clust_Time_s",
    "N_Clusters_found",
    "CalinskiHarabasz", "DaviesBouldin", "Silhouette", "AdjRandIndex",
]


def _low_dim(X, max_dims=50):
    """Return a PCA-reduced copy of X when its dimensionality exceeds max_dims."""
    if X.shape[1] <= max_dims:
        return X
    return PCA(n_components=max_dims, random_state=RANDOM_STATE).fit_transform(X)


def _auto_eps(X_low):
    """Estimate a sensible DBSCAN eps from the 90th-percentile 5-NN distance."""
    nbrs = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(X_low)
    dists, _ = nbrs.kneighbors(X_low)
    return float(np.percentile(dists[:, -1], 90))


def _compute_metrics(X, labels, y_true):
    mask    = labels != -1
    lab_m   = labels[mask]
    n_valid = len(np.unique(lab_m))

    if n_valid < 2 or mask.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan

    X_m, y_m = X[mask], y_true[mask]
    try:
        ch  = calinski_harabasz_score(X_m, lab_m)
        db  = davies_bouldin_score(X_m, lab_m)
        sil = silhouette_score(
            X_m, lab_m,
            sample_size=min(3000, len(lab_m)),
            random_state=RANDOM_STATE,
        )
        ari = adjusted_rand_score(y_m, lab_m)
    except Exception:
        ch, db, sil, ari = np.nan, np.nan, np.nan, np.nan

    return ch, db, sil, ari


def run_clustering_suite(X_encoded, y_true, dr_name, dr_time, results_df=None):
    """
    Run all 5 clustering algorithms on X_encoded.

    Returns
    -------
    results_df : pd.DataFrame  (updated with new rows)
    kmeans_labels : np.ndarray  (MiniBatchKMeans labels for visualisation)
    """
    if results_df is None:
        results_df = pd.DataFrame(columns=COLUMNS)

    X_low = _low_dim(X_encoded)
    eps   = _auto_eps(X_low)

    algorithms = {
        "MiniBatchKMeans": (
            MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=5),
            X_encoded,
        ),
        "DBSCAN": (
            DBSCAN(eps=eps, min_samples=5, n_jobs=-1),
            X_low,
        ),
        "AgglomerativeClustering": (
            AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward"),
            X_low,
        ),
        "GaussianMixture": (
            GaussianMixture(
                n_components=N_CLUSTERS, random_state=RANDOM_STATE,
                max_iter=300, reg_covar=1e-4,
            ),
            X_low,
        ),
        "Birch": (
            Birch(n_clusters=N_CLUSTERS, threshold=0.5),
            X_low,
        ),
    }

    kmeans_labels = None

    for alg_name, (alg, X_alg) in algorithms.items():
        print(f"    [{alg_name}] …", end=" ", flush=True)
        t0 = time.time()
        try:
            labels = alg.fit_predict(X_alg)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue
        clust_time = time.time() - t0

        if alg_name == "MiniBatchKMeans":
            kmeans_labels = labels

        n_found = len(np.unique(labels[labels != -1]))
        ch, db, sil, ari = _compute_metrics(X_alg, labels, y_true)

        sil_str = f"{sil:.3f}" if not np.isnan(sil) else "N/A"
        print(f"done ({clust_time:.1f}s) | k={n_found} | sil={sil_str}")

        row = {
            "DimReduction":     dr_name,
            "Clustering":       alg_name,
            "DR_TrainTime_s":   round(dr_time,    3),
            "Clust_Time_s":     round(clust_time, 3),
            "N_Clusters_found": n_found,
            "CalinskiHarabasz": ch  if not np.isnan(ch)  else np.nan,
            "DaviesBouldin":    db  if not np.isnan(db)  else np.nan,
            "Silhouette":       sil if not np.isnan(sil) else np.nan,
            "AdjRandIndex":     ari if not np.isnan(ari) else np.nan,
        }
        results_df = pd.concat(
            [results_df, pd.DataFrame([row])], ignore_index=True
        )

    return results_df, kmeans_labels
