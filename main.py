import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import LabelEncoder

import umap

# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_CLUSTERS = 10          # fashion-mnist has 10 classes
BATCH_SIZE = 256
EPOCHS = 30
LATENT_DIM = 64          # shared latent size for autoencoders
VAL_SPLIT = 0.1          # fraction of train set used for validation
TEST_SAMPLES = 10000     # full test set

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
SELECTED_CLASSES = [0, 2, 5, 9]   # for the per-class clustering visualisation

os.makedirs("figures", exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# 1. Load & split data
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading Fashion-MNIST …")
(x_full, y_full), (x_test_raw, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalise to [0, 1]
x_full = x_full.astype("float32") / 255.0
x_test_raw = x_test_raw.astype("float32") / 255.0

# Train / validation split (stratified)
from sklearn.model_selection import train_test_split
x_train_raw, x_val_raw, y_train, y_val = train_test_split(
    x_full, y_full, test_size=VAL_SPLIT, random_state=RANDOM_STATE, stratify=y_full
)

print(f"  Train:      {x_train_raw.shape[0]} samples")
print(f"  Validation: {x_val_raw.shape[0]} samples")
print(f"  Test:       {x_test_raw.shape[0]} samples")

# Flat versions (for non-CNN methods)
x_train_flat = x_train_raw.reshape(len(x_train_raw), -1)
x_val_flat   = x_val_raw.reshape(len(x_val_raw),   -1)
x_test_flat  = x_test_raw.reshape(len(x_test_raw), -1)

# CNN versions keep (H, W, 1)
x_train_cnn = x_train_raw[..., np.newaxis]
x_val_cnn   = x_val_raw[..., np.newaxis]
x_test_cnn  = x_test_raw[..., np.newaxis]

# ──────────────────────────────────────────────────────────────────
# 2. Visualise random images (one per class)
# ──────────────────────────────────────────────────────────────────
def plot_sample_images(x, y, title, fname, reconstructed=None):
    fig, axes = plt.subplots(
        2 if reconstructed is not None else 1,
        10, figsize=(15, 3 if reconstructed is None else 5)
    )
    if reconstructed is None:
        axes = [axes]
    for c in range(10):
        idx = np.where(y == c)[0][0]
        axes[0][c].imshow(x[idx].reshape(28, 28), cmap="gray")
        axes[0][c].set_title(CLASS_NAMES[c], fontsize=7)
        axes[0][c].axis("off")
        if reconstructed is not None:
            axes[1][c].imshow(reconstructed[idx].reshape(28, 28), cmap="gray")
            axes[1][c].set_title("Recon.", fontsize=7)
            axes[1][c].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")

plot_sample_images(x_test_raw, y_test, "Fashion-MNIST – one per class",
                   "figures/00_sample_images.png")

# ──────────────────────────────────────────────────────────────────
# 3. Results DataFrame
# ──────────────────────────────────────────────────────────────────
COLUMNS = [
    "DimReduction", "Clustering",
    "DR_TrainTime_s", "Clust_Time_s",
    "N_Clusters_found",
    "CalinskiHarabasz", "DaviesBouldin", "Silhouette", "AdjRandIndex",
]
results_df = pd.DataFrame(columns=COLUMNS)

# ──────────────────────────────────────────────────────────────────
# Helper: clustering suite
# ──────────────────────────────────────────────────────────────────
def run_clustering_suite(X_encoded, y_true, dr_name, dr_time):
    """Apply all 5 clustering algorithms to X_encoded and log results."""
    global results_df

    # For GaussianMixture & AgglomeativeClustering, use PCA-50 if high-dim
    if X_encoded.shape[1] > 100:
        pca_low = PCA(n_components=50, random_state=RANDOM_STATE)
        X_low = pca_low.fit_transform(X_encoded)
    else:
        X_low = X_encoded

    # Auto-estimate a reasonable eps for DBSCAN from nearest-neighbour distances
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(X_low)
    dists, _ = nbrs.kneighbors(X_low)
    eps_auto = float(np.percentile(dists[:, -1], 90))

    algorithms = {
        "MiniBatchKMeans": (
            MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=5),
            X_encoded,
        ),
        "DBSCAN": (
            DBSCAN(eps=eps_auto, min_samples=5, n_jobs=-1),
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
            X_encoded,
        ),
    }

    for alg_name, (alg, X_alg) in algorithms.items():
        print(f"    Clustering: {alg_name} …", end=" ", flush=True)
        t0 = time.time()
        try:
            labels = alg.fit_predict(X_alg)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue
        clust_time = time.time() - t0

        unique_labels = np.unique(labels[labels != -1])
        n_found = len(unique_labels)

        # Metrics require ≥ 2 clusters; use the space the algorithm saw
        mask = labels != -1
        if n_found < 2 or mask.sum() < 2:
            ch, db, sil, ari = np.nan, np.nan, np.nan, np.nan
        else:
            X_m, lab_m, y_m = X_alg[mask], labels[mask], y_true[mask]
            ch  = calinski_harabasz_score(X_m, lab_m)
            db  = davies_bouldin_score(X_m, lab_m)
            sil = silhouette_score(X_m, lab_m, sample_size=3000, random_state=RANDOM_STATE)
            ari = adjusted_rand_score(y_m, lab_m)

        row = {
            "DimReduction":      dr_name,
            "Clustering":        alg_name,
            "DR_TrainTime_s":    round(dr_time, 3),
            "Clust_Time_s":      round(clust_time, 3),
            "N_Clusters_found":  n_found,
            "CalinskiHarabasz":  round(ch, 4) if not np.isnan(ch) else np.nan,
            "DaviesBouldin":     round(db, 4) if not np.isnan(db) else np.nan,
            "Silhouette":        round(sil, 4) if not np.isnan(sil) else np.nan,
            "AdjRandIndex":      round(ari, 4) if not np.isnan(ari) else np.nan,
        }
        results_df = pd.concat(
            [results_df, pd.DataFrame([row])], ignore_index=True
        )
        print(f"done ({clust_time:.1f}s) | clusters={n_found} | sil={sil:.3f}" if not np.isnan(sil) else f"done ({clust_time:.1f}s) | clusters={n_found} | sil=N/A")

# ──────────────────────────────────────────────────────────────────
# Helper: cluster visualisation for 4 classes
# ──────────────────────────────────────────────────────────────────
def plot_cluster_examples(x_images, y_true, labels, dr_name):
    fig, axes = plt.subplots(len(SELECTED_CLASSES), 5, figsize=(10, 8))
    for row_i, cls in enumerate(SELECTED_CLASSES):
        cls_mask = y_true == cls
        cls_labels = labels[cls_mask]
        cls_images = x_images[cls_mask]
        unique_cl = np.unique(cls_labels[cls_labels != -1])
        # pick cluster with most images of this class
        if len(unique_cl) == 0:
            continue
        best_cl = max(unique_cl, key=lambda c: np.sum(cls_labels == c))
        cl_imgs = cls_images[cls_labels == best_cl]
        n_show = min(5, len(cl_imgs))
        for col_i in range(5):
            axes[row_i][col_i].axis("off")
            if col_i < n_show:
                axes[row_i][col_i].imshow(cl_imgs[col_i].reshape(28, 28), cmap="gray")
        axes[row_i][0].set_ylabel(CLASS_NAMES[cls], fontsize=9)
    fig.suptitle(f"Cluster examples – {dr_name} / MiniBatchKMeans")
    plt.tight_layout()
    fname = f"figures/cluster_examples_{dr_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")

# ──────────────────────────────────────────────────────────────────
# Helper: 2-D scatter of encoded space
# ──────────────────────────────────────────────────────────────────
def plot_2d_scatter(X_enc, y, dr_name):
    """Project to 2-D with PCA for visualisation (fast)."""
    X2 = X_enc if X_enc.shape[1] == 2 else PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_enc)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap="tab10", s=2, alpha=0.5)
    plt.colorbar(scatter, ax=ax, ticks=range(10))
    ax.set_title(f"Encoded space – {dr_name}")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    fname = f"figures/scatter_{dr_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")

# ══════════════════════════════════════════════════════════════════
# DIMENSIONALITY REDUCTION TECHNIQUES
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# DR-0  Raw (baseline – no dim reduction)
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DR-0: RAW (no dimensionality reduction)")
print("=" * 60)

X_test_enc = x_test_flat.copy()
plot_2d_scatter(X_test_enc[:3000], y_test[:3000], "Raw")
run_clustering_suite(X_test_enc, y_test, "Raw", dr_time=0.0)

# Cluster visualisation (MiniBatchKMeans labels from last run)
kmeans_raw = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=5)
raw_labels = kmeans_raw.fit_predict(X_test_enc)
plot_cluster_examples(x_test_raw, y_test, raw_labels, "Raw")

# ──────────────────────────────────────────────────────────────────
# DR-1  PCA
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DR-1: PCA")
print("=" * 60)

t0 = time.time()
pca = PCA(n_components=LATENT_DIM, random_state=RANDOM_STATE)
pca.fit(x_train_flat)
pca_train_time = time.time() - t0
print(f"  PCA fit done ({pca_train_time:.1f}s) | explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Visualisation: explained variance curve
fig, ax = plt.subplots(figsize=(7, 4))
pca_full = PCA(random_state=RANDOM_STATE).fit(x_train_flat[:5000])
ax.plot(np.cumsum(pca_full.explained_variance_ratio_) * 100)
ax.axvline(LATENT_DIM, color="red", linestyle="--", label=f"n={LATENT_DIM}")
ax.set_xlabel("Number of components"); ax.set_ylabel("Cumulative explained variance (%)")
ax.set_title("PCA – Cumulative Explained Variance")
ax.legend(); plt.tight_layout()
plt.savefig("figures/pca_explained_variance.png", dpi=100)
plt.close()
print("  Saved: figures/pca_explained_variance.png")

X_test_enc = pca.transform(x_test_flat)
plot_2d_scatter(X_test_enc[:3000], y_test[:3000], "PCA")
run_clustering_suite(X_test_enc, y_test, "PCA", pca_train_time)

kmeans_pca = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=5)
pca_labels = kmeans_pca.fit_predict(X_test_enc)
plot_cluster_examples(x_test_raw, y_test, pca_labels, "PCA")

# ──────────────────────────────────────────────────────────────────
# DR-2  Stacked Autoencoder (SAE)
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DR-2: Stacked Autoencoder (SAE)")
print("=" * 60)

inp = keras.Input(shape=(784,))
e   = layers.Dense(512, activation="relu")(inp)
e   = layers.BatchNormalization()(e)
e   = layers.Dense(256, activation="relu")(e)
e   = layers.BatchNormalization()(e)
encoded_sae = layers.Dense(LATENT_DIM, activation="relu", name="latent")(e)

d   = layers.Dense(256, activation="relu")(encoded_sae)
d   = layers.BatchNormalization()(d)
d   = layers.Dense(512, activation="relu")(d)
d   = layers.BatchNormalization()(d)
decoded_sae = layers.Dense(784, activation="sigmoid")(d)

sae = Model(inp, decoded_sae, name="SAE")
sae_encoder = Model(inp, encoded_sae, name="SAE_encoder")
sae.compile(optimizer="adam", loss="mse")

t0 = time.time()
sae.fit(
    x_train_flat, x_train_flat,
    validation_data=(x_val_flat, x_val_flat),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=0,
)
sae_train_time = time.time() - t0
print(f"  SAE fit done ({sae_train_time:.1f}s)")

# Reconstructions
recon_sae = sae.predict(x_test_flat, verbose=0).reshape(-1, 28, 28)
plot_sample_images(x_test_raw, y_test, "SAE – Original vs Reconstructed",
                   "figures/sae_reconstructions.png", reconstructed=recon_sae)

X_test_enc = sae_encoder.predict(x_test_flat, verbose=0)
plot_2d_scatter(X_test_enc[:3000], y_test[:3000], "SAE")
run_clustering_suite(X_test_enc, y_test, "SAE", sae_train_time)

kmeans_sae = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=5)
sae_labels = kmeans_sae.fit_predict(X_test_enc)
plot_cluster_examples(x_test_raw, y_test, sae_labels, "SAE")

# ──────────────────────────────────────────────────────────────────
# DR-3  Convolutional Stacked Autoencoder (CNN-SAE)
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DR-3: Convolutional SAE (CNN-SAE)")
print("=" * 60)

inp_c = keras.Input(shape=(28, 28, 1))
e_c   = layers.Conv2D(32, 3, activation="relu", padding="same")(inp_c)
e_c   = layers.MaxPooling2D(2)(e_c)                     # 14×14×32
e_c   = layers.Conv2D(64, 3, activation="relu", padding="same")(e_c)
e_c   = layers.MaxPooling2D(2)(e_c)                     # 7×7×64
e_c   = layers.Flatten()(e_c)
encoded_cnn = layers.Dense(LATENT_DIM, activation="relu", name="latent")(e_c)

d_c   = layers.Dense(7 * 7 * 64, activation="relu")(encoded_cnn)
d_c   = layers.Reshape((7, 7, 64))(d_c)
d_c   = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(d_c)
d_c   = layers.UpSampling2D(2)(d_c)                     # 14×14×64
d_c   = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(d_c)
d_c   = layers.UpSampling2D(2)(d_c)                     # 28×28×32
decoded_cnn = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(d_c)

cnn_sae = Model(inp_c, decoded_cnn, name="CNN_SAE")
cnn_encoder = Model(inp_c, encoded_cnn, name="CNN_encoder")
cnn_sae.compile(optimizer="adam", loss="mse")

t0 = time.time()
cnn_sae.fit(
    x_train_cnn, x_train_cnn,
    validation_data=(x_val_cnn, x_val_cnn),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=0,
)
cnn_train_time = time.time() - t0
print(f"  CNN-SAE fit done ({cnn_train_time:.1f}s)")

recon_cnn = cnn_sae.predict(x_test_cnn, verbose=0).squeeze()
plot_sample_images(x_test_raw, y_test, "CNN-SAE – Original vs Reconstructed",
                   "figures/cnnsae_reconstructions.png", reconstructed=recon_cnn)

X_test_enc = cnn_encoder.predict(x_test_cnn, verbose=0)
plot_2d_scatter(X_test_enc[:3000], y_test[:3000], "CNN-SAE")
run_clustering_suite(X_test_enc, y_test, "CNN-SAE", cnn_train_time)

kmeans_cnn = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=5)
cnn_labels = kmeans_cnn.fit_predict(X_test_enc)
plot_cluster_examples(x_test_raw, y_test, cnn_labels, "CNN-SAE")

# ──────────────────────────────────────────────────────────────────
# DR-4  t-SNE
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DR-4: t-SNE")
print("=" * 60)

# t-SNE is expensive; we pre-reduce with PCA first (standard practice)
pca50 = PCA(n_components=50, random_state=RANDOM_STATE).fit(x_train_flat)

t0 = time.time()
X_train_pca50 = pca50.transform(x_train_flat[:5000])   # fit reference (not strictly needed for t-SNE)
tsne_train_time = time.time() - t0                      # PCA step time (t-SNE has no explicit "fit")

X_test_pca50 = pca50.transform(x_test_flat)
print("  Running t-SNE on test set …", end=" ", flush=True)
t0 = time.time()
X_test_enc = TSNE(
    n_components=2, perplexity=30, n_iter=1000,
    random_state=RANDOM_STATE, n_jobs=-1
).fit_transform(X_test_pca50)
tsne_test_time = time.time() - t0
print(f"done ({tsne_test_time:.1f}s)")

tsne_total_time = tsne_train_time + tsne_test_time

# Scatter is the visualisation itself for t-SNE
plot_2d_scatter(X_test_enc[:3000], y_test[:3000], "t-SNE")
run_clustering_suite(X_test_enc, y_test, "t-SNE", tsne_total_time)

kmeans_tsne = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=5)
tsne_labels = kmeans_tsne.fit_predict(X_test_enc)
plot_cluster_examples(x_test_raw, y_test, tsne_labels, "t-SNE")

# ──────────────────────────────────────────────────────────────────
# DR-5  UMAP
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DR-5: UMAP")
print("=" * 60)

t0 = time.time()
umap_model = umap.UMAP(
    n_components=LATENT_DIM, n_neighbors=15, min_dist=0.1,
    random_state=RANDOM_STATE
)
umap_model.fit(x_train_flat)
umap_train_time = time.time() - t0
print(f"  UMAP fit done ({umap_train_time:.1f}s)")

# Visualisation: 2-D UMAP of test set
umap2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
X_test_umap2d = umap2d.fit_transform(x_test_flat)

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(X_test_umap2d[:, 0], X_test_umap2d[:, 1], c=y_test, cmap="tab10", s=2, alpha=0.5)
plt.colorbar(sc, ax=ax, ticks=range(10))
ax.set_title("UMAP 2-D embedding (test set)")
plt.tight_layout(); plt.savefig("figures/umap_2d.png", dpi=100); plt.close()
print("  Saved: figures/umap_2d.png")

X_test_enc = umap_model.transform(x_test_flat)
plot_2d_scatter(X_test_enc[:3000], y_test[:3000], "UMAP")
run_clustering_suite(X_test_enc, y_test, "UMAP", umap_train_time)

kmeans_umap = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=5)
umap_labels = kmeans_umap.fit_predict(X_test_enc)
plot_cluster_examples(x_test_raw, y_test, umap_labels, "UMAP")

# ══════════════════════════════════════════════════════════════════
# Print results table
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.4f}".format)
print(results_df.to_string(index=False))

results_df.to_csv("results.csv", index=False)
print("\nResults saved to results.csv")

# ══════════════════════════════════════════════════════════════════
# Summary heatmap
# ══════════════════════════════════════════════════════════════════
metrics = ["CalinskiHarabasz", "DaviesBouldin", "Silhouette", "AdjRandIndex"]
pivot_sil = results_df.pivot(index="DimReduction", columns="Clustering", values="Silhouette")

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot_sil, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
ax.set_title("Silhouette Score – DR technique × Clustering algorithm")
plt.tight_layout()
plt.savefig("figures/heatmap_silhouette.png", dpi=100)
plt.close()
print("Saved: figures/heatmap_silhouette.png")

print("\nDone.")
