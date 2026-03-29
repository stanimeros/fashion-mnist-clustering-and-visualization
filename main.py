"""
Fashion-MNIST  –  Dimensionality Reduction + Clustering pipeline
================================================================
Modules
  config.py        – all hyper-parameters and the QUICK_RUN flag
  models.py        – PCA / SAE / CNN-SAE / t-SNE / UMAP builders
  clustering.py    – 5 clustering algorithms + 4 evaluation metrics
  visualization.py – all matplotlib helpers

Set  QUICK_RUN = True  in config.py for a fast end-to-end smoke-test.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ── local modules ─────────────────────────────────────────────────
import config                                          # side-effect: creates figures/
from config import (
    QUICK_RUN, RANDOM_STATE, CLUSTER_SAMPLES, TSNE_SAMPLES,
    LATENT_DIM,
)
from models import build_pca, build_sae, build_cnn_sae, build_tsne, build_umap
from clustering import run_clustering_suite, COLUMNS
from visualization import (
    plot_sample_images, plot_2d_scatter, plot_pca_variance,
    plot_umap_2d, plot_cluster_examples, plot_results_heatmap,
)

import tensorflow as tf
from tensorflow import keras

# ──────────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"Loading Fashion-MNIST  (QUICK_RUN={QUICK_RUN}) …")

(x_full, y_full), (x_test_raw, y_test) = keras.datasets.fashion_mnist.load_data()

x_full     = x_full.astype("float32")     / 255.0
x_test_raw = x_test_raw.astype("float32") / 255.0

x_train_raw, x_val_raw, y_train, y_val = train_test_split(
    x_full, y_full,
    test_size=0.1, random_state=RANDOM_STATE, stratify=y_full,
)

print(f"  Train:      {len(x_train_raw):,}")
print(f"  Validation: {len(x_val_raw):,}")
print(f"  Test:       {len(x_test_raw):,}  (clustering on first {CLUSTER_SAMPLES:,})")

# Flat (H*W) versions for dense/PCA models
x_train_flat = x_train_raw.reshape(len(x_train_raw), -1)
x_val_flat   = x_val_raw.reshape(len(x_val_raw),   -1)
x_test_flat  = x_test_raw.reshape(len(x_test_raw), -1)

# CNN versions keep (H, W, 1)
x_train_cnn = x_train_raw[..., np.newaxis]
x_val_cnn   = x_val_raw[..., np.newaxis]
x_test_cnn  = x_test_raw[..., np.newaxis]

# Subsample for clustering & t-SNE
N = CLUSTER_SAMPLES
x_clust_flat = x_test_flat[:N]
x_clust_cnn  = x_test_cnn[:N]
x_clust_raw  = x_test_raw[:N]
y_clust      = y_test[:N]

# ──────────────────────────────────────────────────────────────────
# 2. Show one image per class
# ──────────────────────────────────────────────────────────────────
plot_sample_images(
    x_clust_raw, y_clust,
    "Fashion-MNIST – one sample per class",
    "figures/00_sample_images.png",
)

# ──────────────────────────────────────────────────────────────────
# 3. Results DataFrame
# ──────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(columns=COLUMNS)


def _run(dr_name, X_enc, y_true, dr_time, x_images_for_plot):
    """Helper: scatter + clustering + cluster-example plot."""
    global results_df
    plot_2d_scatter(X_enc[:min(3000, len(X_enc))], y_true[:min(3000, len(y_true))], dr_name)
    results_df, kmeans_labels = run_clustering_suite(
        X_enc, y_true, dr_name, dr_time, results_df
    )
    if kmeans_labels is not None:
        plot_cluster_examples(x_images_for_plot, y_true, kmeans_labels, dr_name)


# ══════════════════════════════════════════════════════════════════
# DR-0  Raw baseline
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DR-0: RAW (no dimensionality reduction)")
print("=" * 60)
_run("Raw", x_clust_flat, y_clust, 0.0, x_clust_raw)

# ══════════════════════════════════════════════════════════════════
# DR-1  PCA
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DR-1: PCA")
print("=" * 60)

encode_pca, pca_time, pca_extras = build_pca(x_train_flat)

# Visualisation: cumulative explained variance (fit on full train, any dim)
pca_full = PCA(random_state=RANDOM_STATE).fit(x_train_flat[:5000])
plot_pca_variance(pca_full)

X_enc_pca = encode_pca(x_clust_flat)
_run("PCA", X_enc_pca, y_clust, pca_time, x_clust_raw)

# ══════════════════════════════════════════════════════════════════
# DR-2  Stacked Autoencoder
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DR-2: Stacked Autoencoder (SAE)")
print("=" * 60)

encode_sae, sae_time, sae_extras = build_sae(x_train_flat, x_val_flat)

recon_sae = sae_extras["reconstruct"](x_clust_flat)
plot_sample_images(
    x_clust_raw, y_clust,
    "SAE – Original vs Reconstructed",
    "figures/sae_reconstructions.png",
    reconstructed=recon_sae,
)

X_enc_sae = encode_sae(x_clust_flat)
_run("SAE", X_enc_sae, y_clust, sae_time, x_clust_raw)

# ══════════════════════════════════════════════════════════════════
# DR-3  Convolutional SAE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DR-3: Convolutional SAE (CNN-SAE)")
print("=" * 60)

encode_cnn, cnn_time, cnn_extras = build_cnn_sae(x_train_cnn, x_val_cnn)

recon_cnn = cnn_extras["reconstruct"](x_clust_cnn)
plot_sample_images(
    x_clust_raw, y_clust,
    "CNN-SAE – Original vs Reconstructed",
    "figures/cnnsae_reconstructions.png",
    reconstructed=recon_cnn,
)

X_enc_cnn = encode_cnn(None, x_clust_cnn)
_run("CNN-SAE", X_enc_cnn, y_clust, cnn_time, x_clust_raw)

# ══════════════════════════════════════════════════════════════════
# DR-4  t-SNE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DR-4: t-SNE")
print("=" * 60)

# t-SNE encodes the test subset directly (no separate transform step)
X_enc_tsne, tsne_time, _ = build_tsne(x_train_flat, x_test_flat, TSNE_SAMPLES)
y_tsne = y_test[:TSNE_SAMPLES]
x_raw_tsne = x_test_raw[:TSNE_SAMPLES]

_run("t-SNE", X_enc_tsne, y_tsne, tsne_time, x_raw_tsne)

# ══════════════════════════════════════════════════════════════════
# DR-5  UMAP
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DR-5: UMAP")
print("=" * 60)

encode_umap, umap_time, _ = build_umap(x_train_flat)

# 2-D UMAP for visualisation (separate fit)
plot_umap_2d(x_clust_flat, y_clust)

X_enc_umap = encode_umap(x_clust_flat)
_run("UMAP", X_enc_umap, y_clust, umap_time, x_clust_raw)

# ══════════════════════════════════════════════════════════════════
# 4. Results summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

pd.set_option("display.max_columns", None)
pd.set_option("display.width",       160)
pd.set_option("display.float_format", "{:.4f}".format)
print(results_df.to_string(index=False))

results_df.to_csv("results.csv", index=False)
print("\nResults saved to results.csv")

plot_results_heatmap(results_df)
print("\nDone.")
