"""
All plotting helpers used by main.py.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from config import CLASS_NAMES, SELECTED_CLASSES, RANDOM_STATE


# ──────────────────────────────────────────────────────────────────
def save(fname):
    plt.tight_layout()
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved: {fname}")


# ──────────────────────────────────────────────────────────────────
def plot_sample_images(x_images, y_labels, title, fname, reconstructed=None):
    """One image per class; optionally show reconstructions below."""
    rows = 2 if reconstructed is not None else 1
    fig, axes = plt.subplots(rows, 10, figsize=(15, 3 * rows))
    if rows == 1:
        axes = [axes]

    for c in range(10):
        idx = np.where(y_labels == c)[0][0]
        axes[0][c].imshow(x_images[idx].reshape(28, 28), cmap="gray")
        axes[0][c].set_title(CLASS_NAMES[c], fontsize=7)
        axes[0][c].axis("off")
        if reconstructed is not None:
            axes[1][c].imshow(reconstructed[idx].reshape(28, 28), cmap="gray")
            axes[1][c].set_title("Recon.", fontsize=7)
            axes[1][c].axis("off")

    fig.suptitle(title)
    save(fname)


# ──────────────────────────────────────────────────────────────────
def plot_2d_scatter(X_enc, y, dr_name):
    """Project to 2-D (via PCA if needed) and colour by true label."""
    X2 = (
        X_enc
        if X_enc.shape[1] == 2
        else PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_enc)
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap="tab10", s=2, alpha=0.5)
    plt.colorbar(sc, ax=ax, ticks=range(10))
    ax.set_title(f"Encoded space – {dr_name}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    save(f"figures/scatter_{dr_name.replace(' ', '_')}.png")


# ──────────────────────────────────────────────────────────────────
def plot_pca_variance(pca_full):
    """Cumulative explained variance curve for PCA."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.cumsum(pca_full.explained_variance_ratio_) * 100)
    from config import LATENT_DIM
    ax.axvline(LATENT_DIM, color="red", linestyle="--", label=f"n={LATENT_DIM}")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA – Cumulative Explained Variance")
    ax.legend()
    save("figures/pca_explained_variance.png")


# ──────────────────────────────────────────────────────────────────
def plot_umap_2d(x_test_flat, y_test):
    """Dedicated 2-D UMAP scatter (fitted fresh, for visualisation only)."""
    import umap as _umap
    model2d = _umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                          random_state=RANDOM_STATE)
    X2 = model2d.fit_transform(x_test_flat)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=y_test, cmap="tab10", s=2, alpha=0.5)
    plt.colorbar(sc, ax=ax, ticks=range(10))
    ax.set_title("UMAP 2-D embedding (test set)")
    save("figures/umap_2d.png")


# ──────────────────────────────────────────────────────────────────
def plot_cluster_examples(x_images, y_true, labels, dr_name):
    """For each selected class, show 5 images from the dominant cluster."""
    fig, axes = plt.subplots(len(SELECTED_CLASSES), 5, figsize=(10, 8))

    for row_i, cls in enumerate(SELECTED_CLASSES):
        cls_mask   = y_true == cls
        cls_labels = labels[cls_mask]
        cls_images = x_images[cls_mask]

        unique_cl = np.unique(cls_labels[cls_labels != -1])
        if len(unique_cl) == 0:
            for ax in axes[row_i]:
                ax.axis("off")
            axes[row_i][0].set_ylabel(CLASS_NAMES[cls], fontsize=9)
            continue

        best_cl  = max(unique_cl, key=lambda c: np.sum(cls_labels == c))
        cl_imgs  = cls_images[cls_labels == best_cl]
        n_show   = min(5, len(cl_imgs))

        for col_i in range(5):
            axes[row_i][col_i].axis("off")
            if col_i < n_show:
                axes[row_i][col_i].imshow(cl_imgs[col_i].reshape(28, 28), cmap="gray")

        axes[row_i][0].set_ylabel(CLASS_NAMES[cls], fontsize=9)

    fig.suptitle(f"Cluster examples – {dr_name} / MiniBatchKMeans")
    fname = f"figures/cluster_examples_{dr_name.replace(' ', '_')}.png"
    save(fname)


# ──────────────────────────────────────────────────────────────────
def plot_results_heatmap(results_df):
    """Silhouette score heatmap: DR technique × clustering algorithm."""
    pivot = results_df.pivot_table(
        index="DimReduction", columns="Clustering", values="Silhouette"
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("Silhouette Score – DR technique × Clustering algorithm")
    save("figures/heatmap_silhouette.png")
