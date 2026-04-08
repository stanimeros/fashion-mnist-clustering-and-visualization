# Fashion-MNIST: Dimensionality Reduction & Clustering

Evaluation of 5 dimensionality reduction techniques combined with 5 clustering algorithms on the Fashion-MNIST dataset, scored with 4 performance metrics.

---

## Project structure

```
cv-clustering/
├── config.py          # Hyper-parameters; quick mode via env FASHION_MNIST_QUICK_RUN
├── models.py          # Dimensionality reduction builders (PCA, SAE, CNN-SAE, t-SNE, UMAP)
├── clustering.py      # 5 clustering algorithms + 4 evaluation metrics
├── visualization.py   # All matplotlib/seaborn helpers
├── main.py            # Pipeline orchestrator
├── run_pipeline.sh    # venv (αν λείπει) + pip μόνο αν άλλαξε το requirements.txt + run
├── logs/              # pipeline-*.log (created by background runs; gitignored)
├── requirements.txt
└── figures/           # All generated plots (created automatically)
```

---

## Setup

**Python:** χρησιμοποίησε έκδοση που υποστηρίζει το TensorFlow (συνήθως **3.10–3.12** για TF 2.21). Το `keras` ως ξεχωριστό πακέτο δεν χρειάζεται στο `requirements.txt` — προτιμάται `tensorflow.keras`.

```bash
python3.11 -m venv .venv   # ή python3.10 / python3.12 αν ταιριάζει με το TF
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running

**One-shot (recommended):** δημιουργεί `.venv` μόνο αν δεν υπάρχει· τρέχει `pip install` μόνο όταν αλλάξει το `requirements.txt` (αποθηκεύεται hash στο `.venv/.requirements.sha256`). Για εκ νέου εγκατάσταση: `FORCE_PIP_INSTALL=1 ./run_pipeline.sh`.

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh        # full run (default)
./run_pipeline.sh full
./run_pipeline.sh quick  # smoke test
```

**Background run** (μπορείς να κλείσεις το terminal· όλη η έξοδος πάει σε `logs/pipeline-YYYYMMDD-HHMMSS.log`):

```bash
./run_pipeline.sh --background          # full
./run_pipeline.sh -b quick              # ή: --background quick
```

Θα εμφανιστεί το path του log και το PID· `logs/last.pid` κρατάει το τελευταίο. Παρακολούθηση: `tail -f logs/pipeline-....log`.

Uses `python3.11` for the venv when available, otherwise `python3`. Override with e.g. `PYTHON=python3.11 ./run_pipeline.sh`.

Manual run (if you already have a venv):

```bash
source .venv/bin/activate
python main.py
```

Full run is the default when `FASHION_MNIST_QUICK_RUN` is unset or `0`.

### Quick smoke-test mode

```bash
FASHION_MNIST_QUICK_RUN=1 python main.py
# or
./run_pipeline.sh quick
```

Quick mode uses 3 training epochs (instead of 30), 500 clustering / t-SNE / TSNE-train samples (instead of 10 000). The full pipeline completes in roughly a couple of minutes in quick mode.

---

## Dimensionality reduction techniques

| # | Technique | Description |
|---|-----------|-------------|
| 0 | **Raw** | Baseline — normalised pixel values (784 dims), no reduction |
| 1 | **PCA** | Principal Component Analysis → 64 components |
| 2 | **SAE** | Stacked Autoencoder (Dense 784→512→256→64→256→512→784) |
| 3 | **CNN-SAE** | Convolutional Stacked Autoencoder (Conv→Pool→Flatten→64) |
| 4 | **t-SNE** | PCA-50, then **openTSNE** fit on a train subset → 2 dims; test points via `transform` |
| 5 | **UMAP** | Uniform Manifold Approximation and Projection → 64 dims |

> SAE and CNN-SAE use `EarlyStopping(patience=5)` and both train and validation sets during fitting. Only the encoder half is used for downstream clustering.  
> t-SNE uses the `openTSNE` library so the embedding is learned from training data and test pixels are mapped with `transform` (after the same PCA-50 as in training).

---

## Clustering algorithms

| Algorithm | Notes |
|-----------|-------|
| **MiniBatchKMeans** | Scalable k-means; k = 10 |
| **DBSCAN** | Density-based; eps estimated automatically from 90th-percentile 5-NN distance |
| **AgglomerativeClustering** | Hierarchical Ward linkage; k = 10 |
| **GaussianMixture** | GMM with 10 components |
| **Birch** | Balanced Iterative Reducing and Clustering using Hierarchies; k = 10 |

All algorithms except MiniBatchKMeans operate on a PCA-50 projection of the encoded space when the encoded dimensionality exceeds 50, to keep computation tractable.

---

## Evaluation metrics

| Metric | Better when |
|--------|-------------|
| **Calinski–Harabasz index** | Higher |
| **Davies–Bouldin index** | Lower |
| **Silhouette score** | Higher (range −1 to 1) |
| **Adjusted Rand Index** | Higher (uses true labels; range −1 to 1) |

---

## Outputs

After a full run the following files are produced:

| File | Description |
|------|-------------|
| `results.csv` | Full results table (one row per DR × clustering combination) |
| `figures/00_sample_images.png` | One image per class from the test set |
| `figures/pca_explained_variance.png` | Cumulative explained variance curve |
| `figures/sae_reconstructions.png` | SAE original vs reconstructed images |
| `figures/cnnsae_reconstructions.png` | CNN-SAE original vs reconstructed images |
| `figures/umap_2d.png` | 2-D UMAP embedding coloured by true label |
| `figures/scatter_<DR>.png` | 2-D encoded-space scatter for each DR method |
| `figures/cluster_examples_<DR>.png` | Sample clustered images for 4 selected classes |
| `figures/heatmap_silhouette.png` | Silhouette score heatmap (DR × clustering) |

---

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70 000 grayscale 28×28 images across 10 clothing categories.

| Split | Samples |
|-------|---------|
| Train | 54 000 |
| Validation | 6 000 |
| Test | 10 000 |

Loaded via `keras.datasets.fashion_mnist.load_data()`. Pixel values are normalised to [0, 1].
