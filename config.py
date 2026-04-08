import os

# ── Toggle this to True for a fast end-to-end smoke-test ──────────
QUICK_RUN = False

RANDOM_STATE = 42
N_CLUSTERS   = 10        # fashion-mnist has 10 classes
LATENT_DIM   = 64        # shared bottleneck size for autoencoders

# Training
BATCH_SIZE   = 256
EPOCHS       = 3  if QUICK_RUN else 30
# Subset sizes fed to clustering (test set is already 10 k; subsample for speed)
CLUSTER_SAMPLES = 500   if QUICK_RUN else 10_000   # how many test samples to cluster
TSNE_SAMPLES    = 500   if QUICK_RUN else 10_000   # test points to embed & cluster
# openTSNE is fit on a train subset; transform maps held-out test PCA features
TSNE_TRAIN_SAMPLES = 500 if QUICK_RUN else 10_000

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
]
SELECTED_CLASSES = [0, 2, 5, 9]   # classes shown in cluster-example plots

os.makedirs("figures", exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
