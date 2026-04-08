"""
Dimensionality-reduction models:
  - PCA
  - Stacked Autoencoder (SAE)
  - Convolutional Stacked Autoencoder (CNN-SAE)
  - t-SNE  (PCA-50 + openTSNE on train subset, transform test)
  - UMAP

Each public function returns (encoder_fn, train_time, extras).
  encoder_fn(X_flat, X_cnn) -> np.ndarray of encoded test features
  extras: dict with any model-specific info (recon function, etc.)
"""
import time
import warnings
import numpy as np
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

import umap
from openTSNE import TSNE as OpenTSNE

warnings.filterwarnings('ignore')

from config import RANDOM_STATE, LATENT_DIM, EPOCHS, BATCH_SIZE


# ──────────────────────────────────────────────────────────────────
# PCA
# ──────────────────────────────────────────────────────────────────
def build_pca(x_train_flat):
    t0  = time.time()
    pca = PCA(n_components=LATENT_DIM, random_state=RANDOM_STATE)
    pca.fit(x_train_flat)
    train_time = time.time() - t0

    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA fit done ({train_time:.1f}s) | explained variance: {explained:.3f}")

    def encode(x_flat, _x_cnn=None):
        return pca.transform(x_flat)

    return encode, train_time, {"pca": pca, "explained": explained}


# ──────────────────────────────────────────────────────────────────
# Stacked Autoencoder (SAE)
# ──────────────────────────────────────────────────────────────────
def build_sae(x_train_flat, x_val_flat):
    inp = keras.Input(shape=(784,))
    e   = layers.Dense(512, activation="relu")(inp)
    e   = layers.BatchNormalization()(e)
    e   = layers.Dense(256, activation="relu")(e)
    e   = layers.BatchNormalization()(e)
    enc = layers.Dense(LATENT_DIM, activation="relu", name="latent")(e)

    d   = layers.Dense(256, activation="relu")(enc)
    d   = layers.BatchNormalization()(d)
    d   = layers.Dense(512, activation="relu")(d)
    d   = layers.BatchNormalization()(d)
    dec = layers.Dense(784, activation="sigmoid")(d)

    autoencoder = Model(inp, dec, name="SAE")
    encoder     = Model(inp, enc, name="SAE_encoder")
    autoencoder.compile(optimizer="adam", loss="mse")

    t0 = time.time()
    autoencoder.fit(
        x_train_flat, x_train_flat,
        validation_data=(x_val_flat, x_val_flat),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0,
    )
    train_time = time.time() - t0
    print(f"  SAE fit done ({train_time:.1f}s)")

    def encode(x_flat, _x_cnn=None):
        return encoder.predict(x_flat, verbose=0)

    def reconstruct(x_flat):
        return autoencoder.predict(x_flat, verbose=0).reshape(-1, 28, 28)

    return encode, train_time, {"reconstruct": reconstruct}


# ──────────────────────────────────────────────────────────────────
# Convolutional SAE (CNN-SAE)
# ──────────────────────────────────────────────────────────────────
def build_cnn_sae(x_train_cnn, x_val_cnn):
    inp = keras.Input(shape=(28, 28, 1))

    e   = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    e   = layers.MaxPooling2D(2)(e)                       # 14×14×32
    e   = layers.Conv2D(64, 3, activation="relu", padding="same")(e)
    e   = layers.MaxPooling2D(2)(e)                       # 7×7×64
    e   = layers.Flatten()(e)
    enc = layers.Dense(LATENT_DIM, activation="relu", name="latent")(e)

    d   = layers.Dense(7 * 7 * 64, activation="relu")(enc)
    d   = layers.Reshape((7, 7, 64))(d)
    d   = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(d)
    d   = layers.UpSampling2D(2)(d)                       # 14×14×64
    d   = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(d)
    d   = layers.UpSampling2D(2)(d)                       # 28×28×32
    dec = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(d)

    autoencoder = Model(inp, dec, name="CNN_SAE")
    encoder     = Model(inp, enc, name="CNN_encoder")
    autoencoder.compile(optimizer="adam", loss="mse")

    t0 = time.time()
    autoencoder.fit(
        x_train_cnn, x_train_cnn,
        validation_data=(x_val_cnn, x_val_cnn),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0,
    )
    train_time = time.time() - t0
    print(f"  CNN-SAE fit done ({train_time:.1f}s)")

    def encode(_x_flat, x_cnn):
        return encoder.predict(x_cnn, verbose=0)

    def reconstruct(x_cnn):
        return autoencoder.predict(x_cnn, verbose=0).squeeze()

    return encode, train_time, {"reconstruct": reconstruct}


# ──────────────────────────────────────────────────────────────────
# t-SNE  (PCA-50 on train; openTSNE fit on train subset, transform test)
# ──────────────────────────────────────────────────────────────────
def build_tsne(x_train_flat, x_test_flat, tsne_samples, tsne_train_samples):
    t0 = time.time()
    pca50 = PCA(n_components=50, random_state=RANDOM_STATE)
    pca50.fit(x_train_flat)
    pca_time = time.time() - t0

    rng = np.random.default_rng(RANDOM_STATE)
    n_train = min(tsne_train_samples, len(x_train_flat))
    train_idx = rng.choice(len(x_train_flat), size=n_train, replace=False)
    X_tr_pca = pca50.transform(x_train_flat[train_idx])

    perplexity = min(30, max(5, (n_train - 1) // 3))

    print(
        f"  openTSNE: fitting on {n_train} train points (perplexity={perplexity}) …",
        end=" ",
        flush=True,
    )
    t0 = time.time()
    tsne = OpenTSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    embedding = tsne.fit(X_tr_pca)
    tsne_fit_time = time.time() - t0
    print(f"done ({tsne_fit_time:.1f}s)")

    t0 = time.time()
    X_test_pca = pca50.transform(x_test_flat[:tsne_samples])
    X_enc = np.asarray(embedding.transform(X_test_pca), dtype=np.float32)
    transform_time = time.time() - t0
    print(f"  transform(test, n={tsne_samples}): {transform_time:.2f}s")

    train_time = pca_time + tsne_fit_time
    return X_enc, train_time, {"pca50": pca50, "transform_s": transform_time}


# ──────────────────────────────────────────────────────────────────
# UMAP
# ──────────────────────────────────────────────────────────────────
def build_umap(x_train_flat):
    t0    = time.time()
    model = umap.UMAP(
        n_components=LATENT_DIM, n_neighbors=15, min_dist=0.1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train_flat)
    train_time = time.time() - t0
    print(f"  UMAP fit done ({train_time:.1f}s)")

    def encode(x_flat, _x_cnn=None):
        return model.transform(x_flat)

    return encode, train_time, {"umap_model": model}
