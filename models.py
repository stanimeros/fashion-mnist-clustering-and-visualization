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

Trained artefacts can be saved under saved_models/ (see persist.py).
"""
import time
import warnings
import joblib
import numpy as np
from sklearn.decomposition import PCA

from tensorflow import keras
from tensorflow.keras import layers, Model

import umap
from openTSNE import TSNE as OpenTSNE

warnings.filterwarnings("ignore")

from config import RANDOM_STATE, LATENT_DIM, EPOCHS, BATCH_SIZE
from persist import (
    load_pickle,
    note_saved,
    path_cnnsae,
    path_pca,
    path_sae,
    path_tsne_bundle,
    path_umap,
    save_pickle,
    should_save,
    should_try_load,
)


def _load_keras_ae(path, name_for_print):
    """Load full autoencoder; return (autoencoder, encoder) with latent layer 'latent'."""
    try:
        ae = keras.models.load_model(path, compile=False)
    except Exception as exc:
        raise OSError(f"Failed to load {path}: {exc}") from exc
    latent = ae.get_layer("latent")
    encoder = Model(inputs=ae.input, outputs=latent.output, name=f"{name_for_print}_encoder")
    return ae, encoder


# ──────────────────────────────────────────────────────────────────
# PCA
# ──────────────────────────────────────────────────────────────────
def build_pca(x_train_flat):
    pca = None
    train_time = 0.0

    if should_try_load() and path_pca().is_file():
        try:
            pca = joblib.load(path_pca())
            print(f"  PCA loaded from {path_pca()} (no training)")
        except Exception as exc:
            print(f"  PCA load failed ({exc}), refitting …")
            pca = None

    if pca is None:
        t0 = time.time()
        pca = PCA(n_components=LATENT_DIM, random_state=RANDOM_STATE)
        pca.fit(x_train_flat)
        train_time = time.time() - t0
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA fit done ({train_time:.1f}s) | explained variance: {explained:.3f}")
        if should_save():
            joblib.dump(pca, path_pca())
            note_saved()
            print(f"  Saved: {path_pca()}")
    else:
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA explained variance (cached): {explained:.3f}")

    def encode(x_flat, _x_cnn=None):
        return pca.transform(x_flat)

    return encode, train_time, {"pca": pca, "explained": explained}


# ──────────────────────────────────────────────────────────────────
# Stacked Autoencoder (SAE)
# ──────────────────────────────────────────────────────────────────
def build_sae(x_train_flat, x_val_flat):
    autoencoder = None
    encoder = None
    train_time = 0.0

    if should_try_load() and path_sae().is_file():
        try:
            autoencoder, encoder = _load_keras_ae(path_sae(), "SAE")
            print(f"  SAE loaded from {path_sae()} (no training)")
        except Exception as exc:
            print(f"  SAE load failed ({exc}), refitting …")
            autoencoder = None
            encoder = None

    if autoencoder is None:
        inp = keras.Input(shape=(784,))
        e = layers.Dense(512, activation="relu")(inp)
        e = layers.BatchNormalization()(e)
        e = layers.Dense(256, activation="relu")(e)
        e = layers.BatchNormalization()(e)
        enc = layers.Dense(LATENT_DIM, activation="relu", name="latent")(e)

        d = layers.Dense(256, activation="relu")(enc)
        d = layers.BatchNormalization()(d)
        d = layers.Dense(512, activation="relu")(d)
        d = layers.BatchNormalization()(d)
        dec = layers.Dense(784, activation="sigmoid")(d)

        autoencoder = Model(inp, dec, name="SAE")
        encoder = Model(inp, enc, name="SAE_encoder")
        autoencoder.compile(optimizer="adam", loss="mse")

        t0 = time.time()
        autoencoder.fit(
            x_train_flat,
            x_train_flat,
            validation_data=(x_val_flat, x_val_flat),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=0,
        )
        train_time = time.time() - t0
        print(f"  SAE fit done ({train_time:.1f}s)")
        if should_save():
            autoencoder.save(path_sae())
            note_saved()
            print(f"  Saved: {path_sae()}")

    def encode(x_flat, _x_cnn=None):
        return encoder.predict(x_flat, verbose=0)

    def reconstruct(x_flat):
        return autoencoder.predict(x_flat, verbose=0).reshape(-1, 28, 28)

    return encode, train_time, {"reconstruct": reconstruct}


# ──────────────────────────────────────────────────────────────────
# Convolutional SAE (CNN-SAE)
# ──────────────────────────────────────────────────────────────────
def build_cnn_sae(x_train_cnn, x_val_cnn):
    autoencoder = None
    encoder = None
    train_time = 0.0

    if should_try_load() and path_cnnsae().is_file():
        try:
            autoencoder, encoder = _load_keras_ae(path_cnnsae(), "CNN_SAE")
            print(f"  CNN-SAE loaded from {path_cnnsae()} (no training)")
        except Exception as exc:
            print(f"  CNN-SAE load failed ({exc}), refitting …")
            autoencoder = None
            encoder = None

    if autoencoder is None:
        inp = keras.Input(shape=(28, 28, 1))

        e = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
        e = layers.MaxPooling2D(2)(e)
        e = layers.Conv2D(64, 3, activation="relu", padding="same")(e)
        e = layers.MaxPooling2D(2)(e)
        e = layers.Flatten()(e)
        enc = layers.Dense(LATENT_DIM, activation="relu", name="latent")(e)

        d = layers.Dense(7 * 7 * 64, activation="relu")(enc)
        d = layers.Reshape((7, 7, 64))(d)
        d = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(d)
        d = layers.UpSampling2D(2)(d)
        d = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(d)
        d = layers.UpSampling2D(2)(d)
        dec = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(d)

        autoencoder = Model(inp, dec, name="CNN_SAE")
        encoder = Model(inp, enc, name="CNN_encoder")
        autoencoder.compile(optimizer="adam", loss="mse")

        t0 = time.time()
        autoencoder.fit(
            x_train_cnn,
            x_train_cnn,
            validation_data=(x_val_cnn, x_val_cnn),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=0,
        )
        train_time = time.time() - t0
        print(f"  CNN-SAE fit done ({train_time:.1f}s)")
        if should_save():
            autoencoder.save(path_cnnsae())
            note_saved()
            print(f"  Saved: {path_cnnsae()}")

    def encode(_x_flat, x_cnn):
        return encoder.predict(x_cnn, verbose=0)

    def reconstruct(x_cnn):
        return autoencoder.predict(x_cnn, verbose=0).squeeze()

    return encode, train_time, {"reconstruct": reconstruct}


# ──────────────────────────────────────────────────────────────────
# t-SNE  (PCA-50 on train; openTSNE fit on train subset, transform test)
# ──────────────────────────────────────────────────────────────────
def build_tsne(x_train_flat, x_test_flat, tsne_samples, tsne_train_samples):
    pca50 = None
    embedding = None
    pca_time = 0.0
    tsne_fit_time = 0.0

    if should_try_load() and path_tsne_bundle().is_file():
        try:
            bundle = load_pickle(path_tsne_bundle())
            pca50 = bundle["pca50"]
            embedding = bundle["embedding"]
            print(f"  t-SNE bundle loaded from {path_tsne_bundle()} (no training)")
        except Exception as exc:
            print(f"  t-SNE load failed ({exc}), refitting …")
            pca50 = None
            embedding = None

    if pca50 is None or embedding is None:
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

        if should_save():
            save_pickle({"pca50": pca50, "embedding": embedding}, path_tsne_bundle())
            note_saved()
            print(f"  Saved: {path_tsne_bundle()}")

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
    model = None
    train_time = 0.0

    if should_try_load() and path_umap().is_file():
        try:
            model = load_pickle(path_umap())
            print(f"  UMAP loaded from {path_umap()} (no training)")
        except Exception as exc:
            print(f"  UMAP load failed ({exc}), refitting …")
            model = None

    if model is None:
        t0 = time.time()
        model = umap.UMAP(
            n_components=LATENT_DIM,
            n_neighbors=15,
            min_dist=0.1,
            random_state=RANDOM_STATE,
        )
        model.fit(x_train_flat)
        train_time = time.time() - t0
        print(f"  UMAP fit done ({train_time:.1f}s)")
        if should_save():
            save_pickle(model, path_umap())
            note_saved()
            print(f"  Saved: {path_umap()}")

    def encode(x_flat, _x_cnn=None):
        return model.transform(x_flat)

    return encode, train_time, {"umap_model": model}
