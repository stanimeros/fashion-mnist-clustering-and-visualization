"""
Save / load trained DR models under saved_models/.

Environment:
  FMNIST_MODEL_DIR      – root directory (default: saved_models)
  FMNIST_SAVE_MODELS    – 1 = save after training (default: 1)
  FMNIST_REUSE_MODELS   – 1 = load if manifest matches (default: 1)
  FMNIST_FORCE_RETRAIN  – 1 = always train, ignore cache (default: 0)
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

MODEL_VERSION = 1


def model_dir() -> Path:
    return Path(os.environ.get("FMNIST_MODEL_DIR", "saved_models"))


def save_enabled() -> bool:
    return os.environ.get("FMNIST_SAVE_MODELS", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def reuse_enabled() -> bool:
    return os.environ.get("FMNIST_REUSE_MODELS", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def force_retrain() -> bool:
    return os.environ.get("FMNIST_FORCE_RETRAIN", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def training_config_snapshot() -> dict:
    import config as cfg

    return {
        "v": MODEL_VERSION,
        "quick_run": cfg.QUICK_RUN,
        "latent_dim": cfg.LATENT_DIM,
        "epochs": cfg.EPOCHS,
        "batch_size": cfg.BATCH_SIZE,
        "tsne_train_samples": cfg.TSNE_TRAIN_SAMPLES,
        "tsne_samples": cfg.TSNE_SAMPLES,
        "random_state": cfg.RANDOM_STATE,
    }


def manifest_path() -> Path:
    return model_dir() / "manifest.json"


def _write_manifest(cfg: dict) -> None:
    model_dir().mkdir(parents=True, exist_ok=True)
    manifest_path().write_text(
        json.dumps({"training_config": cfg}, indent=2),
        encoding="utf-8",
    )


def manifest_matches_disk() -> bool:
    p = manifest_path()
    if not p.is_file():
        return False
    try:
        disk = json.loads(p.read_text(encoding="utf-8"))
        return disk.get("training_config") == training_config_snapshot()
    except (json.JSONDecodeError, OSError):
        return False


def should_try_load() -> bool:
    return reuse_enabled() and not force_retrain() and manifest_matches_disk()


def should_save() -> bool:
    return save_enabled()


def note_saved() -> None:
    _write_manifest(training_config_snapshot())


# ── sklearn PCA ───────────────────────────────────────────────────
def path_pca() -> Path:
    return model_dir() / "pca_latent.joblib"


# ── Keras autoencoders ─────────────────────────────────────────────
def path_sae() -> Path:
    return model_dir() / "sae_autoencoder.keras"


def path_cnnsae() -> Path:
    return model_dir() / "cnnsae_autoencoder.keras"


# ── t-SNE bundle (PCA50 + openTSNE embedding) ─────────────────────
def path_tsne_bundle() -> Path:
    return model_dir() / "tsne_bundle.pkl"


# ── UMAP ──────────────────────────────────────────────────────────
def path_umap() -> Path:
    return model_dir() / "umap_fitted.pkl"


def save_pickle(obj, path: Path) -> None:
    model_dir().mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)
