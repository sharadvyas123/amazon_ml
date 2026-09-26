"""
LightGBM training for entity resolution.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

from src.config import LGBM_PARAMS, MODEL_DIR, RANDOM_STATE
from src.features import FEATURE_NAMES


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict = None,
    progress: bool = True,
) -> LGBMClassifier:
    """
    Train a LightGBM binary classifier.

    Uses early stopping on validation set if supported.
    """
    p = dict(LGBM_PARAMS)
    if params:
        p.update(params)

    if progress:
        print("\n=== Training LightGBM ===")
        print(f"  Train: {X_train.shape[0]:,} samples")
        print(f"  Val:   {X_val.shape[0]:,} samples")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Params: {p}")

    model = LGBMClassifier(**p)

    # Try early stopping
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                __import__("lightgbm").early_stopping(50, verbose=progress),
                __import__("lightgbm").log_evaluation(100 if progress else 0),
            ],
        )
    except TypeError:
        # Fallback for older LightGBM without callbacks API
        model.fit(X_train, y_train)

    if progress:
        best_iter = getattr(model, "best_iteration_", p.get("n_estimators", "N/A"))
        print(f"  Best iteration: {best_iter}")

    return model


def save_lightgbm(model: LGBMClassifier, save_dir: Path = None, prefix: str = "lgbm"):
    """Save model, feature names, and config."""
    save_dir = save_dir or MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{prefix}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "feature_names": FEATURE_NAMES,
        "params": model.get_params(),
        "best_iteration": getattr(model, "best_iteration_", None),
    }
    meta_path = save_dir / f"{prefix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"  Saved model to {model_path}")
    print(f"  Saved metadata to {meta_path}")

    return model_path


def load_lightgbm(save_dir: Path = None, prefix: str = "lgbm") -> LGBMClassifier:
    """Load a saved LightGBM model."""
    save_dir = save_dir or MODEL_DIR
    model_path = save_dir / f"{prefix}_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)
