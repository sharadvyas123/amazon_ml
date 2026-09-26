"""
XGBoost training for entity resolution.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from src.config import XGB_PARAMS, MODEL_DIR, RANDOM_STATE
from src.features import FEATURE_NAMES


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict = None,
    progress: bool = True,
) -> XGBClassifier:
    """
    Train an XGBoost binary classifier.

    Uses early stopping on validation set.
    """
    p = dict(XGB_PARAMS)
    if params:
        p.update(params)

    # Pull out n_estimators so we can use it for early stopping rounds
    n_estimators = p.pop("n_estimators", 1000)
    eval_metric = p.pop("eval_metric", "logloss")

    if progress:
        print("\n=== Training XGBoost ===")
        print(f"  Train: {X_train.shape[0]:,} samples")
        print(f"  Val:   {X_val.shape[0]:,} samples")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  n_estimators: {n_estimators}")

    model = XGBClassifier(
        n_estimators=n_estimators,
        eval_metric=eval_metric,
        **p,
    )

    # Try early stopping
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100 if progress else 0,
            early_stopping_rounds=50,
        )
    except TypeError:
        # Fallback for versions with different API
        try:
            model.set_params(early_stopping_rounds=50)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=100 if progress else 0,
            )
        except Exception:
            model.fit(X_train, y_train)

    if progress:
        best_iter = getattr(model, "best_iteration", n_estimators)
        print(f"  Best iteration: {best_iter}")

    return model


def save_xgboost(model: XGBClassifier, save_dir: Path = None, prefix: str = "xgb"):
    """Save model, feature names, and config."""
    save_dir = save_dir or MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{prefix}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "feature_names": FEATURE_NAMES,
        "params": model.get_params(),
        "best_iteration": getattr(model, "best_iteration", None),
    }
    meta_path = save_dir / f"{prefix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"  Saved model to {model_path}")
    print(f"  Saved metadata to {meta_path}")

    return model_path


def load_xgboost(save_dir: Path = None, prefix: str = "xgb") -> XGBClassifier:
    """Load a saved XGBoost model."""
    save_dir = save_dir or MODEL_DIR
    model_path = save_dir / f"{prefix}_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)
