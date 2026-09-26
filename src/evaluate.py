"""
F0.5 threshold tuning and evaluation utilities.
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score
from src.config import THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP


def f05_score(precision: float, recall: float) -> float:
    """Compute F0.5 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = None,
    progress: bool = True,
) -> dict:
    """
    Search for the threshold that maximizes F0.5.

    Returns dict with:
        best_threshold, best_f05, best_precision, best_recall,
        all_results (list of dicts per threshold)
    """
    if thresholds is None:
        thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)

    best = {"threshold": 0.5, "f05": 0.0, "precision": 0.0, "recall": 0.0}
    all_results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        n_pred = y_pred.sum()

        if n_pred == 0:
            p, r, f = 0.0, 0.0, 0.0
        else:
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f = f05_score(p, r)

        result = {"threshold": round(t, 3), "precision": p, "recall": r, "f05": f, "n_predicted": int(n_pred)}
        all_results.append(result)

        if f > best["f05"]:
            best = {"threshold": round(t, 3), "f05": f, "precision": p, "recall": r, "n_predicted": int(n_pred)}

    if progress:
        print(f"\n  Best threshold: {best['threshold']:.3f}")
        print(f"  F0.5:      {best['f05']:.4f}")
        print(f"  Precision: {best['precision']:.4f}")
        print(f"  Recall:    {best['recall']:.4f}")
        print(f"  Predicted: {best.get('n_predicted', 0):,}")

    return {"best": best, "all_results": all_results}


def per_entity_f05(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    pair_ids: list[tuple[str, str]],
    threshold: float,
) -> dict:
    """
    Compute the competition-style per-Source-1 macro-averaged F0.5.

    Groups predictions by S1 entity, computes F0.5 per entity, then averages.
    Singletons (no GT matches) score 1.0 if predicted empty, 0.0 otherwise.
    """
    y_pred = (y_proba >= threshold).astype(int)

    # Group by S1 entity
    entity_preds = {}   # s1_id -> set of predicted candidate_ids
    entity_truth = {}   # s1_id -> set of true candidate_ids

    for i, (s1_id, cand_id) in enumerate(pair_ids):
        if s1_id not in entity_preds:
            entity_preds[s1_id] = set()
            entity_truth[s1_id] = set()

        if y_pred[i] == 1:
            entity_preds[s1_id].add(cand_id)
        if y_true[i] == 1:
            entity_truth[s1_id].add(cand_id)

    f05_scores = []
    for s1_id in entity_preds:
        pred_set = entity_preds[s1_id]
        true_set = entity_truth[s1_id]

        if len(true_set) == 0:
            # Singleton: 1.0 if no predictions, 0.0 otherwise
            f05_scores.append(1.0 if len(pred_set) == 0 else 0.0)
        elif len(pred_set) == 0:
            f05_scores.append(0.0)
        else:
            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f05_scores.append(f05_score(p, r))

    macro_f05 = np.mean(f05_scores) if f05_scores else 0.0

    return {
        "macro_f05": macro_f05,
        "n_entities": len(f05_scores),
        "threshold": threshold,
    }


def evaluate_model(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_pair_ids: list,
    model_name: str = "Model",
    progress: bool = True,
) -> dict:
    """
    Full evaluation: threshold tuning + per-entity F0.5.
    """
    y_proba = model.predict_proba(X_val)[:, 1]

    if progress:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")

    # Pair-level threshold tuning
    threshold_results = find_best_threshold(y_val, y_proba, progress=progress)
    best_t = threshold_results["best"]["threshold"]

    # Per-entity F0.5
    entity_results = per_entity_f05(y_val, y_proba, val_pair_ids, best_t)

    if progress:
        print(f"\n  Per-entity macro F0.5 at threshold {best_t:.3f}: {entity_results['macro_f05']:.4f}")
        print(f"  Entities evaluated: {entity_results['n_entities']:,}")

    return {
        "model_name": model_name,
        "threshold_results": threshold_results,
        "entity_results": entity_results,
        "y_proba": y_proba,
        "best_threshold": best_t,
    }


def error_analysis(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_pair_ids: list,
    conn,
    threshold: float,
    max_errors: int = 50,
    progress: bool = True,
) -> dict:
    """
    Produce error analysis output for false positives and false negatives.
    """
    from src.candidate_generation import fetch_candidate_details
    from src.features import FEATURE_NAMES

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    false_positives = []
    false_negatives = []

    for i in range(len(y_val)):
        s1_id, cand_id = val_pair_ids[i]

        if y_pred[i] == 1 and y_val[i] == 0:
            false_positives.append({
                "s1_id": s1_id, "cand_id": cand_id,
                "proba": float(y_proba[i]),
                "features": {fn: float(X_val[i, j]) for j, fn in enumerate(FEATURE_NAMES)},
            })
        elif y_pred[i] == 0 and y_val[i] == 1:
            false_negatives.append({
                "s1_id": s1_id, "cand_id": cand_id,
                "proba": float(y_proba[i]),
                "features": {fn: float(X_val[i, j]) for j, fn in enumerate(FEATURE_NAMES)},
            })

    # Enrich with entity details
    all_error_ids = set()
    for err in false_positives[:max_errors] + false_negatives[:max_errors]:
        all_error_ids.add(err["s1_id"])
        all_error_ids.add(err["cand_id"])

    # Fetch details
    s1_ids_needed = [eid for eid in all_error_ids if eid.startswith("S1")]
    cand_ids_needed = [eid for eid in all_error_ids if not eid.startswith("S1")]

    cand_details = fetch_candidate_details(conn, cand_ids_needed) if cand_ids_needed else {}

    # Fetch S1 details
    s1_details = {}
    for chunk_start in range(0, len(s1_ids_needed), 500):
        chunk = s1_ids_needed[chunk_start:chunk_start + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT entity_id, business_name, business_address, country FROM source1 "
            f"WHERE entity_id IN ({placeholders})", chunk
        ).fetchall()
        for row in rows:
            s1_details[row[0]] = {
                "entity_id": row[0], "business_name": row[1],
                "business_address": row[2], "country": row[3],
            }

    def enrich(err):
        s1 = s1_details.get(err["s1_id"], {})
        cand = cand_details.get(err["cand_id"], {})
        err["s1_name"] = s1.get("business_name", "")
        err["s1_address"] = s1.get("business_address", "")
        err["s1_country"] = s1.get("country", "")
        err["cand_name"] = cand.get("business_name", "")
        err["cand_address"] = cand.get("business_address", "")
        err["cand_country"] = cand.get("country", "")
        return err

    fps = [enrich(e) for e in false_positives[:max_errors]]
    fns = [enrich(e) for e in false_negatives[:max_errors]]

    if progress:
        print(f"\n  False Positives: {len(false_positives):,} (showing {len(fps)})")
        print(f"  False Negatives: {len(false_negatives):,} (showing {len(fns)})")

    return {"false_positives": fps, "false_negatives": fns}
