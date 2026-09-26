"""
Inference pipeline for entity resolution.

Processes test Source-1 entities in batches:
  1. Generate candidates via blocking
  2. Compute features for each (S1, candidate) pair
  3. Score with trained model
  4. Apply threshold
  5. Output competition-ready submission file
"""
import csv
import pickle
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import (
    MODEL_DIR, OUTPUT_DIR, BATCH_SIZE,
    MAX_CANDIDATES_PER_S1,
)
from src.candidate_generation import (
    fetch_s1_blocking_data,
    generate_candidates_for_entity,
    fetch_candidate_details,
)
from src.features import compute_features, NUM_FEATURES


def load_model(model_path: Path):
    """Load a pickled model."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_for_s1_batch(
    conn: sqlite3.Connection,
    model,
    s1_entities: list[dict],
    threshold: float,
) -> dict[str, list[str]]:
    """
    Run full inference for a batch of S1 entities.

    Returns dict mapping s1_id -> list of predicted matching candidate_ids.
    """
    predictions = {}

    for s1_ent in s1_entities:
        s1_id = s1_ent["entity_id"]

        # Generate candidates
        candidates = generate_candidates_for_entity(
            conn,
            s1_id=s1_id,
            s1_name_norm=s1_ent.get("name_norm") or "",
            s1_name_translit=s1_ent.get("name_translit") or "",
            s1_name_no_legal=s1_ent.get("name_no_legal") or "",
            s1_name_compact=s1_ent.get("name_compact") or "",
            s1_name_first_last=s1_ent.get("name_first_last") or "",
            s1_addr_norm=s1_ent.get("addr_norm") or "",
            s1_country_norm=s1_ent.get("country_norm") or "",
        )

        if not candidates:
            predictions[s1_id] = []
            continue

        # Fetch candidate details
        cand_details = fetch_candidate_details(conn, candidates)

        # Compute features and predict
        matched = []
        # Process in sub-batches to avoid huge arrays
        sub_batch_size = 500
        for sub_start in range(0, len(candidates), sub_batch_size):
            sub_cands = candidates[sub_start:sub_start + sub_batch_size]
            X_batch = np.empty((len(sub_cands), NUM_FEATURES), dtype=np.float32)
            valid_indices = []
            valid_cand_ids = []

            for j, cand_id in enumerate(sub_cands):
                cand_ent = cand_details.get(cand_id)
                if cand_ent is None:
                    continue
                X_batch[j] = compute_features(s1_ent, cand_ent)
                valid_indices.append(j)
                valid_cand_ids.append(cand_id)

            if not valid_indices:
                continue

            X_valid = X_batch[valid_indices]
            probas = model.predict_proba(X_valid)[:, 1]

            for k, cand_id in enumerate(valid_cand_ids):
                if probas[k] >= threshold:
                    matched.append(cand_id)

        predictions[s1_id] = matched

    return predictions


def run_inference(
    conn: sqlite3.Connection,
    model,
    threshold: float,
    output_path: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
    progress: bool = True,
    max_s1: Optional[int] = None,
) -> Path:
    """
    Run inference on all test Source-1 entities.

    Processes in batches to limit memory usage.
    Writes a competition-ready CSV submission file.

    Args:
        conn: SQLite connection (must have test data loaded)
        model: trained classifier with predict_proba
        threshold: classification threshold
        output_path: where to write the submission file
        batch_size: S1 entities per batch
        progress: whether to print progress
        max_s1: optional limit for debugging

    Returns:
        Path to the output submission file
    """
    output_path = output_path or (OUTPUT_DIR / "submission.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count total test S1 entities
    total_s1 = conn.execute("SELECT COUNT(*) FROM source1").fetchone()[0]
    if max_s1 is not None:
        total_s1 = min(total_s1, max_s1)

    if progress:
        print(f"\n{'='*60}")
        print(f"Running Inference")
        print(f"{'='*60}")
        print(f"  Total S1 entities: {total_s1:,}")
        print(f"  Threshold: {threshold:.3f}")
        print(f"  Batch size: {batch_size:,}")
        print(f"  Output: {output_path}")

    all_predictions = {}
    processed = 0

    offset = 0
    while processed < total_s1:
        current_batch = min(batch_size, total_s1 - processed)

        # Fetch S1 entities for this batch
        s1_batch = fetch_s1_blocking_data(
            conn, limit=current_batch, offset=offset
        )

        if not s1_batch:
            break

        # Run predictions
        batch_preds = predict_for_s1_batch(conn, model, s1_batch, threshold)
        all_predictions.update(batch_preds)

        processed += len(s1_batch)
        offset += len(s1_batch)

        if progress and processed % 1000 == 0:
            n_matched = sum(1 for v in all_predictions.values() if v)
            print(f"  Processed {processed:,}/{total_s1:,} S1 entities | "
                  f"Matched: {n_matched:,}")

    # Write submission file
    _write_submission(all_predictions, output_path)

    # Summary
    n_with_matches = sum(1 for v in all_predictions.values() if v)
    total_matches = sum(len(v) for v in all_predictions.values())

    if progress:
        print(f"\n  Inference complete.")
        print(f"  S1 entities processed: {processed:,}")
        print(f"  S1 with ≥1 match: {n_with_matches:,}")
        print(f"  Total predicted matches: {total_matches:,}")
        print(f"  Saved to: {output_path}")

    return output_path


def _write_submission(
    predictions: dict[str, list[str]],
    output_path: Path,
):
    """
    Write predictions in the competition submission format.

    Format: source1_entity_id, matched_entity_ids (comma-separated)
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        for s1_id in sorted(predictions.keys()):
            matched = predictions[s1_id]
            matched_str = ", ".join(sorted(matched)) if matched else ""
            writer.writerow([s1_id, matched_str])


def run_inference_on_ids(
    conn: sqlite3.Connection,
    model,
    threshold: float,
    s1_ids: list[str],
    progress: bool = True,
) -> dict[str, list[str]]:
    """
    Run inference on a specific list of S1 entity IDs.

    Useful for validation or partial inference.
    Returns dict mapping s1_id -> list of predicted candidate_ids.
    """
    predictions = {}
    n = len(s1_ids)

    batch_size = BATCH_SIZE
    for batch_start in range(0, n, batch_size):
        batch_ids = s1_ids[batch_start:batch_start + batch_size]

        s1_batch = fetch_s1_blocking_data(conn, s1_ids=batch_ids)
        batch_preds = predict_for_s1_batch(conn, model, s1_batch, threshold)
        predictions.update(batch_preds)

        if progress and (batch_start + batch_size) % 1000 == 0:
            print(f"  Inferred {min(batch_start + batch_size, n):,}/{n:,}")

    return predictions
