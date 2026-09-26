"""
Training data generation: positive/negative pairs with entity-level split.

Splits by Source-1 entity to prevent leakage.
Negative sampling is controlled and configurable.
"""
import sqlite3
import random
from typing import Optional

import numpy as np

from src.config import (
    RANDOM_STATE, NEGATIVE_TO_POSITIVE_RATIO, VAL_FRACTION,
    BATCH_SIZE, SQL_IN_CHUNK,
)
from src.candidate_generation import (
    fetch_s1_blocking_data, generate_candidates_for_entity,
    fetch_candidate_details, measure_candidate_recall,
)
from src.features import compute_features, NUM_FEATURES, FEATURE_NAMES


def get_s1_train_val_split(
    conn: sqlite3.Connection,
    val_fraction: float = VAL_FRACTION,
    random_state: int = RANDOM_STATE,
) -> tuple[list[str], list[str]]:
    """
    Split S1 entity_ids into train/val sets at the entity level.
    Ensures no S1 entity appears in both sets.
    """
    all_s1 = [row[0] for row in conn.execute("SELECT entity_id FROM source1;").fetchall()]
    rng = random.Random(random_state)
    rng.shuffle(all_s1)

    n_val = int(len(all_s1) * val_fraction)
    val_ids = all_s1[:n_val]
    train_ids = all_s1[n_val:]

    # Verify no overlap
    assert len(set(train_ids) & set(val_ids)) == 0, "Train/Val S1 overlap detected!"

    return train_ids, val_ids


def _get_positive_pairs(
    conn: sqlite3.Connection,
    s1_ids: list[str],
) -> list[tuple[str, str]]:
    """Get all ground-truth positive pairs for given S1 IDs."""
    positives = []
    for chunk_start in range(0, len(s1_ids), SQL_IN_CHUNK):
        chunk = s1_ids[chunk_start:chunk_start + SQL_IN_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        sql = f"""
            SELECT source1_entity_id, candidate_entity_id
            FROM ground_truth_pairs
            WHERE source1_entity_id IN ({placeholders}) AND label = 1
        """
        rows = conn.execute(sql, chunk).fetchall()
        positives.extend(rows)
    return positives


def generate_training_data(
    conn: sqlite3.Connection,
    s1_ids: list[str],
    neg_ratio: float = NEGATIVE_TO_POSITIVE_RATIO,
    random_state: int = RANDOM_STATE,
    progress: bool = True,
    max_s1: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """
    Generate training feature matrix, labels, and pair IDs.

    Pipeline per S1 entity:
      1. Generate candidates via blocking
      2. Look up ground-truth positives
      3. Keep ALL positives that appear in candidates
      4. Sample negatives from candidates that are NOT positive
      5. Compute features for selected pairs

    Args:
        conn: SQLite connection
        s1_ids: S1 entity IDs to process
        neg_ratio: number of negatives per positive
        random_state: for reproducible negative sampling
        progress: print progress
        max_s1: optional limit on number of S1 entities (for smoke tests)

    Returns:
        X: feature matrix (n_pairs, n_features)
        y: labels (n_pairs,)
        pair_ids: list of (s1_id, cand_id) tuples
    """
    rng = random.Random(random_state)

    if max_s1 is not None:
        s1_ids = s1_ids[:max_s1]

    all_X = []
    all_y = []
    all_pair_ids = []

    total_positives_found = 0
    total_negatives_sampled = 0
    total_candidates = 0

    n_s1 = len(s1_ids)
    batch_size = BATCH_SIZE

    for batch_start in range(0, n_s1, batch_size):
        batch_ids = s1_ids[batch_start:batch_start + batch_size]

        # Fetch S1 blocking data
        s1_data = fetch_s1_blocking_data(conn, s1_ids=batch_ids)
        s1_lookup = {d["entity_id"]: d for d in s1_data}

        # Get ground truth for this batch
        gt_positives = _get_positive_pairs(conn, batch_ids)
        gt_by_s1 = {}
        for s1_id, cand_id in gt_positives:
            gt_by_s1.setdefault(s1_id, set()).add(cand_id)

        for s1_id in batch_ids:
            s1_ent = s1_lookup.get(s1_id)
            if s1_ent is None:
                continue

            # Generate candidates
            candidates = generate_candidates_for_entity(
                conn,
                s1_id=s1_ent["entity_id"],
                s1_name_norm=s1_ent.get("name_norm") or "",
                s1_name_translit=s1_ent.get("name_translit") or "",
                s1_name_no_legal=s1_ent.get("name_no_legal") or "",
                s1_name_compact=s1_ent.get("name_compact") or "",
                s1_name_first_last=s1_ent.get("name_first_last") or "",
                s1_addr_norm=s1_ent.get("addr_norm") or "",
                s1_country_norm=s1_ent.get("country_norm") or "",
            )
            total_candidates += len(candidates)

            if not candidates:
                continue

            gt_set = gt_by_s1.get(s1_id, set())

            # Separate positives (in candidates) and negatives
            positive_cands = [c for c in candidates if c in gt_set]
            negative_cands = [c for c in candidates if c not in gt_set]

            # Sample negatives
            n_pos = len(positive_cands)
            n_neg_wanted = max(int(n_pos * neg_ratio), 1) if n_pos > 0 else 0

            # Even if no positives, sample a few negatives for balance
            if n_pos == 0 and negative_cands:
                n_neg_wanted = min(1, len(negative_cands))

            if len(negative_cands) > n_neg_wanted:
                negative_cands = rng.sample(negative_cands, n_neg_wanted)

            selected_cands = positive_cands + negative_cands
            if not selected_cands:
                continue

            # Fetch candidate details
            cand_details = fetch_candidate_details(conn, selected_cands)

            # Compute features
            for cand_id in selected_cands:
                cand_ent = cand_details.get(cand_id)
                if cand_ent is None:
                    continue
                feats = compute_features(s1_ent, cand_ent)
                label = 1 if cand_id in gt_set else 0

                all_X.append(feats)
                all_y.append(label)
                all_pair_ids.append((s1_id, cand_id))

                if label == 1:
                    total_positives_found += 1
                else:
                    total_negatives_sampled += 1

        if progress and (batch_start + batch_size) % (batch_size * 10) == 0:
            processed = min(batch_start + batch_size, n_s1)
            print(f"  Processed {processed:,}/{n_s1:,} S1 entities | "
                  f"Pos: {total_positives_found:,} | Neg: {total_negatives_sampled:,} | "
                  f"Candidates: {total_candidates:,}")

    X = np.vstack(all_X) if all_X else np.empty((0, NUM_FEATURES), dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)

    if progress:
        print(f"\n  Training data generation complete:")
        print(f"    S1 entities processed: {min(len(s1_ids), max_s1 or len(s1_ids)):,}")
        print(f"    Total pairs: {len(all_pair_ids):,}")
        print(f"    Positives: {total_positives_found:,}")
        print(f"    Negatives: {total_negatives_sampled:,}")
        print(f"    Total candidates generated: {total_candidates:,}")
        print(f"    Features per pair: {NUM_FEATURES}")

    return X, y, all_pair_ids


def validate_training_data(
    X: np.ndarray,
    y: np.ndarray,
    pair_ids: list,
    train_s1: list[str],
    val_s1: list[str],
    feature_names: list[str] = FEATURE_NAMES,
):
    """
    Run validation checks on the generated training data.
    Raises AssertionError on critical issues.
    """
    print("\n=== Training Data Validation ===")

    # Check shapes
    assert X.shape[0] == len(y) == len(pair_ids), \
        f"Shape mismatch: X={X.shape[0]}, y={len(y)}, pairs={len(pair_ids)}"
    assert X.shape[1] == len(feature_names), \
        f"Feature count mismatch: X has {X.shape[1]} but {len(feature_names)} names"

    print(f"  Pairs: {len(y):,}")
    print(f"  Positives: {(y == 1).sum():,}")
    print(f"  Negatives: {(y == 0).sum():,}")
    print(f"  Features: {X.shape[1]}")

    # Check for NaN/inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    assert nan_count == 0, f"Feature matrix contains {nan_count} NaN values!"
    assert inf_count == 0, f"Feature matrix contains {inf_count} Inf values!"
    print(f"  NaN/Inf check: PASSED")

    # Check train/val S1 overlap
    train_set = set(train_s1)
    val_set = set(val_s1)
    overlap = train_set & val_set
    assert len(overlap) == 0, f"Train/Val S1 overlap: {len(overlap)} entities!"
    print(f"  Train/Val overlap check: PASSED")

    # Check positive recall
    pos_rate = (y == 1).sum() / len(y) if len(y) > 0 else 0
    print(f"  Positive rate: {pos_rate:.4f}")

    print("  All validation checks PASSED.\n")
