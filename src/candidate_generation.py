"""
Efficient candidate generation via multi-strategy blocking.

Uses SQLite indexes to avoid Cartesian products.
Multiple blocking keys are unioned and deduplicated per S1 entity.
A hard safety cap prevents pathological bucket explosions.
"""
import sqlite3
from typing import Optional

from src.config import MAX_CANDIDATES_PER_S1, SQL_IN_CHUNK, BATCH_SIZE


def _query_candidates_by_key(
    conn: sqlite3.Connection,
    target_table: str,
    country_norm: str,
    key_column: str,
    key_value: str,
    limit: int = MAX_CANDIDATES_PER_S1,
) -> list[str]:
    """
    Retrieve candidate entity_ids from target_table where
    country_norm and key_column match the given values.
    """
    if not key_value:
        return []
    sql = f"""
        SELECT entity_id FROM "{target_table}"
        WHERE country_norm = ? AND "{key_column}" = ?
        LIMIT ?
    """
    return [row[0] for row in conn.execute(sql, (country_norm, key_value, limit)).fetchall()]


def _query_candidates_cross_country(
    conn: sqlite3.Connection,
    target_table: str,
    key_column: str,
    key_value: str,
    limit: int = 50,
) -> list[str]:
    """
    Cross-country blocking: match on key_column ignoring country.
    Used sparingly as a fallback to catch cross-country duplicates.
    """
    if not key_value:
        return []
    sql = f"""
        SELECT entity_id FROM "{target_table}"
        WHERE "{key_column}" = ?
        LIMIT ?
    """
    return [row[0] for row in conn.execute(sql, (key_value, limit)).fetchall()]


def generate_candidates_for_entity(
    conn: sqlite3.Connection,
    s1_id: str,
    s1_name_norm: str,
    s1_name_translit: str,
    s1_name_no_legal: str,
    s1_name_compact: str,
    s1_name_first_last: str,
    s1_addr_norm: str,
    s1_country_norm: str,
    cap: int = MAX_CANDIDATES_PER_S1,
) -> list[str]:
    """
    Generate deduplicated candidate IDs from source2 and source3 for one S1 entity.

    Blocking strategies (unioned):
      1. Exact normalized name match (same country)
      2. Exact transliterated name match (same country)
      3. Name without legal suffix match (same country)
      4. Compact name match (same country) — catches spacing variants
      5. First+last token match (same country)
      6. Country + address anchor (same country, exact addr_norm)
      7. Cross-country transliterated name (no country filter) — small limit
    """
    candidates = set()

    for target_table in ["source2", "source3"]:
        # Strategy 1: exact normalized name
        candidates.update(_query_candidates_by_key(
            conn, target_table, s1_country_norm, "name_norm", s1_name_norm, limit=cap
        ))

        # Strategy 2: transliterated name
        if s1_name_translit and s1_name_translit != s1_name_norm:
            candidates.update(_query_candidates_by_key(
                conn, target_table, s1_country_norm, "name_translit", s1_name_translit, limit=cap
            ))

        # Strategy 3: name without legal suffix
        if s1_name_no_legal and s1_name_no_legal != s1_name_norm:
            candidates.update(_query_candidates_by_key(
                conn, target_table, s1_country_norm, "name_no_legal", s1_name_no_legal, limit=cap
            ))

        # Strategy 4: compact name (no spaces)
        if s1_name_compact:
            candidates.update(_query_candidates_by_key(
                conn, target_table, s1_country_norm, "name_compact", s1_name_compact, limit=cap
            ))

        # Strategy 5: first+last token
        if s1_name_first_last:
            candidates.update(_query_candidates_by_key(
                conn, target_table, s1_country_norm, "name_first_last", s1_name_first_last, limit=cap
            ))

        # Strategy 6: address blocking (same country + exact address)
        if s1_addr_norm and len(s1_addr_norm) >= 8:
            candidates.update(_query_candidates_by_key(
                conn, target_table, s1_country_norm, "addr_norm", s1_addr_norm, limit=50
            ))

        # Strategy 7: cross-country transliterated name (small limit)
        if s1_name_translit:
            candidates.update(_query_candidates_cross_country(
                conn, target_table, "name_translit", s1_name_translit, limit=30
            ))

    # Hard safety cap
    candidate_list = list(candidates)
    if len(candidate_list) > cap:
        candidate_list = candidate_list[:cap]

    return candidate_list


def generate_candidates_batch(
    conn: sqlite3.Connection,
    s1_entities: list[dict],
    progress: bool = True,
) -> dict[str, list[str]]:
    """
    Generate candidates for a batch of S1 entities.

    Args:
        s1_entities: list of dicts with keys:
            entity_id, name_norm, name_translit, name_no_legal,
            name_compact, name_first_last, addr_norm, country_norm
        progress: whether to print progress

    Returns:
        dict mapping s1_id -> list of candidate entity_ids
    """
    results = {}
    for i, ent in enumerate(s1_entities):
        candidates = generate_candidates_for_entity(
            conn,
            s1_id=ent["entity_id"],
            s1_name_norm=ent.get("name_norm", ""),
            s1_name_translit=ent.get("name_translit", ""),
            s1_name_no_legal=ent.get("name_no_legal", ""),
            s1_name_compact=ent.get("name_compact", ""),
            s1_name_first_last=ent.get("name_first_last", ""),
            s1_addr_norm=ent.get("addr_norm", ""),
            s1_country_norm=ent.get("country_norm", ""),
        )
        results[ent["entity_id"]] = candidates

    return results


def fetch_s1_blocking_data(
    conn: sqlite3.Connection,
    s1_ids: Optional[list[str]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[dict]:
    """
    Fetch S1 entities with all their blocking keys.
    If s1_ids is given, fetch only those. Otherwise paginate with limit/offset.
    """
    cols = [
        "entity_id", "business_name", "business_address", "country",
        "name_norm", "name_translit", "name_no_legal", "name_compact",
        "name_first_last", "addr_norm", "country_norm",
    ]
    col_str = ", ".join(f'"{c}"' for c in cols)

    if s1_ids is not None:
        # Batch fetch by IDs
        results = []
        for chunk_start in range(0, len(s1_ids), SQL_IN_CHUNK):
            chunk = s1_ids[chunk_start:chunk_start + SQL_IN_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            sql = f'SELECT {col_str} FROM source1 WHERE entity_id IN ({placeholders})'
            rows = conn.execute(sql, chunk).fetchall()
            results.extend(dict(zip(cols, row)) for row in rows)
        return results
    else:
        sql = f'SELECT {col_str} FROM source1'
        params = []
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = [limit, offset]
        rows = conn.execute(sql, params).fetchall()
        return [dict(zip(cols, row)) for row in rows]


def fetch_candidate_details(
    conn: sqlite3.Connection,
    candidate_ids: list[str],
) -> dict[str, dict]:
    """
    Fetch full details for a list of candidate entity_ids from S2/S3 tables.
    Returns a dict mapping entity_id -> row dict.
    """
    cols = [
        "entity_id", "business_name", "business_address", "country",
        "name_norm", "name_translit", "name_no_legal", "name_compact",
        "addr_norm", "country_norm",
    ]
    col_str = ", ".join(f'"{c}"' for c in cols)

    results = {}

    # Split by source
    s2_ids = [cid for cid in candidate_ids if cid.startswith("S2")]
    s3_ids = [cid for cid in candidate_ids if cid.startswith("S3")]

    for table, ids in [("source2", s2_ids), ("source3", s3_ids)]:
        for chunk_start in range(0, len(ids), SQL_IN_CHUNK):
            chunk = ids[chunk_start:chunk_start + SQL_IN_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            sql = f'SELECT {col_str} FROM "{table}" WHERE entity_id IN ({placeholders})'
            rows = conn.execute(sql, chunk).fetchall()
            for row in rows:
                d = dict(zip(cols, row))
                results[d["entity_id"]] = d

    return results


def measure_candidate_recall(
    conn: sqlite3.Connection,
    candidates_dict: dict[str, list[str]],
    progress: bool = True,
) -> dict:
    """
    Measure how many ground-truth positives are found by the candidate generation.

    Returns dict with:
      total_positives, found_positives, recall, missed_pairs
    """
    total_positives = 0
    found_positives = 0
    missed_pairs = []

    s1_ids = list(candidates_dict.keys())

    for chunk_start in range(0, len(s1_ids), SQL_IN_CHUNK):
        chunk = s1_ids[chunk_start:chunk_start + SQL_IN_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        sql = f"""
            SELECT source1_entity_id, candidate_entity_id
            FROM ground_truth_pairs
            WHERE source1_entity_id IN ({placeholders}) AND label = 1
        """
        rows = conn.execute(sql, chunk).fetchall()
        for s1_id, cand_id in rows:
            total_positives += 1
            if cand_id in set(candidates_dict.get(s1_id, [])):
                found_positives += 1
            else:
                missed_pairs.append((s1_id, cand_id))

    recall = found_positives / total_positives if total_positives > 0 else 0.0

    if progress:
        print(f"  Candidate recall: {found_positives:,}/{total_positives:,} = {recall:.4f}")
        if missed_pairs:
            print(f"  Missed {len(missed_pairs):,} true positive pairs")

    return {
        "total_positives": total_positives,
        "found_positives": found_positives,
        "recall": recall,
        "missed_pairs": missed_pairs[:100],  # keep only a sample
    }
