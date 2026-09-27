"""
Candidate generation for entity resolution.

IMPORTANT:
Candidate generation is batch-oriented to avoid executing many SQLite
queries independently for every S1 entity.

The database contains:
    source1 ~2.2M rows
    source2 ~5.0M rows
    source3 ~5.3M rows

Therefore candidate generation must rely on indexed blocking columns.
"""

import sqlite3
from typing import Optional

from src.config import (
    MAX_CANDIDATES_PER_S1,
    MIN_TOKEN_OVERLAP,
    SQL_IN_CHUNK,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunked(items, size):
    """Yield lists of at most `size` items."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _add_candidates(result, rows):
    """
    Add candidate IDs to the result dictionary.

    rows must contain:
        source1_entity_id, candidate_entity_id
    """
    for s1_id, cand_id in rows:
        bucket = result.setdefault(s1_id, set())

        # Safety cap.
        if len(bucket) < MAX_CANDIDATES_PER_S1:
            bucket.add(cand_id)


# ---------------------------------------------------------------------------
# S1 blocking data
# ---------------------------------------------------------------------------

def fetch_s1_blocking_data(
    conn: sqlite3.Connection,
    s1_ids: Optional[list[str]] = None,
    limit: Optional[int] = None,
):
    """
    Fetch only the columns required for blocking.

    If s1_ids is provided, fetch those entities.
    Otherwise fetch up to `limit` S1 entities.
    """

    columns = """
        entity_id,
        business_name,
        business_address,
        country,
        country_norm,
        name_norm,
        name_translit,
        name_no_legal,
        name_compact,
        addr_norm,
        name_first_last
    """

    if s1_ids is not None:
        if not s1_ids:
            return []

        result = []

        for chunk in _chunked(s1_ids, SQL_IN_CHUNK):
            placeholders = ",".join("?" for _ in chunk)

            sql = f"""
                SELECT {columns}
                FROM source1
                WHERE entity_id IN ({placeholders})
            """

            result.extend(conn.execute(sql, chunk).fetchall())

        col_names = [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "country_norm",
            "name_norm",
            "name_translit",
            "name_no_legal",
            "name_compact",
            "addr_norm",
            "name_first_last",
        ]

        return [dict(zip(col_names, row)) for row in result]

    sql = f"""
        SELECT {columns}
        FROM source1
        LIMIT ?
    """

    rows = conn.execute(sql, (limit or 1000,)).fetchall()

    col_names = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "country_norm",
        "name_norm",
        "name_translit",
        "name_no_legal",
        "name_compact",
        "addr_norm",
        "name_first_last",
    ]

    return [dict(zip(col_names, row)) for row in rows]


# ---------------------------------------------------------------------------
# Token-overlap blocking (V2)
# ---------------------------------------------------------------------------

def _token_overlap_blocking(
    conn,
    s1_data,
    result,
    min_overlap=MIN_TOKEN_OVERLAP,
):
    """
    Find candidates sharing >= min_overlap name tokens with S1 entities.

    Uses the pre-built ``name_token_index`` table (built by
    ``database.build_token_index``).  Tokens with very high document
    frequency were already excluded during index construction.

    This strategy is inherently cross-country because the token index
    does not store country information.

    Processes S1 entities in sub-batches (1 000 at a time) to keep the
    in-memory token→candidates mapping at a manageable size.
    """

    # Check if the token index exists
    try:
        conn.execute("SELECT 1 FROM name_token_index LIMIT 1")
    except Exception:
        return  # Index not built yet — silently skip

    SUB_BATCH = 1_000

    for sb_start in range(0, len(s1_data), SUB_BATCH):
        sb = s1_data[sb_start : sb_start + SUB_BATCH]

        # Step 1 — collect tokens from this sub-batch
        s1_tokens: dict[str, set[str]] = {}
        all_tokens: set[str] = set()

        for s1 in sb:
            s1_id = s1["entity_id"]

            if len(result.get(s1_id, set())) >= MAX_CANDIDATES_PER_S1:
                continue

            name_norm = s1.get("name_norm") or ""
            if not name_norm:
                continue

            tokens = set(
                t for t in name_norm.split() if len(t) >= 2
            )

            # Only entities with enough tokens can benefit
            if len(tokens) < min_overlap:
                continue

            s1_tokens[s1_id] = tokens
            all_tokens.update(tokens)

        if not all_tokens:
            continue

        # Step 2 — query the token index in SQL_IN_CHUNK-sized chunks
        token_to_cands: dict[str, list[str]] = {}

        for chunk in _chunked(sorted(all_tokens), SQL_IN_CHUNK):
            placeholders = ",".join("?" for _ in chunk)

            rows = conn.execute(
                f"SELECT token, entity_id "
                f"FROM name_token_index "
                f"WHERE token IN ({placeholders})",
                chunk,
            ).fetchall()

            for token, cand_id in rows:
                token_to_cands.setdefault(token, []).append(
                    cand_id
                )

        # Step 3 — count overlaps and select candidates
        for s1 in sb:
            s1_id = s1["entity_id"]
            tokens = s1_tokens.get(s1_id)

            if not tokens:
                continue
            if len(result.get(s1_id, set())) >= MAX_CANDIDATES_PER_S1:
                continue

            cand_counts: dict[str, int] = {}

            for token in tokens:
                for cand_id in token_to_cands.get(token, []):
                    if cand_id != s1_id:
                        cand_counts[cand_id] = (
                            cand_counts.get(cand_id, 0) + 1
                        )

            # Sort by overlap count (best first)
            sorted_cands = sorted(
                cand_counts.items(), key=lambda x: -x[1]
            )

            bucket = result.setdefault(s1_id, set())

            for cand_id, cnt in sorted_cands:
                if cnt >= min_overlap:
                    bucket.add(cand_id)
                    if len(bucket) >= MAX_CANDIDATES_PER_S1:
                        break


# ---------------------------------------------------------------------------
# Cross-country compact blocking (V2)
# ---------------------------------------------------------------------------

def _cross_country_compact_blocking(
    conn,
    s1_data,
    result,
    min_len=5,
):
    """
    Exact blocking on ``name_compact`` and ``name_no_legal`` WITHOUT
    the country constraint.

    Catches matches where the country is missing or coded differently
    between S1 and S2/S3.  Only applies to values with length >= min_len
    to avoid ambiguous short names generating excessive candidates.

    Requires single-column indexes on name_compact and name_no_legal
    (created by ``database.ensure_indexes``).
    """

    blocking_columns = ["name_compact", "name_no_legal"]

    for column in blocking_columns:

        # Collect unique values from this S1 batch
        values: set[str] = set()

        for s1 in s1_data:
            val = (s1.get(column) or "").strip()
            if val and len(val) >= min_len:
                values.add(val)

        if not values:
            continue

        for table in ["source2", "source3"]:

            for chunk in _chunked(sorted(values), SQL_IN_CHUNK):
                placeholders = ",".join("?" for _ in chunk)

                sql = f"""
                    SELECT entity_id, {column}
                    FROM {table}
                    WHERE {column} IN ({placeholders})
                """

                rows = conn.execute(sql, chunk).fetchall()

                lookup: dict[str, list[str]] = {}

                for entity_id, val in rows:
                    lookup.setdefault(
                        val or "", []
                    ).append(entity_id)

                for s1 in s1_data:
                    s1_id = s1["entity_id"]

                    if (
                        len(result.get(s1_id, set()))
                        >= MAX_CANDIDATES_PER_S1
                    ):
                        continue

                    val = (s1.get(column) or "").strip()

                    if not val or len(val) < min_len:
                        continue

                    bucket = result.setdefault(s1_id, set())

                    for cand_id in lookup.get(val, []):
                        if cand_id != s1_id:
                            bucket.add(cand_id)
                            if len(bucket) >= MAX_CANDIDATES_PER_S1:
                                break


# ---------------------------------------------------------------------------
# Batch candidate generation
# ---------------------------------------------------------------------------

def generate_candidates_batch(
    conn: sqlite3.Connection,
    s1_data: list[dict],
    progress: bool = True,
):
    """
    Generate candidates for MANY S1 entities using set-based SQL.

    This is the important performance path.

    Instead of:

        S1 #1 -> SQL
        S1 #2 -> SQL
        S1 #3 -> SQL
        ...
        S1 #5000 -> SQL

    we use blocking keys from the entire batch and perform indexed queries
    against source2/source3 in chunks.
    """

    if not s1_data:
        return {}

    result = {
        row["entity_id"]: set()
        for row in s1_data
    }

    # ------------------------------------------------------------------
    # Build blocking keys.
    # ------------------------------------------------------------------

    blocking_columns = [
        "country_norm",
        "name_norm",
        "name_translit",
        "name_no_legal",
        "name_compact",
        "name_first_last",
        "addr_norm",
    ]

    # For every blocking column, collect unique non-empty keys.
    keys = {
        column: set()
        for column in blocking_columns
    }

    for row in s1_data:
        for column in blocking_columns:
            value = row.get(column)

            if value:
                value = str(value).strip()

                if value:
                    keys[column].add(
                        (
                            row.get("country_norm") or "",
                            value,
                        )
                    )

    # ------------------------------------------------------------------
    # Query source2/source3 using indexed blocking columns.
    # ------------------------------------------------------------------

    sources = ["source2", "source3"]

    for table in sources:

        # --------------------------------------------------------------
        # Exact normalized-name blocking
        # --------------------------------------------------------------

        for column in [
            "name_norm",
            "name_translit",
            "name_no_legal",
            "name_compact",
            "name_first_last",
        ]:

            pairs = list(keys[column])

            if not pairs:
                continue

            for chunk in _chunked(pairs, SQL_IN_CHUNK):

                conditions = []
                params = []

                for country, value in chunk:
                    conditions.append(
                        "(country_norm = ? AND "
                        f"{column} = ?)"
                    )
                    params.extend([country, value])

                where_clause = " OR ".join(conditions)

                sql = f"""
                    SELECT
                        entity_id,
                        country_norm,
                        {column}
                    FROM {table}
                    WHERE {where_clause}
                """

                rows = conn.execute(sql, params).fetchall()

                # Map returned candidates back to S1 entities.
                lookup = {}

                for entity_id, country, value in rows:
                    lookup.setdefault(
                        (country or "", value or ""),
                        []
                    ).append(entity_id)

                for s1 in s1_data:
                    if len(result[s1["entity_id"]]) >= MAX_CANDIDATES_PER_S1:
                        continue

                    country = s1.get("country_norm") or ""
                    value = s1.get(column) or ""

                    if not value:
                        continue

                    for cand_id in lookup.get((country, value), []):
                        if cand_id != s1["entity_id"]:
                            result[s1["entity_id"]].add(cand_id)

                            if (
                                len(result[s1["entity_id"]])
                                >= MAX_CANDIDATES_PER_S1
                            ):
                                break

        # --------------------------------------------------------------
        # Address blocking
        # --------------------------------------------------------------

        pairs = list(keys["addr_norm"])

        if pairs:

            for chunk in _chunked(pairs, SQL_IN_CHUNK):

                conditions = []
                params = []

                for country, value in chunk:
                    conditions.append(
                        "(country_norm = ? AND addr_norm = ?)"
                    )
                    params.extend([country, value])

                where_clause = " OR ".join(conditions)

                sql = f"""
                    SELECT
                        entity_id,
                        country_norm,
                        addr_norm
                    FROM {table}
                    WHERE {where_clause}
                """

                rows = conn.execute(sql, params).fetchall()

                lookup = {}

                for entity_id, country, value in rows:
                    lookup.setdefault(
                        (country or "", value or ""),
                        []
                    ).append(entity_id)

                for s1 in s1_data:

                    s1_id = s1["entity_id"]

                    if len(result[s1_id]) >= MAX_CANDIDATES_PER_S1:
                        continue

                    country = s1.get("country_norm") or ""
                    value = s1.get("addr_norm") or ""

                    if not value:
                        continue

                    for cand_id in lookup.get((country, value), []):

                        if cand_id != s1_id:
                            result[s1_id].add(cand_id)

                        if (
                            len(result[s1_id])
                            >= MAX_CANDIDATES_PER_S1
                        ):
                            break

    # ------------------------------------------------------------------
    # Strategy 2: Cross-country compact blocking [V2]
    # ------------------------------------------------------------------

    _cross_country_compact_blocking(conn, s1_data, result)

    # ------------------------------------------------------------------
    # Strategy 3: Token-overlap blocking [V2]
    # ------------------------------------------------------------------

    _token_overlap_blocking(conn, s1_data, result)

    # ------------------------------------------------------------------
    # Convert sets to lists.
    # ------------------------------------------------------------------

    result = {
        s1_id: list(candidates)
        for s1_id, candidates in result.items()
    }

    if progress:
        n_entities = len(result)
        total = sum(len(v) for v in result.values())

        nonempty = sum(
            1 for v in result.values()
            if v
        )

        print(
            f"Candidate generation batch complete: "
            f"{n_entities:,} S1 entities | "
            f"{nonempty:,} with candidates | "
            f"{total:,} candidates"
        )

    return result


# ---------------------------------------------------------------------------
# Compatibility wrapper
# ---------------------------------------------------------------------------

def generate_candidates_for_entity(
    conn: sqlite3.Connection,
    s1_id: str,
    s1_name_norm: str = "",
    s1_name_translit: str = "",
    s1_name_no_legal: str = "",
    s1_name_compact: str = "",
    s1_name_first_last: str = "",
    s1_addr_norm: str = "",
    s1_country_norm: str = "",
):
    """
    Compatibility wrapper.

    Used only when one entity genuinely needs candidate generation.

    Training should use generate_candidates_batch() instead.
    """

    s1 = {
        "entity_id": s1_id,
        "name_norm": s1_name_norm,
        "name_translit": s1_name_translit,
        "name_no_legal": s1_name_no_legal,
        "name_compact": s1_name_compact,
        "name_first_last": s1_name_first_last,
        "addr_norm": s1_addr_norm,
        "country_norm": s1_country_norm,
    }

    result = generate_candidates_batch(
        conn,
        [s1],
        progress=False,
    )

    return result.get(s1_id, [])


# ---------------------------------------------------------------------------
# Candidate details
# ---------------------------------------------------------------------------

def fetch_candidate_details(
    conn: sqlite3.Connection,
    candidate_ids: list[str],
):
    """
    Fetch candidate entities from source2/source3 in chunks.

    Avoids one SQL query per candidate.
    """

    if not candidate_ids:
        return {}

    result = {}

    s2_ids = [
        x for x in candidate_ids
        if str(x).startswith("S2")
    ]

    s3_ids = [
        x for x in candidate_ids
        if str(x).startswith("S3")
    ]

    columns = """
        entity_id,
        business_name,
        business_address,
        country,
        country_norm,
        name_norm,
        name_translit,
        name_no_legal,
        name_compact,
        addr_norm,
        name_first_last
    """

    col_names = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "country_norm",
        "name_norm",
        "name_translit",
        "name_no_legal",
        "name_compact",
        "addr_norm",
        "name_first_last",
    ]

    for table, ids in [
        ("source2", s2_ids),
        ("source3", s3_ids),
    ]:

        for chunk in _chunked(ids, SQL_IN_CHUNK):

            placeholders = ",".join("?" for _ in chunk)

            sql = f"""
                SELECT {columns}
                FROM {table}
                WHERE entity_id IN ({placeholders})
            """

            rows = conn.execute(sql, chunk).fetchall()

            for row in rows:
                entity = dict(zip(col_names, row))
                result[entity["entity_id"]] = entity

    return result


# ---------------------------------------------------------------------------
# Candidate recall
# ---------------------------------------------------------------------------

def measure_candidate_recall(
    conn: sqlite3.Connection,
    candidates_dict: dict,
    progress: bool = True,
):
    """
    Measure how many ground-truth matches appear in generated candidates.

    Uses SQL in batches instead of querying ground truth once per S1.
    """

    s1_ids = list(candidates_dict.keys())

    if not s1_ids:
        return {
            "recall": 0.0,
            "total_positive_pairs": 0,
            "found_positive_pairs": 0,
        }

    total_positive = 0
    found_positive = 0

    for chunk in _chunked(s1_ids, SQL_IN_CHUNK):

        placeholders = ",".join("?" for _ in chunk)

        sql = f"""
            SELECT source1_entity_id, candidate_entity_id
            FROM ground_truth_pairs
            WHERE source1_entity_id IN ({placeholders})
        """

        rows = conn.execute(sql, chunk).fetchall()

        gt = {}

        for s1_id, cand_id in rows:
            gt.setdefault(s1_id, set()).add(cand_id)

        for s1_id in chunk:

            true_candidates = gt.get(s1_id, set())
            generated = set(candidates_dict.get(s1_id, []))

            total_positive += len(true_candidates)
            found_positive += len(
                true_candidates & generated
            )

    recall = (
        found_positive / total_positive
        if total_positive
        else 0.0
    )

    if progress:
        print("\n=== Candidate Recall ===")
        print(f"Ground-truth positives : {total_positive:,}")
        print(f"Found in candidates     : {found_positive:,}")
        print(f"Candidate recall        : {recall:.4%}")

    return {
        "recall": recall,
        "total_positive_pairs": total_positive,
        "found_positive_pairs": found_positive,
    }