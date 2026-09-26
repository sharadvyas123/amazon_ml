"""
SQLite database utilities for entity resolution.

Reuses the existing database/amazon_ml.db and adds normalized columns
and blocking indexes if they don't already exist.
"""
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DB_PATH, DB_DIR
from src.normalize import (
    normalize_name, normalize_name_translit, normalize_name_no_legal,
    normalize_name_compact, normalize_address, normalize_country,
    first_last_tokens,
)


def get_connection(db_path: Optional[Path] = None, wal: bool = True) -> sqlite3.Connection:
    """Open a connection to the SQLite database with sensible pragmas."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60)
    conn.execute("PRAGMA journal_mode = WAL;")
    if wal:
        conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.execute("PRAGMA cache_size = -64000;")  # 64 MB cache
    return conn


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a table already has a given column."""
    cols = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}");').fetchall()}
    return column in cols


def _register_udfs(conn: sqlite3.Connection):
    """Register Python normalization functions as SQLite UDFs."""
    conn.create_function("norm_name", 1, normalize_name)
    conn.create_function("norm_name_translit", 1, normalize_name_translit)
    conn.create_function("norm_name_no_legal", 1, normalize_name_no_legal)
    conn.create_function("norm_name_compact", 1, normalize_name_compact)
    conn.create_function("norm_address", 1, normalize_address)
    conn.create_function("norm_country", 1, normalize_country)
    conn.create_function("first_last_tokens", 1, first_last_tokens)


def add_normalized_columns(conn: sqlite3.Connection, table: str, progress: bool = True):
    """
    Add normalized text columns to a source table if they don't exist yet.
    Uses SQLite UDFs to avoid loading the entire table into Python.
    """
    _register_udfs(conn)

    new_columns = {
        "name_norm":          'norm_name("business_name")',
        "name_translit":      'norm_name_translit("business_name")',
        "name_no_legal":      'norm_name_no_legal("business_name")',
        "name_compact":       'norm_name_compact("business_name")',
        "addr_norm":          'norm_address("business_address")',
        "country_norm":       'norm_country("country")',
        "name_first_last":    'first_last_tokens("business_name")',
    }

    cols_to_add = {}
    for col, expr in new_columns.items():
        if not _table_has_column(conn, table, col):
            cols_to_add[col] = expr

    if not cols_to_add:
        if progress:
            print(f"  {table}: all normalized columns already exist, skipping.")
        return

    # Add columns
    for col in cols_to_add:
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" TEXT;')
    conn.commit()

    # Populate in a single UPDATE
    set_clauses = ", ".join(f'"{col}" = {expr}' for col, expr in cols_to_add.items())
    if progress:
        print(f"  {table}: populating {len(cols_to_add)} columns ({', '.join(cols_to_add)})...")

    conn.execute(f'UPDATE "{table}" SET {set_clauses};')
    conn.commit()

    if progress:
        count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        print(f"  {table}: done — {count:,} rows updated.")


def ensure_indexes(conn: sqlite3.Connection, table: str, progress: bool = True):
    """Create blocking indexes if they don't already exist."""
    indexes = [
        (f"idx_{table}_id",               f'"{table}"(entity_id)'),
        (f"idx_{table}_country_name",      f'"{table}"(country_norm, name_norm)'),
        (f"idx_{table}_name_translit",     f'"{table}"(country_norm, name_translit)'),
        (f"idx_{table}_name_no_legal",     f'"{table}"(country_norm, name_no_legal)'),
        (f"idx_{table}_name_compact",      f'"{table}"(country_norm, name_compact)'),
        (f"idx_{table}_first_last",        f'"{table}"(country_norm, name_first_last)'),
        (f"idx_{table}_country_addr",      f'"{table}"(country_norm, addr_norm)'),
    ]

    existing = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index';"
    ).fetchall()}

    for idx_name, idx_def in indexes:
        if idx_name not in existing:
            if progress:
                print(f"  Creating index {idx_name}...")
            conn.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_def};')

    conn.commit()


def ensure_ground_truth_pairs(conn: sqlite3.Connection, progress: bool = True):
    """
    Ensure the ground_truth_pairs table exists.
    If it already exists and has rows, skip recreation.
    """
    try:
        count = conn.execute("SELECT COUNT(*) FROM ground_truth_pairs").fetchone()[0]
        if count > 0:
            if progress:
                print(f"  ground_truth_pairs already exists with {count:,} rows.")
            return
    except sqlite3.OperationalError:
        pass  # table doesn't exist

    if progress:
        print("  Creating ground_truth_pairs table...")

    conn.execute("DROP TABLE IF EXISTS ground_truth_pairs;")
    conn.execute("""
        CREATE TABLE ground_truth_pairs (
            source1_entity_id TEXT,
            candidate_entity_id TEXT,
            candidate_source TEXT,
            label INTEGER
        );
    """)

    # Expand ground_truth comma-separated matches into individual pair rows
    cursor = conn.execute("SELECT source1_entity_id, matched_entity_ids FROM ground_truth;")
    batch = []
    total = 0
    for s1_id, matched_ids in cursor:
        if matched_ids and str(matched_ids).strip():
            for cand_id in str(matched_ids).split(","):
                cand_id = cand_id.strip()
                if cand_id:
                    source = "S2" if cand_id.startswith("S2") else "S3"
                    batch.append((s1_id, cand_id, source, 1))
        if len(batch) >= 100_000:
            conn.executemany(
                "INSERT INTO ground_truth_pairs VALUES (?, ?, ?, ?);",
                batch,
            )
            total += len(batch)
            batch = []
            if progress and total % 1_000_000 == 0:
                print(f"    Inserted {total:,} pairs...")

    if batch:
        conn.executemany("INSERT INTO ground_truth_pairs VALUES (?, ?, ?, ?);", batch)
        total += len(batch)

    conn.commit()

    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gt_pairs_s1 ON ground_truth_pairs(source1_entity_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gt_pairs_composite ON ground_truth_pairs(source1_entity_id, candidate_entity_id);")
    conn.commit()

    if progress:
        print(f"  ground_truth_pairs: {total:,} rows created.")


def setup_database(conn: sqlite3.Connection, progress: bool = True):
    """
    One-stop setup: add normalized columns, indexes, and ground truth pairs
    for all source tables.
    """
    if progress:
        print("Setting up database...")

    for table in ["source1", "source2", "source3"]:
        if progress:
            print(f"\n--- {table} ---")
        add_normalized_columns(conn, table, progress=progress)
        ensure_indexes(conn, table, progress=progress)

    ensure_ground_truth_pairs(conn, progress=progress)

    if progress:
        print("\nDatabase setup complete.")


def fetch_entity(conn: sqlite3.Connection, entity_id: str) -> Optional[dict]:
    """Fetch a single entity by ID from the appropriate source table."""
    prefix = entity_id[:2]
    table_map = {"S1": "source1", "S2": "source2", "S3": "source3"}
    table = table_map.get(prefix)
    if not table:
        return None
    row = conn.execute(f'SELECT * FROM "{table}" WHERE entity_id = ?;', (entity_id,)).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
    return dict(zip([c[1] for c in conn.execute(f'PRAGMA table_info("{table}")').fetchall()], row))


def get_s1_entity_ids(conn: sqlite3.Connection, split: str = "all") -> list[str]:
    """Get Source-1 entity IDs. split='all' for all, or 'train'/'val' after splitting."""
    return [row[0] for row in conn.execute("SELECT entity_id FROM source1;").fetchall()]


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    """Count rows in a table."""
    return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
