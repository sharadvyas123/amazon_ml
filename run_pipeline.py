"""
run_pipeline.py — One-file end-to-end entity resolution pipeline.

Runs on any PC where the project directory exists.  Just do:

    python run_pipeline.py

The script will:

    Phase 0  Check / build the training database   (~5 min first run)
    Phase 1  Build V2 token index + new indexes     (~10-30 min first run)
    Phase 2  Train / val split  +  training data    (~20-40 min)
    Phase 3  Train LightGBM                         (~2-5 min)
    Phase 4  Evaluate  (threshold + Macro F0.5)     (~1 min)
    Phase 5  Build test database                    (~20-40 min first run)
    Phase 6  Inference  →  output/submission.csv    (~30-90 min)

Subsequent runs skip Phase 0 / 1 / 5 if already done (idempotent).

Requirements:
    pip install pandas lightgbm rapidfuzz unidecode scikit-learn numpy
"""

from __future__ import annotations

import builtins
import csv
import gc
import os
import pickle
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Force flush on EVERY print (Windows buffers stdout) ─────────────────
_real_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _real_print(*args, **kwargs)

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (edit if needed)
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT  = Path(__file__).resolve().parent
DATASET_ROOT  = PROJECT_ROOT / "student_resource" / "dataset"

TRAIN_S1      = DATASET_ROOT / "train" / "train_source1.tsv"
TRAIN_S2      = DATASET_ROOT / "train" / "train_source2.tsv"
TRAIN_S3      = DATASET_ROOT / "train" / "train_source3.tsv"
TRAIN_GT      = DATASET_ROOT / "train" / "train_ground_truth.tsv"

TEST_S1       = DATASET_ROOT / "test" / "test_source1.tsv"
TEST_S2       = DATASET_ROOT / "test" / "test_source2.tsv"
TEST_S3       = DATASET_ROOT / "test" / "test_source3.tsv"

DB_DIR        = PROJECT_ROOT / "database"
TRAIN_DB      = DB_DIR / "amazon_ml.db"          # reuse existing
TEST_DB       = DB_DIR / "amazon_ml_test.db"

MODEL_DIR     = PROJECT_ROOT / "models"
OUTPUT_DIR    = PROJECT_ROOT / "output"

RANDOM_STATE  = 42
VAL_FRACTION  = 0.20       # held-out S1 fraction
NEG_RATIO     = 2          # negatives per positive
BATCH_SIZE    = 5_000      # S1 entities per candidate-generation batch
SQL_IN_CHUNK  = 500        # max placeholders per SQL IN(...)

MAX_CANDIDATES_PER_S1 = 500
MIN_TOKEN_OVERLAP     = 2
MAX_TOKEN_FREQ        = 10_000

# LightGBM
LGBM_PARAMS = dict(
    objective         = "binary",
    n_estimators      = 1_500,
    learning_rate     = 0.05,
    num_leaves        = 63,
    max_depth         = -1,
    min_child_samples = 50,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    verbose           = -1,
)

THRESHOLD_MIN  = 0.05
THRESHOLD_MAX  = 0.95
THRESHOLD_STEP = 0.01

# Legal suffixes stripped from business names
LEGAL_SUFFIXES = [
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "plc", "private", "pvt",
    "sarl", "gmbh", "ag", "sa", "sas", "srl", "pty", "nv", "bv", "dba",
]


# ═══════════════════════════════════════════════════════════════════════════
#  NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════

import re, string, unicodedata
from unidecode import unidecode

_WS_RE  = re.compile(r"\s+")
_PUNCT  = str.maketrans("", "", string.punctuation)
_LEGAL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

def _nfkc(t):              return unicodedata.normalize("NFKC", t)
def _casefold(t):           return t.casefold()
def _strip_accents(t):
    nfd = unicodedata.normalize("NFD", t)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
def _collapse_ws(t):        return _WS_RE.sub(" ", t).strip()
def _rm_punct(t):           return t.translate(_PUNCT)
def _transliterate(t):      return unidecode(t)
def _rm_legal(t):           return _collapse_ws(_LEGAL_RE.sub("", t))

def normalize_name(raw):
    if not raw: return ""
    t = _nfkc(str(raw)); t = _casefold(t); t = _strip_accents(t)
    t = _rm_punct(t); t = _collapse_ws(t)
    return t

def normalize_name_translit(raw):
    if not raw: return ""
    t = _nfkc(str(raw)); t = _transliterate(t); t = _casefold(t)
    t = _strip_accents(t); t = _rm_punct(t); t = _collapse_ws(t)
    return t

def normalize_name_no_legal(raw):   return _rm_legal(normalize_name(raw))
def normalize_name_compact(raw):    return normalize_name(raw).replace(" ", "")
def normalize_address(raw):         return normalize_name(raw)    # same pipeline
def normalize_country(raw):
    if not raw: return ""
    return _casefold(str(raw)).strip()

def first_last_tokens(raw):
    tokens = normalize_name(raw).split()
    if not tokens: return ""
    if len(tokens) == 1: return tokens[0]
    return f"{tokens[0]}|{tokens[-1]}"

def char_ngrams(text, n=3):
    if len(text) < n: return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURES  (pairwise)
# ═══════════════════════════════════════════════════════════════════════════

from rapidfuzz import fuzz

FEATURE_NAMES = [
    "name_exact_match", "name_norm_exact_match", "name_translit_exact_match",
    "name_no_legal_exact_match", "name_compact_exact_match",
    "name_fuzz_ratio", "name_fuzz_wratio", "name_fuzz_partial_ratio",
    "name_fuzz_token_sort", "name_fuzz_token_set",
    "name_char_ngram_sim", "name_len_diff", "name_len_ratio", "name_token_jaccard",
    "addr_norm_exact_match", "addr_fuzz_ratio", "addr_fuzz_token_sort",
    "addr_fuzz_partial_ratio", "addr_char_ngram_sim", "addr_len_diff", "addr_len_ratio",
    "same_country", "s1_name_missing", "s2_name_missing",
    "s1_addr_missing", "s2_addr_missing", "source_indicator",
]
NUM_FEATURES = len(FEATURE_NAMES)

def _safe(x):
    if x is None: return ""
    s = str(x)
    return "" if s.lower() in ("nan","none","") else s

def _ngram_sim(a, b, n=3):
    if not a or not b: return 0.0
    na, nb = char_ngrams(a, n), char_ngrams(b, n)
    if not na or not nb: return 0.0
    return len(na & nb) / len(na | nb)

def _tok_jacc(a, b):
    if not a or not b: return 0.0
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0

def compute_features(s1, s2):
    s1r = _safe(s1.get("business_name")); s2r = _safe(s2.get("business_name"))
    s1n = _safe(s1.get("name_norm")) or normalize_name(s1r)
    s2n = _safe(s2.get("name_norm")) or normalize_name(s2r)
    s1t = _safe(s1.get("name_translit")) or normalize_name_translit(s1r)
    s2t = _safe(s2.get("name_translit")) or normalize_name_translit(s2r)
    s1nl = _safe(s1.get("name_no_legal")) or normalize_name_no_legal(s1r)
    s2nl = _safe(s2.get("name_no_legal")) or normalize_name_no_legal(s2r)
    s1c = _safe(s1.get("name_compact")) or normalize_name_compact(s1r)
    s2c = _safe(s2.get("name_compact")) or normalize_name_compact(s2r)
    s1a = _safe(s1.get("addr_norm")) or normalize_address(_safe(s1.get("business_address")))
    s2a = _safe(s2.get("addr_norm")) or normalize_address(_safe(s2.get("business_address")))
    s1co = _safe(s1.get("country_norm")) or normalize_country(_safe(s1.get("country")))
    s2co = _safe(s2.get("country_norm")) or normalize_country(_safe(s2.get("country")))

    def _ex(a,b): return float(a == b) if a and b else 0.0
    def _fr(a,b): return fuzz.ratio(a,b)/100 if a and b else 0.0
    def _wr(a,b): return fuzz.WRatio(a,b)/100 if a and b else 0.0
    def _pr(a,b): return fuzz.partial_ratio(a,b)/100 if a and b else 0.0
    def _ts(a,b): return fuzz.token_sort_ratio(a,b)/100 if a and b else 0.0
    def _tset(a,b): return fuzz.token_set_ratio(a,b)/100 if a and b else 0.0
    def _ld(a,b): return abs(len(a)-len(b))
    def _lr(a,b):
        la,lb=len(a),len(b)
        if la==0 and lb==0: return 1.0
        if la==0 or lb==0: return 0.0
        return min(la,lb)/max(la,lb)

    return np.array([
        _ex(s1r.lower().strip(), s2r.lower().strip()), _ex(s1n,s2n), _ex(s1t,s2t),
        _ex(s1nl,s2nl), _ex(s1c,s2c),
        _fr(s1n,s2n), _wr(s1n,s2n), _pr(s1n,s2n), _ts(s1n,s2n), _tset(s1n,s2n),
        _ngram_sim(s1n,s2n), _ld(s1n,s2n), _lr(s1n,s2n), _tok_jacc(s1n,s2n),
        _ex(s1a,s2a), _fr(s1a,s2a), _ts(s1a,s2a), _pr(s1a,s2a),
        _ngram_sim(s1a,s2a), _ld(s1a,s2a), _lr(s1a,s2a),
        _ex(s1co,s2co),
        float(_safe(s1r)==""), float(_safe(s2r)==""),
        float(_safe(s1.get("business_address"))==""),
        float(_safe(s2.get("business_address"))==""),
        1.0 if _safe(s2.get("entity_id")).startswith("S3") else 0.0,
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  DATABASE  helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_connection(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=120)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.execute("PRAGMA cache_size = -128000;")   # 128 MB
    return conn


def _has_col(conn, table, col):
    return col in {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()}


def load_tsv_to_db(conn, tsv_path, table_name, progress=True):
    """Load a TSV file into a SQLite table (skip if table already has rows)."""
    try:
        cnt = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        if cnt > 0:
            if progress: print(f"  {table_name}: already loaded ({cnt:,} rows)")
            return
    except sqlite3.OperationalError:
        pass

    if progress: print(f"  Loading {tsv_path.name} -> {table_name} ...")
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.commit()
    if progress: print(f"  {table_name}: {len(df):,} rows loaded")


def add_normalized_columns(conn, table, progress=True):
    """Add all normalized text columns using Python UDFs."""
    conn.create_function("norm_name", 1, normalize_name)
    conn.create_function("norm_name_translit", 1, normalize_name_translit)
    conn.create_function("norm_name_no_legal", 1, normalize_name_no_legal)
    conn.create_function("norm_name_compact", 1, normalize_name_compact)
    conn.create_function("norm_address", 1, normalize_address)
    conn.create_function("norm_country", 1, normalize_country)
    conn.create_function("first_last_tokens", 1, first_last_tokens)

    col_expr = {
        "name_norm":       'norm_name("business_name")',
        "name_translit":   'norm_name_translit("business_name")',
        "name_no_legal":   'norm_name_no_legal("business_name")',
        "name_compact":    'norm_name_compact("business_name")',
        "addr_norm":       'norm_address("business_address")',
        "country_norm":    'norm_country("country")',
        "name_first_last": 'first_last_tokens("business_name")',
    }

    to_add = {c: e for c, e in col_expr.items() if not _has_col(conn, table, c)}
    if not to_add:
        if progress: print(f"  {table}: normalised columns already exist")
        return

    for c in to_add:
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}" TEXT;')
    conn.commit()

    sets = ", ".join(f'"{c}" = {e}' for c, e in to_add.items())
    if progress: print(f"  {table}: populating {len(to_add)} columns ...")
    conn.execute(f'UPDATE "{table}" SET {sets};')
    conn.commit()
    cnt = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    if progress: print(f"  {table}: {cnt:,} rows updated")


def ensure_indexes(conn, table, progress=True):
    """Create blocking indexes (idempotent)."""
    indexes = [
        (f"idx_{table}_id",              f'"{table}"(entity_id)'),
        (f"idx_{table}_country_name",    f'"{table}"(country_norm, name_norm)'),
        (f"idx_{table}_name_translit",   f'"{table}"(country_norm, name_translit)'),
        (f"idx_{table}_name_no_legal",   f'"{table}"(country_norm, name_no_legal)'),
        (f"idx_{table}_name_compact",    f'"{table}"(country_norm, name_compact)'),
        (f"idx_{table}_first_last",      f'"{table}"(country_norm, name_first_last)'),
        (f"idx_{table}_country_addr",    f'"{table}"(country_norm, addr_norm)'),
        # V2: cross-country indexes
        (f"idx_{table}_compact_only",    f'"{table}"(name_compact)'),
        (f"idx_{table}_no_legal_only",   f'"{table}"(name_no_legal)'),
    ]
    existing = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index';"
    ).fetchall()}
    for name, defn in indexes:
        if name not in existing:
            if progress: print(f"  Creating index {name} ...")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {defn};")
    conn.commit()


def ensure_ground_truth_pairs(conn, progress=True):
    """Expand ground_truth -> ground_truth_pairs (idempotent)."""
    try:
        cnt = conn.execute("SELECT COUNT(*) FROM ground_truth_pairs").fetchone()[0]
        if cnt > 0:
            if progress: print(f"  ground_truth_pairs: exists ({cnt:,} rows)")
            return
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("SELECT 1 FROM ground_truth LIMIT 1")
    except sqlite3.OperationalError:
        if progress: print("  ground_truth table not found -- skipping.")
        return

    if progress: print("  Creating ground_truth_pairs ...")
    conn.execute("DROP TABLE IF EXISTS ground_truth_pairs;")
    conn.execute("""
        CREATE TABLE ground_truth_pairs (
            source1_entity_id TEXT,
            candidate_entity_id TEXT,
            candidate_source TEXT,
            label INTEGER
        );
    """)

    cursor = conn.execute("SELECT source1_entity_id, matched_entity_ids FROM ground_truth;")
    batch, total = [], 0
    for s1_id, matched in cursor:
        if matched and str(matched).strip():
            for cid in str(matched).split(","):
                cid = cid.strip()
                if cid:
                    src = "S2" if cid.startswith("S2") else "S3"
                    batch.append((s1_id, cid, src, 1))
        if len(batch) >= 100_000:
            conn.executemany("INSERT INTO ground_truth_pairs VALUES (?,?,?,?)", batch)
            total += len(batch); batch = []
    if batch:
        conn.executemany("INSERT INTO ground_truth_pairs VALUES (?,?,?,?)", batch)
        total += len(batch)
    conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gt_pairs_s1 ON ground_truth_pairs(source1_entity_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gt_pairs_composite ON ground_truth_pairs(source1_entity_id, candidate_entity_id);")
    conn.commit()
    if progress: print(f"  ground_truth_pairs: {total:,} rows")


def build_token_index(conn, progress=True):
    """Build inverted token->entity index for S2/S3 (idempotent)."""
    try:
        cnt = conn.execute("SELECT COUNT(*) FROM name_token_index").fetchone()[0]
        if cnt > 0:
            if progress: print(f"  name_token_index: exists ({cnt:,} rows)")
            return
    except sqlite3.OperationalError:
        pass

    if progress: print("  Building name_token_index (this takes 10-30 min) ...")

    # Pass 1: token frequencies
    if progress: print("    Pass 1: counting token frequencies ...")
    tf = {}
    n_ent = 0
    for table in ["source2", "source3"]:
        for _, name_norm in conn.execute(f"SELECT entity_id, name_norm FROM {table}"):
            if not name_norm: continue
            n_ent += 1
            for t in set(str(name_norm).split()):
                if len(t) >= 2:
                    tf[t] = tf.get(t, 0) + 1
    valid = {t for t, f in tf.items() if f <= MAX_TOKEN_FREQ}
    if progress:
        print(f"    Entities: {n_ent:,}  Unique tokens: {len(tf):,}")
        print(f"    Kept (freq <= {MAX_TOKEN_FREQ:,}): {len(valid):,}  Excluded: {len(tf)-len(valid):,}")

    # Create table
    conn.execute("DROP TABLE IF EXISTS name_token_index;")
    conn.execute("CREATE TABLE name_token_index (token TEXT NOT NULL, entity_id TEXT NOT NULL);")

    # Pass 2: populate
    if progress: print("    Pass 2: populating ...")
    batch, total = [], 0
    for table in ["source2", "source3"]:
        for eid, name_norm in conn.execute(f"SELECT entity_id, name_norm FROM {table}"):
            if not name_norm: continue
            for t in set(str(name_norm).split()) & valid:
                if len(t) >= 2:
                    batch.append((t, eid))
            if len(batch) >= 100_000:
                conn.executemany("INSERT INTO name_token_index VALUES (?,?)", batch)
                total += len(batch); batch = []
                if progress and total % 5_000_000 == 0:
                    print(f"      {total:,} rows ...")
    if batch:
        conn.executemany("INSERT INTO name_token_index VALUES (?,?)", batch)
        total += len(batch)
    conn.commit()

    if progress: print("    Creating B-tree index on token ...")
    conn.execute("CREATE INDEX idx_nti_token ON name_token_index(token);")
    conn.commit()
    if progress: print(f"  name_token_index: {total:,} rows built")


def setup_database(conn, with_ground_truth=True, progress=True):
    """Full database setup: normalise -> index -> ground truth -> token index."""
    if progress: print("\n  DATABASE SETUP")
    for table in ["source1", "source2", "source3"]:
        if progress: print(f"\n--- {table} ---")
        add_normalized_columns(conn, table, progress)
        ensure_indexes(conn, table, progress)
    if with_ground_truth:
        ensure_ground_truth_pairs(conn, progress)
    build_token_index(conn, progress)
    if progress: print("\n  Database setup complete.\n")


# ═══════════════════════════════════════════════════════════════════════════
#  CANDIDATE GENERATION  (V2: token-overlap + cross-country)
# ═══════════════════════════════════════════════════════════════════════════

def _chunked(items, sz):
    for i in range(0, len(items), sz):
        yield items[i:i+sz]


def _fetch_s1_data(conn, s1_ids):
    """Fetch S1 blocking columns for given IDs."""
    cols = "entity_id, business_name, business_address, country, country_norm, " \
           "name_norm, name_translit, name_no_legal, name_compact, addr_norm, name_first_last"
    col_names = [c.strip() for c in cols.split(",")]
    rows = []
    for chunk in _chunked(s1_ids, SQL_IN_CHUNK):
        ph = ",".join("?" for _ in chunk)
        rows.extend(conn.execute(
            f"SELECT {cols} FROM source1 WHERE entity_id IN ({ph})", chunk
        ).fetchall())
    return [dict(zip(col_names, r)) for r in rows]


def _fetch_all_s1_data(conn):
    """Fetch ALL S1 blocking columns (for inference)."""
    cols = "entity_id, business_name, business_address, country, country_norm, " \
           "name_norm, name_translit, name_no_legal, name_compact, addr_norm, name_first_last"
    col_names = [c.strip() for c in cols.split(",")]
    rows = conn.execute(f"SELECT {cols} FROM source1").fetchall()
    return [dict(zip(col_names, r)) for r in rows]


def _fetch_candidate_details(conn, cand_ids):
    """Fetch S2/S3 entity details in bulk."""
    if not cand_ids: return {}
    cols = "entity_id, business_name, business_address, country, country_norm, " \
           "name_norm, name_translit, name_no_legal, name_compact, addr_norm, name_first_last"
    col_names = [c.strip() for c in cols.split(",")]
    result = {}
    for table, prefix in [("source2","S2"), ("source3","S3")]:
        ids = [x for x in cand_ids if str(x).startswith(prefix)]
        for chunk in _chunked(ids, SQL_IN_CHUNK):
            ph = ",".join("?" for _ in chunk)
            for row in conn.execute(f"SELECT {cols} FROM {table} WHERE entity_id IN ({ph})", chunk):
                e = dict(zip(col_names, row))
                result[e["entity_id"]] = e
    return result


def generate_candidates_batch(conn, s1_data, label=""):
    """
    Generate candidate sets for a batch of S1 entities.

    Strategy 1: Country-scoped exact blocking (original)
    Strategy 2: Cross-country compact blocking (V2)
    Strategy 3: Token-overlap blocking (V2, SQL-based)

    Returns {s1_id: [cand_id, ...]}.
    """
    if not s1_data: return {}
    result = {r["entity_id"]: set() for r in s1_data}
    t0 = time.time()

    # -- Strategy 1: country-scoped exact blocking --------------------
    blocking_cols = ["name_norm","name_translit","name_no_legal","name_compact","name_first_last"]
    keys = {c: set() for c in blocking_cols + ["addr_norm"]}
    for row in s1_data:
        for c in blocking_cols + ["addr_norm"]:
            v = row.get(c)
            if v:
                v = str(v).strip()
                if v: keys[c].add((row.get("country_norm") or "", v))

    for table in ["source2", "source3"]:
        for col in blocking_cols:
            pairs = list(keys[col])
            if not pairs: continue
            for chunk in _chunked(pairs, SQL_IN_CHUNK):
                conds, params = [], []
                for country, val in chunk:
                    conds.append(f"(country_norm = ? AND {col} = ?)")
                    params.extend([country, val])
                where = " OR ".join(conds)
                rows = conn.execute(
                    f"SELECT entity_id, country_norm, {col} FROM {table} WHERE {where}", params
                ).fetchall()
                lookup = {}
                for eid, country, val in rows:
                    lookup.setdefault((country or "", val or ""), []).append(eid)
                for s1 in s1_data:
                    sid = s1["entity_id"]
                    if len(result[sid]) >= MAX_CANDIDATES_PER_S1: continue
                    country = s1.get("country_norm") or ""
                    val = s1.get(col) or ""
                    if not val: continue
                    for cid in lookup.get((country, val), []):
                        if cid != sid:
                            result[sid].add(cid)
                            if len(result[sid]) >= MAX_CANDIDATES_PER_S1: break

        # address blocking
        pairs = list(keys["addr_norm"])
        if pairs:
            for chunk in _chunked(pairs, SQL_IN_CHUNK):
                conds, params = [], []
                for country, val in chunk:
                    conds.append("(country_norm = ? AND addr_norm = ?)")
                    params.extend([country, val])
                where = " OR ".join(conds)
                rows = conn.execute(
                    f"SELECT entity_id, country_norm, addr_norm FROM {table} WHERE {where}", params
                ).fetchall()
                lookup = {}
                for eid, country, val in rows:
                    lookup.setdefault((country or "", val or ""), []).append(eid)
                for s1 in s1_data:
                    sid = s1["entity_id"]
                    if len(result[sid]) >= MAX_CANDIDATES_PER_S1: continue
                    country = s1.get("country_norm") or ""
                    val = s1.get("addr_norm") or ""
                    if not val: continue
                    for cid in lookup.get((country, val), []):
                        if cid != sid:
                            result[sid].add(cid)
                            if len(result[sid]) >= MAX_CANDIDATES_PER_S1: break

    # -- Strategy 2: cross-country compact blocking -------------------
    for col in ["name_compact", "name_no_legal"]:
        values = set()
        for s1 in s1_data:
            v = (s1.get(col) or "").strip()
            if v and len(v) >= 5: values.add(v)
        if not values: continue
        for table in ["source2", "source3"]:
            for chunk in _chunked(sorted(values), SQL_IN_CHUNK):
                ph = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT entity_id, {col} FROM {table} WHERE {col} IN ({ph})", chunk
                ).fetchall()
                lookup = {}
                for eid, val in rows:
                    lookup.setdefault(val or "", []).append(eid)
                for s1 in s1_data:
                    sid = s1["entity_id"]
                    if len(result[sid]) >= MAX_CANDIDATES_PER_S1: continue
                    v = (s1.get(col) or "").strip()
                    if not v or len(v) < 5: continue
                    for cid in lookup.get(v, []):
                        if cid != sid:
                            result[sid].add(cid)
                            if len(result[sid]) >= MAX_CANDIDATES_PER_S1: break

    # -- Strategy 3: token-overlap blocking (SQL-based) ----------------
    #
    # OLD approach loaded millions of (token, entity_id) pairs into Python
    # dicts and did nested loops to count overlaps.  ~30M dict ops per
    # sub-batch → hangs on Windows for minutes/hours.
    #
    # NEW approach: build a tiny temp table with this batch's S1 tokens,
    # then let SQLite's C engine do the JOIN + GROUP BY + HAVING in one
    # query.  Runs in seconds instead of minutes.
    #
    try:
        conn.execute("SELECT 1 FROM name_token_index LIMIT 1")
    except Exception:
        pass  # index not built; skip
    else:
        SUB = 200  # smaller sub-batches for memory safety
        for sb_start in range(0, len(s1_data), SUB):
            sb = s1_data[sb_start:sb_start+SUB]

            # Build temp table with (s1_id, token) pairs for this sub-batch
            conn.execute("DROP TABLE IF EXISTS _s1_batch_tokens;")
            conn.execute(
                "CREATE TEMP TABLE _s1_batch_tokens "
                "(s1_id TEXT NOT NULL, token TEXT NOT NULL);"
            )

            insert_rows = []
            for s1 in sb:
                sid = s1["entity_id"]
                if len(result.get(sid, set())) >= MAX_CANDIDATES_PER_S1:
                    continue
                nn = s1.get("name_norm") or ""
                if not nn:
                    continue
                toks = set(t for t in nn.split() if len(t) >= 2)
                if len(toks) < MIN_TOKEN_OVERLAP:
                    continue
                for t in toks:
                    insert_rows.append((sid, t))

            if not insert_rows:
                conn.execute("DROP TABLE IF EXISTS _s1_batch_tokens;")
                continue

            conn.executemany(
                "INSERT INTO _s1_batch_tokens VALUES (?,?)", insert_rows
            )
            # Index the temp table so SQLite picks the right join plan
            conn.execute(
                "CREATE INDEX _idx_sbt_token ON _s1_batch_tokens(token);"
            )

            # One SQL query does ALL the counting in C:
            #   JOIN on token  →  GROUP BY (s1_id, candidate)  →  HAVING >= 2
            rows = conn.execute("""
                SELECT st.s1_id, nti.entity_id, COUNT(*) AS overlap
                FROM _s1_batch_tokens st
                JOIN name_token_index nti ON st.token = nti.token
                GROUP BY st.s1_id, nti.entity_id
                HAVING overlap >= ?
                ORDER BY st.s1_id, overlap DESC
            """, (MIN_TOKEN_OVERLAP,)).fetchall()

            conn.execute("DROP TABLE IF EXISTS _s1_batch_tokens;")

            # Merge results (already sorted by overlap DESC per s1_id)
            for s1_id, cand_id, _overlap in rows:
                bucket = result.setdefault(s1_id, set())
                if len(bucket) < MAX_CANDIDATES_PER_S1:
                    bucket.add(cand_id)

    elapsed = time.time() - t0
    total_cands = sum(len(c) for c in result.values())
    if label:
        print(f"    [{label}] {len(s1_data):,} S1 -> {total_cands:,} cands  ({elapsed:.1f}s)")
    return {sid: list(cands) for sid, cands in result.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING DATA  generation
# ═══════════════════════════════════════════════════════════════════════════

def get_train_val_split(conn):
    """Split S1 entity IDs into 80/20 train/val (entity-level, no leakage)."""
    all_s1 = [r[0] for r in conn.execute("SELECT entity_id FROM source1").fetchall()]
    rng = random.Random(RANDOM_STATE)
    rng.shuffle(all_s1)
    n_val = int(len(all_s1) * VAL_FRACTION)
    return all_s1[n_val:], all_s1[:n_val]


def _get_gt_positives(conn, s1_ids):
    positives = []
    for chunk in _chunked(s1_ids, SQL_IN_CHUNK):
        ph = ",".join("?" * len(chunk))
        positives.extend(conn.execute(
            f"SELECT source1_entity_id, candidate_entity_id "
            f"FROM ground_truth_pairs WHERE source1_entity_id IN ({ph}) AND label = 1",
            chunk
        ).fetchall())
    return positives


def generate_training_data(conn, s1_ids, progress=True, max_s1=None):
    """Generate (X, y, pair_ids) with V2 candidate generation."""
    rng = random.Random(RANDOM_STATE)
    if max_s1: s1_ids = s1_ids[:max_s1]
    all_X, all_y, all_pairs = [], [], []
    n_pos = n_neg = n_cands = 0
    n = len(s1_ids)

    for bstart in range(0, n, BATCH_SIZE):
        batch_t0 = time.time()
        bids = s1_ids[bstart:bstart+BATCH_SIZE]
        s1_data = _fetch_s1_data(conn, bids)
        s1_map = {d["entity_id"]: d for d in s1_data}

        # Ground truth
        gt = {}
        for sid, cid in _get_gt_positives(conn, bids):
            gt.setdefault(sid, set()).add(cid)

        # Batch candidate generation (V2)
        batch_num = bstart // BATCH_SIZE + 1
        total_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
        all_cands = generate_candidates_batch(
            conn, s1_data, label=f"batch {batch_num}/{total_batches}"
        )

        for sid in bids:
            s1_ent = s1_map.get(sid)
            if s1_ent is None: continue
            cands = all_cands.get(sid, [])
            n_cands += len(cands)
            if not cands: continue

            gt_set = gt.get(sid, set())
            pos_cands = [c for c in cands if c in gt_set]
            neg_cands = [c for c in cands if c not in gt_set]

            n_neg_want = max(int(len(pos_cands) * NEG_RATIO), 1) if pos_cands else 0
            if not pos_cands and neg_cands:
                n_neg_want = min(1, len(neg_cands))
            if len(neg_cands) > n_neg_want:
                neg_cands = rng.sample(neg_cands, n_neg_want)

            selected = pos_cands + neg_cands
            if not selected: continue

            details = _fetch_candidate_details(conn, selected)
            for cid in selected:
                c_ent = details.get(cid)
                if c_ent is None: continue
                all_X.append(compute_features(s1_ent, c_ent))
                label = 1 if cid in gt_set else 0
                all_y.append(label)
                all_pairs.append((sid, cid))
                if label == 1: n_pos += 1
                else: n_neg += 1

        if progress:
            done = min(bstart + BATCH_SIZE, n)
            batch_elapsed = time.time() - batch_t0
            print(f"  {done:,}/{n:,} S1 | pos={n_pos:,}  neg={n_neg:,}  "
                  f"cands={n_cands:,}  ({batch_elapsed:.1f}s)")

    X = np.vstack(all_X) if all_X else np.empty((0, NUM_FEATURES), dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    if progress:
        print(f"\n  Training data: {len(all_pairs):,} pairs  "
              f"({n_pos:,} pos / {n_neg:,} neg)  features={NUM_FEATURES}")
    return X, y, all_pairs


# ═══════════════════════════════════════════════════════════════════════════
#  BLOCKING RECALL  measurement
# ═══════════════════════════════════════════════════════════════════════════

def measure_blocking_recall(conn, candidates_dict, progress=True):
    s1_ids = list(candidates_dict.keys())
    if not s1_ids: return {"recall": 0.0}
    total_pos = found = 0
    for chunk in _chunked(s1_ids, SQL_IN_CHUNK):
        ph = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT source1_entity_id, candidate_entity_id "
            f"FROM ground_truth_pairs WHERE source1_entity_id IN ({ph})", chunk
        ).fetchall()
        gt = {}
        for sid, cid in rows:
            gt.setdefault(sid, set()).add(cid)
        for sid in chunk:
            true = gt.get(sid, set())
            gen = set(candidates_dict.get(sid, []))
            total_pos += len(true)
            found += len(true & gen)
    recall = found / total_pos if total_pos else 0.0
    if progress:
        print(f"\n  === Blocking Recall ===")
        print(f"  Ground-truth positives : {total_pos:,}")
        print(f"  Found in candidates    : {found:,}")
        print(f"  Blocking recall        : {recall:.4%}")
    return {"recall": recall, "total": total_pos, "found": found}


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def f05(p, r):
    if p + r == 0: return 0.0
    return 1.25 * p * r / (0.25 * p + r)


def find_best_threshold(y_true, y_proba):
    from sklearn.metrics import precision_score, recall_score
    best = {"t": 0.5, "f05": 0.0, "p": 0.0, "r": 0.0}
    for t in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP):
        yp = (y_proba >= t).astype(int)
        if yp.sum() == 0: continue
        p = precision_score(y_true, yp, zero_division=0)
        r = recall_score(y_true, yp, zero_division=0)
        f = f05(p, r)
        if f > best["f05"]:
            best = {"t": round(t, 3), "f05": f, "p": p, "r": r}
    return best


def per_entity_macro_f05(y_true, y_proba, pair_ids, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    preds, truth = {}, {}
    for i, (sid, cid) in enumerate(pair_ids):
        preds.setdefault(sid, set()); truth.setdefault(sid, set())
        if y_pred[i] == 1: preds[sid].add(cid)
        if y_true[i] == 1: truth[sid].add(cid)
    scores = []
    for sid in preds:
        ps, ts = preds[sid], truth[sid]
        if not ts:
            scores.append(1.0 if not ps else 0.0)
        elif not ps:
            scores.append(0.0)
        else:
            tp = len(ps & ts); fp = len(ps - ts); fn = len(ts - ps)
            p = tp/(tp+fp) if tp+fp else 0.0
            r = tp/(tp+fn) if tp+fn else 0.0
            scores.append(f05(p, r))
    return np.mean(scores) if scores else 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(conn, model, threshold, output_path, progress=True):
    """Run inference on ALL source1 entities -> submission CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_s1 = _fetch_all_s1_data(conn)
    total = len(all_s1)
    if progress:
        print(f"\n  INFERENCE")
        print(f"  S1 entities: {total:,}  threshold: {threshold:.3f}")

    predictions = {}
    for bstart in range(0, total, BATCH_SIZE):
        batch = all_s1[bstart:bstart+BATCH_SIZE]
        all_cands = generate_candidates_batch(conn, batch)

        # Collect all unique candidate IDs for bulk fetch
        all_cand_ids = set()
        for cids in all_cands.values():
            all_cand_ids.update(cids)
        details = _fetch_candidate_details(conn, list(all_cand_ids))

        for s1_ent in batch:
            sid = s1_ent["entity_id"]
            cands = all_cands.get(sid, [])
            if not cands:
                predictions[sid] = []
                continue

            # Compute features + predict in sub-batches
            matched = []
            for sub_start in range(0, len(cands), 500):
                sub = cands[sub_start:sub_start+500]
                X = np.empty((len(sub), NUM_FEATURES), dtype=np.float32)
                valid_j, valid_cids = [], []
                for j, cid in enumerate(sub):
                    ce = details.get(cid)
                    if ce is None: continue
                    X[j] = compute_features(s1_ent, ce)
                    valid_j.append(j); valid_cids.append(cid)
                if not valid_j: continue
                probas = model.predict_proba(X[valid_j])[:, 1]
                for k, cid in enumerate(valid_cids):
                    if probas[k] >= threshold:
                        matched.append(cid)
            predictions[sid] = matched

        if progress:
            done = min(bstart + BATCH_SIZE, total)
            n_matched = sum(1 for v in predictions.values() if v)
            print(f"  {done:,}/{total:,} | matched: {n_matched:,}")

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        for sid in sorted(predictions):
            m = predictions[sid]
            w.writerow([sid, ", ".join(sorted(m)) if m else ""])

    n_with = sum(1 for v in predictions.values() if v)
    tot_m = sum(len(v) for v in predictions.values())
    if progress:
        print(f"\n  Inference complete.")
        print(f"  S1 with >= 1 match: {n_with:,}")
        print(f"  Total matches:      {tot_m:,}")
        print(f"  Saved to: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN  PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    sys.stdout.reconfigure(encoding="utf-8")

    # -- Phase 0: Ensure training database ----------------------------
    print("\n" + "="*60)
    print("  PHASE 0: Training database")
    print("="*60)
    conn = get_connection(TRAIN_DB)

    # Load TSVs if tables are empty (first run on a fresh PC)
    load_tsv_to_db(conn, TRAIN_S1, "source1")
    load_tsv_to_db(conn, TRAIN_S2, "source2")
    load_tsv_to_db(conn, TRAIN_S3, "source3")

    # Load ground truth
    try:
        cnt = conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
        if cnt == 0: raise sqlite3.OperationalError("empty")
        print(f"  ground_truth: exists ({cnt:,} rows)")
    except sqlite3.OperationalError:
        print(f"  Loading ground truth ...")
        df = pd.read_csv(TRAIN_GT, sep="\t", dtype=str, keep_default_na=False)
        df.to_sql("ground_truth", conn, if_exists="replace", index=False)
        conn.commit()
        print(f"  ground_truth: {len(df):,} rows loaded")

    # -- Phase 1: Database setup (normalise + indexes + token index) --
    print("\n" + "="*60)
    print("  PHASE 1: Database setup (V2)")
    print("="*60)
    setup_database(conn, with_ground_truth=True)

    # -- Phase 2: Train/Val split + training data --------------------
    print("\n" + "="*60)
    print("  PHASE 2: Training data generation")
    print("="*60)
    train_ids, val_ids = get_train_val_split(conn)
    print(f"  Train S1: {len(train_ids):,}  Val S1: {len(val_ids):,}")

    # Measure blocking recall on a SMALL validation sample first
    print("\n  Measuring blocking recall on val set (sample=500) ...")
    val_sample = val_ids[:500]
    val_data = _fetch_s1_data(conn, val_sample)
    val_cands = generate_candidates_batch(conn, val_data, label="blocking-recall-sample")
    measure_blocking_recall(conn, val_cands)
    del val_data, val_cands; gc.collect()

    print("\n  Generating TRAIN data ...")
    X_train, y_train, train_pairs = generate_training_data(conn, train_ids)

    print("\n  Generating VAL data ...")
    X_val, y_val, val_pairs = generate_training_data(conn, val_ids)

    # -- Phase 3: Train LightGBM -------------------------------------
    print("\n" + "="*60)
    print("  PHASE 3: LightGBM training")
    print("="*60)
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    model = LGBMClassifier(**LGBM_PARAMS)
    print(f"  Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Features: {NUM_FEATURES}")

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(50, verbose=True), log_evaluation(100)],
    )
    print(f"  Best iteration: {getattr(model, 'best_iteration_', '?')}")

    # Feature importances
    print("\n  Feature importances (top 15):")
    imp = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    for name, score in imp[:15]:
        print(f"    {name:35s} {score:>6}")

    # -- Phase 4: Evaluation -----------------------------------------
    print("\n" + "="*60)
    print("  PHASE 4: Evaluation")
    print("="*60)
    y_proba_val = model.predict_proba(X_val)[:, 1]

    # Pair-level threshold tuning
    best = find_best_threshold(y_val, y_proba_val)
    threshold = best["t"]
    print(f"\n  Best pair-level threshold: {threshold:.3f}")
    print(f"  Pair F0.5:      {best['f05']:.4f}")
    print(f"  Pair Precision: {best['p']:.4f}")
    print(f"  Pair Recall:    {best['r']:.4f}")

    # Per-entity Macro F0.5
    macro = per_entity_macro_f05(y_val, y_proba_val, val_pairs, threshold)
    print(f"\n  ** Per-entity Macro F0.5: {macro:.4f} **")

    # Save model + threshold
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    import json
    meta = {"threshold": threshold, "macro_f05": float(macro),
            "features": FEATURE_NAMES,
            "best_iteration": getattr(model, "best_iteration_", None)}
    with open(MODEL_DIR / "lgbm_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Model saved to {model_path}")
    print(f"  Threshold: {threshold}")

    conn.close()
    del X_train, y_train, X_val, y_val; gc.collect()

    # -- Phase 5: Build test database --------------------------------
    print("\n" + "="*60)
    print("  PHASE 5: Test database")
    print("="*60)
    test_conn = get_connection(TEST_DB)

    load_tsv_to_db(test_conn, TEST_S1, "source1")
    load_tsv_to_db(test_conn, TEST_S2, "source2")
    load_tsv_to_db(test_conn, TEST_S3, "source3")
    setup_database(test_conn, with_ground_truth=False)

    # -- Phase 6: Inference -> submission.csv -------------------------
    print("\n" + "="*60)
    print("  PHASE 6: Inference")
    print("="*60)
    submission_path = OUTPUT_DIR / "submission.csv"
    run_inference(test_conn, model, threshold, submission_path)
    test_conn.close()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE  ({elapsed/60:.1f} min)")
    print(f"  Submission: {submission_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
