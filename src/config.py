"""
Central configuration for the entity-resolution pipeline.
All tunables live here so every module reads the same values.
"""
from pathlib import Path
import os

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "student_resource" / "dataset"
TRAIN_ROOT   = DATASET_ROOT / "train"
TEST_ROOT    = DATASET_ROOT / "test"

TRAIN_S1   = TRAIN_ROOT / "train_source1.tsv"
TRAIN_S2   = TRAIN_ROOT / "train_source2.tsv"
TRAIN_S3   = TRAIN_ROOT / "train_source3.tsv"
TRAIN_GT   = TRAIN_ROOT / "train_ground_truth.tsv"

TEST_S1    = TEST_ROOT / "test_source1.tsv"
TEST_S2    = TEST_ROOT / "test_source2.tsv"
TEST_S3    = TEST_ROOT / "test_source3.tsv"

DB_DIR     = PROJECT_ROOT / "database"
DB_PATH    = DB_DIR / "amazon_ml.db"

MODEL_DIR  = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ── Reproducibility ─────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Candidate generation ────────────────────────────────────────────────
BATCH_SIZE   = 5_000          # S1 entities per candidate-generation batch
MAX_CANDIDATES_PER_S1 = 200   # hard safety cap on candidates per S1 entity
SQL_IN_CHUNK = 500            # max placeholders per IN(...) clause

# ── Training pairs ──────────────────────────────────────────────────────
NEGATIVE_TO_POSITIVE_RATIO = 2
VAL_FRACTION = 0.2            # fraction of S1 entities held out for validation

# ── LightGBM defaults ───────────────────────────────────────────────────
LGBM_PARAMS = dict(
    objective        = "binary",
    n_estimators     = 1000,
    learning_rate    = 0.05,
    num_leaves       = 63,
    max_depth        = -1,
    min_child_samples= 50,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
    verbose          = -1,
)

# ── XGBoost defaults ────────────────────────────────────────────────────
XGB_PARAMS = dict(
    objective        = "binary:logistic",
    n_estimators     = 1000,
    learning_rate    = 0.05,
    max_depth        = 8,
    min_child_weight = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    gamma            = 0.1,
    reg_alpha        = 0.1,
    reg_lambda        = 1.0,
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
    eval_metric      = "logloss",
    verbosity        = 0,
)

# ── Threshold search range ──────────────────────────────────────────────
THRESHOLD_MIN  = 0.05
THRESHOLD_MAX  = 0.95
THRESHOLD_STEP = 0.01

# ── Legal suffixes to strip from business names ─────────────────────────
LEGAL_SUFFIXES = [
    "inc", "incorporated",
    "corp", "corporation",
    "co", "company",
    "ltd", "limited",
    "llc", "llp", "plc",
    "private", "pvt",
    "sarl",
    "gmbh", "ag",
    "sa", "sas", "srl",
    "pty",
    "nv", "bv",
    "dba",
]
