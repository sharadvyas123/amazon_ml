"""
Pairwise feature engineering for entity resolution.

All features are deterministic and return a clean numeric vector.
No categorical country encoding — only relational features (same_country).
"""
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from src.normalize import (
    normalize_name, normalize_name_translit, normalize_name_no_legal,
    normalize_name_compact, normalize_address, normalize_country,
    char_ngrams, name_tokens,
)


# Feature names — must be kept in sync with compute_features()
FEATURE_NAMES = [
    # Name features
    "name_exact_match",
    "name_norm_exact_match",
    "name_translit_exact_match",
    "name_no_legal_exact_match",
    "name_compact_exact_match",
    "name_fuzz_ratio",
    "name_fuzz_wratio",
    "name_fuzz_partial_ratio",
    "name_fuzz_token_sort",
    "name_fuzz_token_set",
    "name_char_ngram_sim",
    "name_len_diff",
    "name_len_ratio",
    "name_token_jaccard",
    # Address features
    "addr_norm_exact_match",
    "addr_fuzz_ratio",
    "addr_fuzz_token_sort",
    "addr_fuzz_partial_ratio",
    "addr_char_ngram_sim",
    "addr_len_diff",
    "addr_len_ratio",
    # Other features
    "same_country",
    "s1_name_missing",
    "s2_name_missing",
    "s1_addr_missing",
    "s2_addr_missing",
    "source_indicator",  # 0 = S2, 1 = S3
]

NUM_FEATURES = len(FEATURE_NAMES)


def _safe_str(x) -> str:
    """Convert to string safely, treating None/NaN as empty."""
    if x is None:
        return ""
    s = str(x)
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def _is_missing(x) -> bool:
    """Check if a value is missing/empty."""
    return _safe_str(x) == ""


def _char_ngram_similarity(a: str, b: str, n: int = 3) -> float:
    """Jaccard similarity of character n-grams."""
    if not a or not b:
        return 0.0
    ngrams_a = char_ngrams(a, n)
    ngrams_b = char_ngrams(b, n)
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)
    return intersection / union if union > 0 else 0.0


def _token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity over word tokens."""
    if not a or not b:
        return 0.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def _len_diff(a: str, b: str) -> float:
    """Absolute length difference."""
    return abs(len(a) - len(b))


def _len_ratio(a: str, b: str) -> float:
    """Length ratio (shorter / longer), 0 if both empty."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def compute_features(s1: dict, s2: dict) -> np.ndarray:
    """
    Compute the feature vector for a single (S1, candidate) pair.

    Args:
        s1: dict with keys like business_name, business_address, country,
            name_norm, name_translit, name_no_legal, name_compact, addr_norm, country_norm
        s2: same schema as s1

    Returns:
        numpy array of shape (NUM_FEATURES,)
    """
    # Raw values
    s1_name_raw = _safe_str(s1.get("business_name"))
    s2_name_raw = _safe_str(s2.get("business_name"))

    # Normalized values (use precomputed if available, else compute)
    s1_name_norm    = _safe_str(s1.get("name_norm")) or normalize_name(s1_name_raw)
    s2_name_norm    = _safe_str(s2.get("name_norm")) or normalize_name(s2_name_raw)

    s1_name_trans   = _safe_str(s1.get("name_translit")) or normalize_name_translit(s1_name_raw)
    s2_name_trans   = _safe_str(s2.get("name_translit")) or normalize_name_translit(s2_name_raw)

    s1_name_nolegal = _safe_str(s1.get("name_no_legal")) or normalize_name_no_legal(s1_name_raw)
    s2_name_nolegal = _safe_str(s2.get("name_no_legal")) or normalize_name_no_legal(s2_name_raw)

    s1_name_compact = _safe_str(s1.get("name_compact")) or normalize_name_compact(s1_name_raw)
    s2_name_compact = _safe_str(s2.get("name_compact")) or normalize_name_compact(s2_name_raw)

    s1_addr_norm    = _safe_str(s1.get("addr_norm")) or normalize_address(_safe_str(s1.get("business_address")))
    s2_addr_norm    = _safe_str(s2.get("addr_norm")) or normalize_address(_safe_str(s2.get("business_address")))

    s1_country_norm = _safe_str(s1.get("country_norm")) or normalize_country(_safe_str(s1.get("country")))
    s2_country_norm = _safe_str(s2.get("country_norm")) or normalize_country(_safe_str(s2.get("country")))

    # ── Name features ───────────────────────────────────────────────────
    name_exact         = float(s1_name_raw.lower().strip() == s2_name_raw.lower().strip()) if s1_name_raw and s2_name_raw else 0.0
    name_norm_exact    = float(s1_name_norm == s2_name_norm) if s1_name_norm and s2_name_norm else 0.0
    name_trans_exact   = float(s1_name_trans == s2_name_trans) if s1_name_trans and s2_name_trans else 0.0
    name_nolegal_exact = float(s1_name_nolegal == s2_name_nolegal) if s1_name_nolegal and s2_name_nolegal else 0.0
    name_compact_exact = float(s1_name_compact == s2_name_compact) if s1_name_compact and s2_name_compact else 0.0

    # RapidFuzz similarities (0-100 scale → 0-1)
    name_fuzz_ratio   = fuzz.ratio(s1_name_norm, s2_name_norm) / 100.0 if s1_name_norm and s2_name_norm else 0.0
    name_fuzz_wratio  = fuzz.WRatio(s1_name_norm, s2_name_norm) / 100.0 if s1_name_norm and s2_name_norm else 0.0
    name_fuzz_partial = fuzz.partial_ratio(s1_name_norm, s2_name_norm) / 100.0 if s1_name_norm and s2_name_norm else 0.0
    name_fuzz_tsort   = fuzz.token_sort_ratio(s1_name_norm, s2_name_norm) / 100.0 if s1_name_norm and s2_name_norm else 0.0
    name_fuzz_tset    = fuzz.token_set_ratio(s1_name_norm, s2_name_norm) / 100.0 if s1_name_norm and s2_name_norm else 0.0

    name_ngram_sim    = _char_ngram_similarity(s1_name_norm, s2_name_norm)
    name_len_diff_v   = _len_diff(s1_name_norm, s2_name_norm)
    name_len_ratio_v  = _len_ratio(s1_name_norm, s2_name_norm)
    name_tok_jaccard  = _token_jaccard(s1_name_norm, s2_name_norm)

    # ── Address features ────────────────────────────────────────────────
    addr_norm_exact    = float(s1_addr_norm == s2_addr_norm) if s1_addr_norm and s2_addr_norm else 0.0
    addr_fuzz_ratio    = fuzz.ratio(s1_addr_norm, s2_addr_norm) / 100.0 if s1_addr_norm and s2_addr_norm else 0.0
    addr_fuzz_tsort    = fuzz.token_sort_ratio(s1_addr_norm, s2_addr_norm) / 100.0 if s1_addr_norm and s2_addr_norm else 0.0
    addr_fuzz_partial  = fuzz.partial_ratio(s1_addr_norm, s2_addr_norm) / 100.0 if s1_addr_norm and s2_addr_norm else 0.0
    addr_ngram_sim     = _char_ngram_similarity(s1_addr_norm, s2_addr_norm)
    addr_len_diff_v    = _len_diff(s1_addr_norm, s2_addr_norm)
    addr_len_ratio_v   = _len_ratio(s1_addr_norm, s2_addr_norm)

    # ── Other features ──────────────────────────────────────────────────
    same_country       = float(s1_country_norm == s2_country_norm) if s1_country_norm and s2_country_norm else 0.0
    s1_name_miss       = float(_is_missing(s1_name_raw))
    s2_name_miss       = float(_is_missing(s2_name_raw))
    s1_addr_miss       = float(_is_missing(s1.get("business_address")))
    s2_addr_miss       = float(_is_missing(s2.get("business_address")))

    cand_id = _safe_str(s2.get("entity_id"))
    source_ind = 1.0 if cand_id.startswith("S3") else 0.0

    return np.array([
        name_exact, name_norm_exact, name_trans_exact, name_nolegal_exact, name_compact_exact,
        name_fuzz_ratio, name_fuzz_wratio, name_fuzz_partial,
        name_fuzz_tsort, name_fuzz_tset,
        name_ngram_sim, name_len_diff_v, name_len_ratio_v, name_tok_jaccard,
        addr_norm_exact, addr_fuzz_ratio, addr_fuzz_tsort, addr_fuzz_partial,
        addr_ngram_sim, addr_len_diff_v, addr_len_ratio_v,
        same_country, s1_name_miss, s2_name_miss, s1_addr_miss, s2_addr_miss,
        source_ind,
    ], dtype=np.float32)


def compute_features_batch(
    s1_list: list[dict],
    s2_list: list[dict],
) -> np.ndarray:
    """
    Compute features for a batch of (S1, candidate) pairs.
    s1_list[i] is paired with s2_list[i].

    Returns: numpy array of shape (n_pairs, NUM_FEATURES)
    """
    n = len(s1_list)
    X = np.empty((n, NUM_FEATURES), dtype=np.float32)
    for i in range(n):
        X[i] = compute_features(s1_list[i], s2_list[i])
    return X
