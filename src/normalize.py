"""
Text normalization for entity resolution.

Multiple complementary text views are produced so that blocking and
feature engineering can exploit exact matches on different representations.
"""
import re
import string
import unicodedata
from typing import Optional

from unidecode import unidecode

from src.config import LEGAL_SUFFIXES

# Pre-compiled patterns
_WS_RE  = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Build a regex that matches any legal suffix as a whole word (case-insensitive)
_LEGAL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def nfkc(text: str) -> str:
    """Apply Unicode NFKC normalization."""
    return unicodedata.normalize("NFKC", text)


def casefold(text: str) -> str:
    """Full Unicode casefolding (more aggressive than .lower())."""
    return text.casefold()


def strip_accents(text: str) -> str:
    """Remove combining diacritical marks (accents) while keeping base chars."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space, strip edges."""
    return _WS_RE.sub(" ", text).strip()


def remove_punctuation(text: str) -> str:
    """Remove ASCII punctuation characters."""
    return text.translate(_PUNCT_TABLE)


def transliterate(text: str) -> str:
    """Transliterate Unicode text to closest ASCII representation (Devanagari → Latin, etc.)."""
    return unidecode(text)


def remove_legal_suffixes(text: str) -> str:
    """Remove common legal suffixes (Inc, Corp, Ltd, …) from a business name."""
    cleaned = _LEGAL_RE.sub("", text)
    return collapse_whitespace(cleaned)


# ─── Composite normalizers ──────────────────────────────────────────────


def normalize_name(raw: Optional[str]) -> str:
    """
    Standard name normalization pipeline:
      NFKC → casefold → strip accents → remove punctuation → collapse whitespace
    """
    if not raw:
        return ""
    text = nfkc(str(raw))
    text = casefold(text)
    text = strip_accents(text)
    text = remove_punctuation(text)
    text = collapse_whitespace(text)
    return text


def normalize_name_translit(raw: Optional[str]) -> str:
    """
    Like normalize_name but also transliterates non-Latin scripts to ASCII.
    """
    if not raw:
        return ""
    text = nfkc(str(raw))
    text = transliterate(text)
    text = casefold(text)
    text = strip_accents(text)
    text = remove_punctuation(text)
    text = collapse_whitespace(text)
    return text


def normalize_name_no_legal(raw: Optional[str]) -> str:
    """
    normalize_name + remove legal suffixes.
    """
    text = normalize_name(raw)
    return remove_legal_suffixes(text)


def normalize_name_compact(raw: Optional[str]) -> str:
    """
    Compact representation: normalized name with all spaces removed.
    Useful for catching variations like "Wal Mart" vs "Walmart".
    """
    return normalize_name(raw).replace(" ", "")


def normalize_address(raw: Optional[str]) -> str:
    """
    Address normalization pipeline (same as name normalization for now).
    """
    if not raw:
        return ""
    text = nfkc(str(raw))
    text = casefold(text)
    text = strip_accents(text)
    text = remove_punctuation(text)
    text = collapse_whitespace(text)
    return text


def normalize_country(raw: Optional[str]) -> str:
    """Normalize country string: casefold + strip."""
    if not raw:
        return ""
    return casefold(str(raw)).strip()


def name_tokens(raw: Optional[str]) -> list[str]:
    """Return sorted, deduplicated tokens of a normalized name."""
    norm = normalize_name(raw)
    if not norm:
        return []
    return sorted(set(norm.split()))


def first_last_tokens(raw: Optional[str]) -> str:
    """Return 'first_token|last_token' blocking key from a normalized name."""
    tokens = normalize_name(raw).split()
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return f"{tokens[0]}|{tokens[-1]}"


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Generate character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}
