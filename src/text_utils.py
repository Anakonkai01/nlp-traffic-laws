"""Shared Vietnamese text normalization utilities."""

import re
from functools import lru_cache

_pyvi_tk = None


def _vn_tokenizer():
    global _pyvi_tk
    if _pyvi_tk is None:
        from pyvi import ViTokenizer
        _pyvi_tk = ViTokenizer
    return _pyvi_tk


def vn_tokenize(text: str) -> str:
    """Vietnamese word segmentation -> space-separated normalized tokens."""
    try:
        return _vn_tokenizer().tokenize(text.lower())
    except Exception:
        return text.lower()


@lru_cache(maxsize=200_000)
def normalize(text: str) -> str:
    """pyvi word-segmented, whitespace-normalized Vietnamese text."""
    return re.sub(r"\s+", " ", vn_tokenize(text)).strip()
