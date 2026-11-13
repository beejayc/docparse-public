"""Token agreement between VLM text and the PDF text layer (DESIGN section 4). Pure."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from rapidfuzz import fuzz, process

_HYPHEN_BREAK = re.compile(r"(\w)[-­]\s*\n\s*(\w)")
_TAGS = re.compile(r"<[^>]+>")
_MD = re.compile(r"[*_`#>|~]+|!\[[^\]]*\]\([^)]*\)")
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
FUZZY_MIN = 92  # rapidfuzz ratio to pair leftover hyphen/line-break fragments
FUZZY_MAX_TOKENS = 200  # bound the quadratic tie-break


_LETTERS = re.compile(r"[^\W\d_]+", re.UNICODE)
MATH_SPAN = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$|\\\(.+?\\\)|\\\[.+?\\\]", re.DOTALL)


def strip_math(text: str) -> str:
    """Drop ``$..$``, ``$$..$$``, ``\\(..\\)`` and ``\\[..\\]`` spans (ADR-025)."""
    return MATH_SPAN.sub(" ", text)


def tokens(text: str, min_len: int = 2, letters_only: bool = False) -> list[str]:
    """NFKC, case-fold, strip markdown/HTML, keep word characters (umlauts and ß survive).

    ``letters_only`` keeps alphabetic runs only (no digits, no glyph clusters): maths and symbol
    fonts then stop producing tokens the other side cannot match (ADR-025).
    """
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    if letters_only:
        text = _TAGS.sub(" ", text)
        return [t for t in _LETTERS.findall(text.casefold()) if len(t) >= min_len]
    text = _TAGS.sub(" ", text)
    text = _MD.sub(" ", text)
    text = _NON_WORD.sub(" ", text.casefold().replace("_", " "))
    return [t for t in text.split() if len(t) >= min_len]


def token_agreement(
    vlm_text: str, layer_text: str, min_len: int = 2, letters_only: bool = False
) -> tuple[float, float]:
    """(recall, precision) of multisets of tokens.

    recall = share of text-layer tokens found in the VLM text; precision = share of VLM tokens found
    in the text layer. ``rapidfuzz`` only pairs leftover fragments (hyphenated line breaks).
    """
    v, layer = (
        Counter(tokens(vlm_text, min_len, letters_only)),
        Counter(tokens(layer_text, min_len, letters_only)),
    )
    n_v, n_l = sum(v.values()), sum(layer.values())
    if n_l == 0 or n_v == 0:
        return (1.0 if n_l == 0 else 0.0, 1.0 if n_v == 0 else 0.0)
    common = sum((v & layer).values())
    left_v, left_l = list((v - layer).elements()), list((layer - v).elements())
    if left_v and left_l and len(left_v) <= FUZZY_MAX_TOKENS and len(left_l) <= FUZZY_MAX_TOKENS:
        pool = list(left_v)
        for tok in left_l:
            hit = process.extractOne(tok, pool, scorer=fuzz.ratio, score_cutoff=FUZZY_MIN)
            if hit is not None:
                pool.pop(hit[2])
                common += 1
    return (min(1.0, common / n_l), min(1.0, common / n_v))
