"""Text normalization and tokenization that treat Arabic and English the same way."""

from __future__ import annotations

import html
import re
import unicodedata

_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # harakat, Quranic marks and tatweel
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_TAGS = re.compile(r"<[^>]+>")
# Arabic definite article and common one-letter prefixes (و، ب، ل، ف، ك) in front of "ال"
_AR_PREFIX = re.compile(r"^(?:[وفبلك]?ال|لل)(?=\w{2,})")
_AR_PLURAL = re.compile(r"(?<=\w{3})(?:ات|ون|ين)$")

STOPWORDS = frozenset(
    "the of in and a an to for by on at is was were what how many much which did does do with from "
    "per each between during total number qatar "
    "في من على الى عن ما ماذا كم هل كان كانت هو هي التي الذي و او مع بين خلال حسب عدد اجمالي قطر دوله"
    .split()
)


def strip_html(text: str) -> str:
    return " ".join(html.unescape(_TAGS.sub(" ", text or "")).split())


def normalize(text: str) -> str:
    """Case-fold, unify digits, strip Arabic diacritics and unify common letter variants."""
    t = unicodedata.normalize("NFKC", text or "").casefold().translate(_DIGITS)
    t = _DIACRITICS.sub("", t)
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"[^\w\s.]", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)  # keep decimal points only
    return " ".join(t.split())


def light_stem(token: str) -> str:
    """A very light Arabic stemmer: drop the definite article and a few plural suffixes."""
    if token.isascii():
        return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")) else token
    token = _AR_PREFIX.sub("", token)
    return _AR_PLURAL.sub("", token)


def tokenize(text: str) -> list[str]:
    return [light_stem(t) for t in normalize(text).split() if t not in STOPWORDS]


def is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum("؀" <= c <= "ۿ" for c in letters) / len(letters) > 0.5


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def numbers_in(text: str) -> set[str]:
    """Numbers in a text, in a canonical form (Arabic digits converted, thousands separators dropped)."""
    out = set()
    for raw in _NUMBER.findall(unicodedata.normalize("NFKC", text or "").translate(_DIGITS).replace("٬", ",").replace("٫", ".")):
        n = raw.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out
