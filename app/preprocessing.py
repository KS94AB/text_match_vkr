from __future__ import annotations

import re
from typing import Iterable, List

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for analysis."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text.strip())
    return text


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(normalize_text(text))


def character_ngrams(text: str, n: int = 5) -> List[str]:
    normalized = normalize_text(text)
    if len(normalized) < n:
        return [normalized] if normalized else []
    return [normalized[i:i+n] for i in range(len(normalized) - n + 1)]


def token_ngrams(tokens: Iterable[str], n: int = 3) -> List[str]:
    tokens = list(tokens)
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def shingles(text: str, n: int = 5) -> set[str]:
    return set(character_ngrams(text, n=n))
