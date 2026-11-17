"""Small helpers. Ports of util/*.cpp from the C++ kernel."""

from __future__ import annotations

import logging
import math
import os
import random
import re
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("util")


_STRAY_TEXT_TAG = re.compile(r"</?text>", re.IGNORECASE)


def strip_llm_markup(text: str) -> str:
    """Some LLMs leak a <text>...</text> wrapper into tool arguments; people in Telegram must never see it.
    Text without such a tag is returned untouched."""
    return _STRAY_TEXT_TAG.sub("", text).strip() if _STRAY_TEXT_TAG.search(text) else text


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a private state file without exposing a partial write to backups or hot-reload."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


# --------------------------------------------------------------------------------------------------------------------
# json_utils.h
# --------------------------------------------------------------------------------------------------------------------
def json_as_int(v) -> int | None:
    """Qwen 3.5 9B bug: outputs integers as scientific notation (e.g. 7.576e+08); coerce manually."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        result = int(v)
        log.warning("coerced number %r to int: %d", v, result)
        return result
    if isinstance(v, str):
        try:
            result = int(float(v))
            log.warning("coerced string %r to int: %d", v, result)
            return result
        except ValueError:
            pass
    log.warning("can't convert JSON value to int: %r", v)
    return None


# --------------------------------------------------------------------------------------------------------------------
# time helpers
# --------------------------------------------------------------------------------------------------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def fmt_ts(ts: int | float) -> str:
    """`2026-05-21 06:38:08` (UTC), same as std::chrono formatting of seconds."""
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def now_str() -> str:
    return utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")


def time_ago(tp: datetime, now: datetime | None = None) -> str:
    now = now or utcnow()
    diff = now - tp
    minutes = int(diff.total_seconds() // 60)
    hours = minutes // 60
    days = hours // 24
    weeks = days // 7
    if minutes < 1:
        return "just now"
    if minutes == 1:
        return "1 minute ago"
    if minutes < 60:
        return f"{minutes} minutes ago"
    if hours == 1:
        return "1 hour ago"
    if hours < 24:
        return f"{hours} hours ago"
    if days == 1:
        return "1 day ago"
    if days < 7:
        return f"{days} days ago"
    if weeks == 1:
        return "1 week ago"
    return f"{weeks} weeks ago"


def format_past_hours(past: timedelta) -> str:
    now = utcnow()
    f = "%Y-%m-%d %H:%M:%S"
    return f"{(now - past).strftime(f)}-{now.strftime(f)}"


# --------------------------------------------------------------------------------------------------------------------
# typos.cpp
# --------------------------------------------------------------------------------------------------------------------
def _build_neighbors() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}

    def build(rows: list[str]) -> None:
        for row_i, r in enumerate(rows):
            for col, key in enumerate(r):
                n: list[str] = []
                if col > 0:
                    n.append(r[col - 1])
                if col + 1 < len(r):
                    n.append(r[col + 1])
                if row_i > 0 and col < len(rows[row_i - 1]):
                    n.append(rows[row_i - 1][col])
                if row_i + 1 < len(rows) and col < len(rows[row_i + 1]):
                    n.append(rows[row_i + 1][col])
                result[key] = n

    build(["qwertyuiop", "asdfghjkl", "zxcvbnm"])  # English QWERTY
    build(["йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю"])  # Russian ЙЦУКЕН
    return result


_NEIGHBORS = _build_neighbors()


def swap_adjacent_chars(original: str, rng: random.Random | None = None) -> str:
    rng = rng or random
    chars = list(original)
    if len(chars) < 2:
        return original
    i = rng.randint(0, len(chars) - 2)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def replace_with_keyboard_neighbor(original: str, rng: random.Random | None = None) -> str:
    rng = rng or random
    chars = list(original)
    if not chars:
        return original
    i = rng.randint(0, len(chars) - 1)
    repl = _NEIGHBORS.get(chars[i].lower())
    if not repl:
        return original
    chars[i] = rng.choice(repl)
    return "".join(chars)


# --------------------------------------------------------------------------------------------------------------------
# react_with_emoji.cpp
# --------------------------------------------------------------------------------------------------------------------
def strip_variation_selector(s: str) -> str:
    """Telegram stores reaction emojis without the U+FE0F variation selector."""
    return s.replace("️", "")


# --------------------------------------------------------------------------------------------------------------------
# OpenAITools.cpp
# --------------------------------------------------------------------------------------------------------------------
def remove_control_characters(s: str) -> str:
    """toml11 inserts beautiful formatting into exceptions, and JSON is not happy about it."""
    return "".join(c for c in s if c in "\n\t" or ord(c) >= 0x20)


# --------------------------------------------------------------------------------------------------------------------
# Lexical similarity (replaces embeddings where a cheap check is enough: anti-repeat, plagiarism, name matching).
# --------------------------------------------------------------------------------------------------------------------
_WORD_RE = re.compile(r"[\w]+", re.UNICODE)


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold().replace("ё", "е")
    return s


def tokens(s: str) -> list[str]:
    return _WORD_RE.findall(normalize_text(s))


def stem(token: str) -> str:
    """Crude language-agnostic stemmer: good enough to match «Алексей/Алексея/Алексею» or «song/songs».

    Words up to 4 letters are kept as they are; longer words are cut to their first 5 letters (4 for 5-letter words).
    """
    if len(token) <= 4:
        return token
    if len(token) == 5:
        return token[:4]
    return token[:5]


def stems(s: str) -> list[str]:
    return [stem(t) for t in tokens(s)]


def stem_key(token: str) -> str:
    """Coarse bucket key for candidate lookup: the first 4 letters. Over-broad on purpose (the exact decision is made
    by `similar_word`); consistent across word lengths, unlike `stem` («алекс» -> «алек», «алексу» -> «алекс»)."""
    return token if len(token) <= 4 else token[:4]


_CASE_ENDINGS = {"ом", "ем", "ой", "ей", "ою", "ею", "ах", "ях", "ам", "ям"}  # Russian two-letter case endings


def similar_word(a: str, b: str) -> bool:
    """Same word in a different grammatical form? Conservative on purpose: it decides whether two names are the same
    person. «Алексей/Алексея/Алексеем» yes; «Иван/Иванов», «Alexander/Alexandra», «Node1/Node2» no."""
    if a == b:
        return True
    if any(c.isdigit() for c in a + b):
        return False
    m = min(len(a), len(b))
    if m < 4:
        return False
    if a.isascii() and b.isascii():  # latin names barely inflect: allow plurals / possessives only
        return a + "s" == b or b + "s" == a or a + "es" == b or b + "es" == a
    cp = 0
    while cp < m and a[cp] == b[cp]:
        cp += 1
    if cp < 3 or cp < m - 1:  # at most the last letter of the shorter word may differ
        return False
    ta, tb = a[cp:], b[cp:]
    return (len(ta) <= 1 or ta in _CASE_ENDINGS) and (len(tb) <= 1 or tb in _CASE_ENDINGS)


_NULLISH = {"", "none", "null", "nil", "undefined", "n/a"}


def opt_str(v) -> str | None:
    """Nullable string argument. Some LLMs pass the *string* "None"/"null" instead of JSON null."""
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in _NULLISH else s


def char_ngrams(s: str, n: int = 3) -> Counter:
    s = " " + " ".join(tokens(s)) + " "
    return Counter(s[i : i + n] for i in range(max(0, len(s) - n + 1)))


def cosine_counters(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def lexical_similarity(a: str, b: str) -> float:
    """Cosine similarity of character trigrams, ∈ [0, 1]. Used when no embedding endpoint is configured."""
    return cosine_counters(char_ngrams(a), char_ngrams(b))


def shingles(s: str, n: int = 3) -> set[str]:
    t = tokens(s)
    if len(t) <= n:
        return {" ".join(t)} if t else set()
    return {" ".join(t[i : i + n]) for i in range(len(t) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
