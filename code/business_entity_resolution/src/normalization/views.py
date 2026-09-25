"""Multiple normalized text views — never overwrite raw fields."""

from __future__ import annotations

import re
import unicodedata

_AND = re.compile(r"\s*&\s*")
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)
_DIGIT = re.compile(r"\d+")

_LEGAL = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "llc": "limited",
    "pvt": "private",
    "co": "company",
    "plc": "public",
}

_STREET = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "hwy": "highway",
    "ste": "suite",
    "apt": "apartment",
}


def nfkc_casefold(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold()


def view_basic(text: str) -> str:
    t = nfkc_casefold(text)
    t = _AND.sub(" and ", t)
    t = _WS.sub(" ", t).strip()
    return t


def view_alnum(text: str) -> str:
    t = view_basic(text)
    t = _NON_ALNUM.sub(" ", t)
    return _WS.sub(" ", t).strip()


def _map_tokens(text: str, mapping: dict[str, str]) -> str:
    return " ".join(mapping.get(tok, tok) for tok in text.split())


def view_name_legal(text: str) -> str:
    return _map_tokens(view_alnum(text), _LEGAL)


def view_address(text: str) -> str:
    return _map_tokens(view_alnum(text), {**_LEGAL, **_STREET})


def tokens(text: str) -> list[str]:
    return [t for t in text.split() if t]


def significant_tokens(text: str, min_len: int = 2) -> list[str]:
    stop = {"the", "and", "of", "a", "an", "for", "to", "in", "on", "at", "by"}
    return [t for t in tokens(text) if len(t) >= min_len and t not in stop]


def numeric_tokens(text: str) -> list[str]:
    return _DIGIT.findall(text or "")


def country_key(country: str) -> str:
    return nfkc_casefold((country or "").strip())


def attach_views(df):
    """Return a copy with normalized view columns; raw columns preserved."""
    out = df.copy()
    out["name_basic"] = out["business_name"].map(view_basic)
    out["name_alnum"] = out["business_name"].map(view_alnum)
    out["name_legal"] = out["business_name"].map(view_name_legal)
    out["addr_basic"] = out["business_address"].map(view_basic)
    out["addr_norm"] = out["business_address"].map(view_address)
    out["country_norm"] = out["country"].map(country_key)
    out["name_tokens"] = out["name_legal"].map(significant_tokens)
    out["addr_nums"] = out["addr_norm"].map(numeric_tokens)
    out["name_prefix3"] = out["name_alnum"].map(lambda x: "".join(x.split())[:3])
    return out
