"""Loader for real person names sourced from Wikidata.

Wikidata's Lexeme space has almost no coverage for personal given/family
names — as of writing, the "given name" lexical category (Q202444) has only
2 entries total across *all* languages, so querying names as Lexemes is a
dead end. Names are sourced from person Items instead: for humans (Q5) with
a given citizenship, pull the "name in native language" claim (P1559),
filtered to the expected language tag. This gives real names in native
script, which is what the benchmark needs to feed into TTS providers.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "pronunciation-benchmark/0.1 (research project)"

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "wikidata"

# category -> list of (country QID, native-name language tag, human-readable label).
# Countries chosen for non-trivial P1559 coverage, verified via COUNT queries
# against the live SPARQL endpoint at write time.
COUNTRY_PRESETS: dict[str, list[tuple[str, str, str]]] = {
    "vietnamese": [("Q881", "vi", "Vietnam")],
    "south_asian": [
        ("Q668", "hi", "India"),
        ("Q843", "ur", "Pakistan"),
        ("Q902", "bn", "Bangladesh"),
    ],
    "middle_eastern": [
        ("Q794", "fa", "Iran"),
        ("Q851", "ar", "Saudi Arabia"),
        ("Q796", "ar", "Iraq"),
    ],
    "african": [
        ("Q114", "sw", "Kenya"),
        ("Q115", "am", "Ethiopia"),
    ],
}


def _run_query(query: str) -> list[dict]:
    response = requests.get(
        WIKIDATA_SPARQL_ENDPOINT,
        params={"query": query},
        headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["results"]["bindings"]


def query_names_by_country(country_qid: str, lang_tag: str, *, limit: int = 200) -> pd.DataFrame:
    """Query humans with citizenship=country_qid, returning their native-language
    name label (P1559), filtered to lang_tag."""
    query = f"""
    SELECT ?person ?nativeName WHERE {{
      ?person wdt:P31 wd:Q5 ;
              wdt:P27 wd:{country_qid} ;
              wdt:P1559 ?nativeName .
      FILTER(LANG(?nativeName) = "{lang_tag}")
    }}
    LIMIT {limit}
    """
    bindings = _run_query(query)
    rows = [
        {
            "name": b["nativeName"]["value"],
            "wikidata_id": b["person"]["value"].rsplit("/", 1)[-1],
            "country_qid": country_qid,
            "lang_tag": lang_tag,
        }
        for b in bindings
    ]
    return pd.DataFrame(rows, columns=["name", "wikidata_id", "country_qid", "lang_tag"])


def load_category(
    category: str,
    *,
    limit_per_country: int = 200,
    cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load and concatenate names for all countries mapped to a COUNTRY_PRESETS category."""
    if category not in COUNTRY_PRESETS:
        raise KeyError(f"Unknown category {category!r}; choose from {sorted(COUNTRY_PRESETS)}")

    cache_path = CACHE_DIR / f"{category}.csv"
    if cache and cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path, encoding="utf-8")

    frames = []
    for country_qid, lang_tag, country_label in COUNTRY_PRESETS[category]:
        df = query_names_by_country(country_qid, lang_tag, limit=limit_per_country)
        df["country_label"] = country_label
        df["category"] = category
        frames.append(df)
        time.sleep(1)  # be polite to the shared public SPARQL endpoint

    combined = pd.concat(frames, ignore_index=True)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cache_path, index=False, encoding="utf-8")
    return combined
