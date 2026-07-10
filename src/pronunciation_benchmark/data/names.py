"""Build the benchmark's name/word list directly from WikiPron.

The original design tried to link real Wikidata person names to WikiPron
IPA entries by exact string match. That worked well for Vietnamese (~72%
full-name match rate - Vietnamese given/family names are literally built
from common single-syllable dictionary words, so they land in a general
dictionary by luck of orthography) but poorly for South Asian / Middle
Eastern names (13-17%) and failed almost entirely for African names
(<1%), since personal names are proper nouns that a general-vocabulary
dictionary mostly doesn't contain - the same reason an English word list
and an English baby-name list barely overlap.

Instead, benchmark items are sourced directly from WikiPron, which
guarantees IPA ground truth by construction - no linkage step, no partial
coverage. For Latin-script languages (which have letter case), a
capitalized first letter is used as a cheap, noisy proper-noun signal.

Caveat: Hindi, Bengali, Tamil, and Urdu (south_asian) and Arabic and
Persian (middle_eastern) are written in scripts with no upper/lowercase
distinction, so this signal doesn't exist for them (verified empirically:
0% of entries have an uppercase-classified first letter). Those two
categories therefore draw from WikiPron's general vocabulary rather than
filtered proper nouns - still valid benchmark items (PROJECT.md's "common
OOV words" category covers general non-Western vocabulary), just not
guaranteed to be literal person names.
"""

from __future__ import annotations

import pandas as pd

from .wikipron import load_category

# Categories whose underlying languages (per wikipron.py's LANGUAGE_PRESETS)
# are written in a script with case: vietnamese -> Latin; african ->
# Yoruba/Swahili, both Latin. south_asian and middle_eastern use scripts
# with no case distinction, so capitalization can't select proper nouns there.
CASED_SCRIPT_CATEGORIES = {"vietnamese", "african"}


def extract_names(category: str, wikipron_df: pd.DataFrame, *, min_length: int = 2) -> pd.DataFrame:
    """Return the subset of a WikiPron category's entries usable as benchmark items.

    For cased-script categories, filters to capitalized-first-letter entries
    (a noisy proper-noun proxy - it also catches place names and
    religious/mythological terms, not just given names). For uncased
    scripts, returns the full vocabulary unfiltered.

    `min_length` drops single-character entries (e.g. "X", "M") - Wiktionary
    letter/abbreviation entries that are degenerate as pronunciation items.
    """
    df = wikipron_df
    if category in CASED_SCRIPT_CATEGORIES:
        is_capitalized = df["word"].str[0].str.isupper()
        df = df[is_capitalized]
    return df[df["word"].str.len() >= min_length].reset_index(drop=True)


def build_benchmark_dataset(
    categories: list[str] | None = None,
    *,
    n_per_category: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the Phase 1 benchmark dataset: up to n_per_category items per
    category (fewer if a category doesn't have that many after filtering),
    each with WikiPron IPA ground truth already attached.
    """
    if categories is None:
        categories = ["vietnamese", "south_asian", "middle_eastern", "african"]

    frames = []
    for category in categories:
        wikipron_df = load_category(category)
        names_df = extract_names(category, wikipron_df)
        sample_size = min(n_per_category, len(names_df))
        frames.append(names_df.sample(n=sample_size, random_state=seed).reset_index(drop=True))

    return pd.concat(frames, ignore_index=True)
