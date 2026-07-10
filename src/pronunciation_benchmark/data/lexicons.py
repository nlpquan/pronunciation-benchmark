"""External word lists used to filter WikiPron's English dictionary into
the `medical` and `oov` benchmark categories.

Both categories draw from the same underlying WikiPron file
(`eng_latn_us_broad_filtered.tsv`, see wikipron.py's LANGUAGE_PRESETS) -
what differs is which external signal selects the subset of it to use, since
WikiPron itself carries no topical/frequency metadata:

- `medical`: intersected against Wiktionary's `en:Medicine` and
  `en:Pharmacology` category members (live MediaWiki API query). Only ~16%
  of category members have a WikiPron IPA entry (many are abbreviations or
  appendix pages with no pronunciation section), but that still yields
  ~1,600 real medical/anatomical terms - verified empirically before
  committing to this approach, given this project's history with the
  Wikidata name-linkage dead end (see project memory).
- `oov`: WikiPron entries *not* in a public top-10k common-English-word
  list, as a cheap proxy for "words a TTS system's training data is less
  likely to have seen often."
"""

from __future__ import annotations

from pathlib import Path

import requests

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "lexicons"

WIKTIONARY_API = "https://en.wiktionary.org/w/api.php"
# Wikimedia asks API clients to identify themselves with a descriptive
# User-Agent; requests without one have been observed returning an empty
# body instead of a clear error.
USER_AGENT = "pronunciation-benchmark/0.1 (research project; contact: nguyenlephuongquan@gmail.com)"

MEDICAL_WIKTIONARY_CATEGORIES = ["Category:en:Medicine", "Category:en:Pharmacology"]
COMMON_WORDS_URL = (
    "https://raw.githubusercontent.com/first20hours/google-10000-english/"
    "master/google-10000-english-no-swears.txt"
)


def _fetch_category_members(cmtitle: str) -> list[str]:
    """Page through every member of a Wiktionary category (namespace 0 only)."""
    titles: list[str] = []
    cmcontinue: str | None = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cmtitle,
            "cmlimit": 500,
            "cmnamespace": 0,
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        response = requests.get(WIKTIONARY_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
        data = response.json()
        titles.extend(member["title"] for member in data["query"]["categorymembers"])
        cont = data.get("continue")
        if not cont:
            break
        cmcontinue = cont["cmcontinue"]
    return titles


def load_medical_terms(*, cache: bool = True, force_refresh: bool = False) -> set[str]:
    """Lowercased medical/pharma terms from Wiktionary's topical categories.

    Filters out multi-word phrases, hyphenated compounds, and all-caps
    abbreviations/initialisms, keeping single alphabetic words - the same
    shape as WikiPron's `word` column. This is deliberately permissive (it
    doesn't itself check against WikiPron): names.py's extract_medical_terms
    does the actual intersection, which is what filters out anything that
    isn't a real WikiPron entry.
    """
    cache_path = CACHE_DIR / "medical_terms.txt"
    if cache and cache_path.exists() and not force_refresh:
        return set(cache_path.read_text(encoding="utf-8").splitlines())

    raw_titles: set[str] = set()
    for category in MEDICAL_WIKTIONARY_CATEGORIES:
        raw_titles.update(_fetch_category_members(category))

    terms = {
        title.lower()
        for title in raw_titles
        if title.isalpha() and not title.isupper() and len(title) >= 3
    }

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("\n".join(sorted(terms)), encoding="utf-8")
    return terms


def load_common_words(*, cache: bool = True, force_refresh: bool = False) -> set[str]:
    """Lowercased top-10k common English words, used to define "OOV"."""
    cache_path = CACHE_DIR / "common_words.txt"
    if cache and cache_path.exists() and not force_refresh:
        return set(cache_path.read_text(encoding="utf-8").splitlines())

    response = requests.get(COMMON_WORDS_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    words = {line.strip().lower() for line in response.text.splitlines() if line.strip()}

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("\n".join(sorted(words)), encoding="utf-8")
    return words
