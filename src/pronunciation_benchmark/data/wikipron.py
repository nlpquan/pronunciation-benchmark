"""Loader for WikiPron pronunciation data.

WikiPron (https://github.com/CUNY-CL/wikipron) ships pre-scraped
<word, IPA> pairs as TSV files under data/scrape/tsv/, one file per
language/script/transcription-style combination, e.g.
"vie_latn_hanoi_narrow.tsv". Each line is `word<TAB>ipa`, where the IPA
column is a sequence of phoneme segments separated by spaces (this
segmentation matters for Phoneme Error Rate scoring, so it is preserved
rather than collapsed).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import requests

WIKIPRON_RAW_BASE = "https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv"
WIKIPRON_API_LISTING = "https://api.github.com/repos/CUNY-CL/wikipron/contents/data/scrape/tsv"

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "wikipron"

# Category -> WikiPron filenames, chosen to match PROJECT.md's target
# categories (Vietnamese, other non-Western names, ...). "_filtered"
# variants drop WikiPron's lower-confidence scrapes and are preferred
# where available.
LANGUAGE_PRESETS: dict[str, list[str]] = {
    "vietnamese": ["vie_latn_hanoi_narrow_filtered.tsv"],
    "south_asian": [
        "hin_deva_broad_filtered.tsv",  # Hindi
        "ben_beng_dhaka_broad_filtered.tsv",  # Bengali
        "tam_taml_broad.tsv",  # Tamil
        "urd_arab_broad.tsv",  # Urdu
    ],
    "middle_eastern": [
        "ara_arab_broad.tsv",  # Arabic
        "fas_arab_broad.tsv",  # Persian
    ],
    "african": [
        "yor_latn_broad.tsv",  # Yoruba
        "swa_latn_broad.tsv",  # Swahili
    ],
    # "medical" and "oov" both draw their raw candidate pool from the same
    # English file - what makes an entry "medical" or "oov" isn't anything
    # WikiPron tags, it's an external lexicon applied afterward in
    # names.py's extract_medical_terms/extract_oov_words (see data/lexicons.py).
    "medical": ["eng_latn_us_broad_filtered.tsv"],
    "oov": ["eng_latn_us_broad_filtered.tsv"],
}


def download_tsv(filename: str, *, force: bool = False) -> Path:
    """Download one WikiPron TSV file to the local cache, returning its path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / filename
    if dest.exists() and not force:
        return dest

    response = requests.get(f"{WIKIPRON_RAW_BASE}/{filename}", timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def parse_tsv(path: Path) -> pd.DataFrame:
    """Parse a WikiPron TSV into a DataFrame with columns [word, ipa].

    `ipa` retains WikiPron's original space-separated phoneme segmentation.
    """
    rows = []
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f"Malformed WikiPron row in {path.name}: {row!r}")
            word, ipa = row
            rows.append({"word": word, "ipa": ipa})
    return pd.DataFrame(rows, columns=["word", "ipa"])


def load_tsv(filename: str, *, force_download: bool = False) -> pd.DataFrame:
    """Download (if needed) and parse a single WikiPron TSV file."""
    path = download_tsv(filename, force=force_download)
    return parse_tsv(path)


def load_category(category: str, *, force_download: bool = False) -> pd.DataFrame:
    """Load and concatenate all WikiPron files mapped to a LANGUAGE_PRESETS category."""
    if category not in LANGUAGE_PRESETS:
        raise KeyError(f"Unknown category {category!r}; choose from {sorted(LANGUAGE_PRESETS)}")

    frames = []
    for filename in LANGUAGE_PRESETS[category]:
        df = load_tsv(filename, force_download=force_download)
        df["source_file"] = filename
        df["category"] = category
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def list_available_files() -> list[str]:
    """Query the WikiPron repo for all available TSV filenames (uncached network call)."""
    response = requests.get(WIKIPRON_API_LISTING, params={"per_page": 1000}, timeout=30)
    response.raise_for_status()
    return sorted(entry["name"] for entry in response.json() if entry["name"].endswith(".tsv"))
