"""Shared leaderboard aggregation: corpus PER by provider/category/language,
and the G2P-vs-TTS comparison.

Both `scripts/compare_g2p_vs_tts.py` (CLI) and `app/streamlit_app.py` (UI)
import from here so the two surfaces can never silently diverge on how PER
is computed or grouped - there's exactly one implementation of "corpus PER
grouped by some columns", reused everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..scoring.normalize import strip_suprasegmentals
from ..scoring.per import corpus_phoneme_error_rate, tokenize_ipa


def load_tts_results(path: Path) -> pd.DataFrame:
    """Load benchmark_results.csv.

    keep_default_na=False: some real words (e.g. Vietnamese "nan") collide
    with pandas' default NA sentinels and would otherwise silently become
    missing values. `per` is the one genuinely-numeric column that needs
    real NaN for unscored (provider, language) gaps, so it keeps its own
    na_values instead of inheriting the blanket override.
    """
    return pd.read_csv(path, keep_default_na=False, na_values={"per": [""]})


def load_g2p_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[])


def score_tts_results(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Corpus PER grouped by arbitrary columns of `results`.

    `reference`/`hypothesis` are already space-joined, already-stripped
    strings (see benchmark.runner.results_to_dataframe) - re-tokenize but
    don't re-strip, to match how they were originally scored. Rows with no
    `per` (unavailable provider/language pairs, e.g. Azure+Yoruba) are
    excluded rather than counted as zero.
    """
    scored = results[results["per"].notna()]
    rows = []
    for key, group in scored.groupby(group_cols):
        key_tuple = key if isinstance(key, tuple) else (key,)
        pairs = [(ref.split(), hyp.split()) for ref, hyp in zip(group["reference"], group["hypothesis"])]
        row = dict(zip(group_cols, key_tuple))
        row["n_items"] = len(group)
        row["corpus_per"] = corpus_phoneme_error_rate(pairs)
        rows.append(row)
    return pd.DataFrame(rows)


def score_g2p_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-language corpus PER for the G2P model, scored the same way as the TTS side."""
    rows = []
    for lang_id, group in predictions.groupby("lang_id"):
        pairs = [
            (
                strip_suprasegmentals(tokenize_ipa(ref)),
                strip_suprasegmentals(tokenize_ipa(hyp)),
            )
            for ref, hyp in zip(group["reference_ipa"], group["predicted_ipa"])
        ]
        rows.append({"lang_id": lang_id, "n_items": len(group), "corpus_per": corpus_phoneme_error_rate(pairs)})
    return pd.DataFrame(rows)


def best_tts_vs_g2p(tts_by_provider_lang: pd.DataFrame, g2p_scores: pd.DataFrame) -> pd.DataFrame:
    """Best-TTS-provider PER vs. fine-tuned G2P PER, per language.

    `tts_by_provider_lang` must be grouped by ["provider", "lang_id"]
    (i.e. the output of `score_tts_results(results, ["provider", "lang_id"])`).
    Positive `improvement` means the G2P model is better.
    """
    best_tts = tts_by_provider_lang.loc[tts_by_provider_lang.groupby("lang_id")["corpus_per"].idxmin()]
    best_tts = best_tts.rename(columns={"provider": "best_tts_provider", "corpus_per": "best_tts_per"})
    comparison = best_tts.merge(
        g2p_scores.rename(columns={"corpus_per": "g2p_per", "n_items": "g2p_n_items"}),
        on="lang_id",
        how="outer",
    )
    comparison["improvement"] = comparison["best_tts_per"] - comparison["g2p_per"]
    return comparison
