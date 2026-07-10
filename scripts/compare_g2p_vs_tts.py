"""Compare the fine-tuned G2P model's PER against Phase 1's raw TTS PER.

Reads `data/results/benchmark_results.csv` (TTS side, produced by
scripts/run_benchmark.py) and `data/results/g2p_predictions.csv` (G2P side,
produced by notebooks/g2p_finetune.ipynb after a Colab training run), joins
on (lang_id, word), and prints a per-language table: each provider's corpus
PER vs. the G2P model's corpus PER. Both sides are scored with the exact
same strip_suprasegmentals + corpus_phoneme_error_rate pipeline so the
numbers are directly comparable - see notebooks/g2p_finetune.ipynb's
docstring for why the G2P test set is guaranteed to be the same items.

Usage:
    python scripts/compare_g2p_vs_tts.py
    python scripts/compare_g2p_vs_tts.py --predictions data/results/g2p_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from pronunciation_benchmark.scoring.normalize import strip_suprasegmentals  # noqa: E402
from pronunciation_benchmark.scoring.per import corpus_phoneme_error_rate, tokenize_ipa  # noqa: E402


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


def score_tts_results(results: pd.DataFrame) -> pd.DataFrame:
    """Per-(provider, language) corpus PER from an already-run benchmark_results.csv.

    `reference`/`hypothesis` columns are already space-joined,
    already-stripped strings (see benchmark.runner.results_to_dataframe) -
    re-tokenize but don't re-strip, to match how they were originally scored.
    A legitimately-empty phoneme sequence (e.g. Allosaurus recognized zero
    phonemes) round-trips as "" thanks to main()'s keep_default_na=False
    read, and "".split() == [] is exactly the empty list we want here.
    """
    scored = results[results["per"].notna()]
    rows = []
    for (provider, lang_id), group in scored.groupby(["provider", "lang_id"]):
        pairs = [
            (ref.split(), hyp.split())
            for ref, hyp in zip(group["reference"], group["hypothesis"])
        ]
        rows.append(
            {
                "provider": provider,
                "lang_id": lang_id,
                "n_items": len(group),
                "corpus_per": corpus_phoneme_error_rate(pairs),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("data/results/benchmark_results.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("data/results/g2p_predictions.csv"))
    args = parser.parse_args()

    if not args.predictions.exists():
        print(
            f"{args.predictions} not found - run notebooks/g2p_finetune.ipynb in Colab first "
            "and move its g2p_predictions.csv download here."
        )
        return

    # keep_default_na=False: some real words (e.g. Vietnamese "nan") collide
    # with pandas' default NA sentinels and would otherwise silently become
    # missing values; it also makes a legitimately-empty phoneme sequence
    # round-trip as "" instead of NaN, which score_tts_results relies on.
    # `per` is the one genuinely-numeric column that needs real NaN for
    # unscored rows (results[results["per"].notna()] below), so it keeps
    # its own na_values instead of inheriting the blanket override.
    results = pd.read_csv(args.results, keep_default_na=False, na_values={"per": [""]})
    predictions = pd.read_csv(args.predictions, keep_default_na=False, na_values=[])

    tts_scores = score_tts_results(results)
    g2p_scores = score_g2p_predictions(predictions)

    print("TTS corpus PER by provider/language:")
    print(tts_scores.sort_values(["lang_id", "corpus_per"]).to_string(index=False))
    print()

    best_tts = tts_scores.loc[tts_scores.groupby("lang_id")["corpus_per"].idxmin()]
    best_tts = best_tts.rename(columns={"provider": "best_tts_provider", "corpus_per": "best_tts_per"})

    comparison = best_tts.merge(
        g2p_scores.rename(columns={"corpus_per": "g2p_per", "n_items": "g2p_n_items"}),
        on="lang_id",
        how="outer",
    )
    comparison["improvement"] = comparison["best_tts_per"] - comparison["g2p_per"]

    print("Best-TTS-provider PER vs. fine-tuned G2P PER (positive improvement = G2P is better):")
    print(comparison.sort_values("lang_id").to_string(index=False))


if __name__ == "__main__":
    main()
