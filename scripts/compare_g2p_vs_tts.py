"""Compare the fine-tuned G2P model's PER against Phase 1's raw TTS PER.

Reads `data/results/benchmark_results.csv` (TTS side, produced by
scripts/run_benchmark.py) and `data/results/g2p_predictions.csv` (G2P side,
produced by notebooks/g2p_finetune.ipynb after a Colab training run), joins
on (lang_id, word), and prints a per-language table: each provider's corpus
PER vs. the G2P model's corpus PER. Both sides are scored with the exact
same strip_suprasegmentals + corpus_phoneme_error_rate pipeline (see
pronunciation_benchmark.leaderboard.data, also used by app/streamlit_app.py)
so the numbers are directly comparable - see notebooks/g2p_finetune.ipynb's
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

from pronunciation_benchmark.leaderboard.data import (  # noqa: E402
    best_tts_vs_g2p,
    load_g2p_predictions,
    load_tts_results,
    score_g2p_predictions,
    score_tts_results,
)


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

    results = load_tts_results(args.results)
    predictions = load_g2p_predictions(args.predictions)

    tts_scores = score_tts_results(results, ["provider", "lang_id"])
    g2p_scores = score_g2p_predictions(predictions)

    print("TTS corpus PER by provider/language:")
    print(tts_scores.sort_values(["lang_id", "corpus_per"]).to_string(index=False))
    print()

    comparison = best_tts_vs_g2p(tts_scores, g2p_scores)

    print("Best-TTS-provider PER vs. fine-tuned G2P PER (positive improvement = G2P is better):")
    print(comparison.sort_values("lang_id").to_string(index=False))


if __name__ == "__main__":
    main()
