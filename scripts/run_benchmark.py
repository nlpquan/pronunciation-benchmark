"""End-to-end benchmark run: dataset -> TTS -> phoneme extraction -> PER.

Runs sampled WikiPron-sourced benchmark items through every configured TTS
provider and reports Phoneme Error Rate per provider/category. Providers
with no API key configured, and (provider, language) pairs with no voice
configured (see benchmark/voice_config.py - currently only Vietnamese is
filled in for Azure/Google), are skipped rather than failing the run.

This makes real, billed API calls - defaults to a small sample so a run
doesn't accidentally spend the full dataset's worth of calls.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --n-per-category 20 --categories vietnamese
    python scripts/run_benchmark.py --providers openai elevenlabs --output data/results/run1.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

from pronunciation_benchmark.benchmark.runner import (  # noqa: E402
    results_to_dataframe,
    run_benchmark,
    summarize_by_provider_and_category,
)
from pronunciation_benchmark.data.names import build_benchmark_dataset  # noqa: E402

load_dotenv()

ALL_CATEGORIES = ["vietnamese", "south_asian", "middle_eastern", "african", "medical", "oov"]
# ElevenLabs is excluded from the default provider set: the account has no
# credit (trial expired, confirmed via repeated 402 Payment Required
# responses), not a code issue. The client still works - pass
# `--providers elevenlabs` explicitly to include it once billing is fixed.
DEFAULT_PROVIDERS = ["openai", "azure", "google"]
ALL_PROVIDERS = ["openai", "elevenlabs", "azure", "google"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-category", type=int, default=5)
    parser.add_argument("--categories", nargs="+", default=ALL_CATEGORIES, choices=ALL_CATEGORIES)
    parser.add_argument("--providers", nargs="+", default=DEFAULT_PROVIDERS, choices=ALL_PROVIDERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/results/benchmark_results.csv"))
    args = parser.parse_args()

    print(f"Building dataset: up to {args.n_per_category}/category across {args.categories}")
    dataset = build_benchmark_dataset(args.categories, n_per_category=args.n_per_category, seed=args.seed)
    print(f"{len(dataset)} items. Running providers: {args.providers}\n")

    results = run_benchmark(dataset, args.providers)
    df = results_to_dataframe(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {args.output}\n")

    n_errors = int(df["error"].notna().sum())
    if n_errors:
        print(f"{n_errors}/{len(df)} item/provider pairs failed - most common errors:")
        print(df.loc[df["error"].notna(), "error"].value_counts().head(10).to_string())
        print()

    summary = summarize_by_provider_and_category(df)
    if summary.empty:
        print("No successful (provider, item) pairs to summarize.")
    else:
        print("Corpus PER by provider/category (lower = better):")
        print(summary.sort_values(["category", "corpus_per"]).to_string(index=False))


if __name__ == "__main__":
    main()
