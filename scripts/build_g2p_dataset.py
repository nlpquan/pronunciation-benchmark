"""Build the Phase 2 G2P fine-tuning dataset: train/val/test CSVs from WikiPron.

`test` reproduces the exact Phase 1 benchmark item set (see
data/g2p_dataset.py) so the eventual G2P-vs-TTS PER comparison
(scripts/compare_g2p_vs_tts.py) is apples-to-apples. `train`/`val` are
everything else in the same four WikiPron categories, with any accidental
overlap with the test set excluded.

Usage:
    python scripts/build_g2p_dataset.py
    python scripts/build_g2p_dataset.py --val-fraction 0.05 --output-dir data/processed/g2p
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pronunciation_benchmark.data.g2p_dataset import build_g2p_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-category", type=int, default=125)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/g2p"))
    args = parser.parse_args()

    splits = build_g2p_dataset(n_per_category=args.n_per_category, val_fraction=args.val_fraction, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, df in splits.items():
        out_path = args.output_dir / f"{split_name}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"{split_name}: {len(df)} rows -> {out_path}")

    print("\nPer-language row counts:")
    for split_name, df in splits.items():
        counts = df["lang_id"].value_counts().sort_index()
        print(f"  {split_name}: {dict(counts)}")


if __name__ == "__main__":
    main()
