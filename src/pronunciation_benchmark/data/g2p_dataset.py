"""Train/val/test split for Phase 2's G2P fine-tuning, built from WikiPron.

The Phase 2 result PROJECT.md cares about is a fair before/after comparison:
fine-tuned G2P model PER vs. the raw TTS PER already collected in Phase 1
(see `data/results/benchmark_results.csv`, produced by
`benchmark.runner.run_benchmark`). That comparison is only meaningful if the
G2P model never saw the benchmark's test words during training - so the test
split here is *exactly* the same benchmark item set already scored on the
TTS side (same `build_benchmark_dataset` call, same seed), and every other
WikiPron entry for those languages (excluding anything that happens to match
a test word) becomes the train/val pool.

Only the four non-Western name categories are included
(vietnamese/south_asian/middle_eastern/african) - `medical`/`oov`/`eng` are
English and out of scope for Phase 2, which PROJECT.md frames as improving
"Vietnamese and other underrepresented names/languages," not English.
"""

from __future__ import annotations

import pandas as pd

from ..benchmark.runner import lang_id_from_source_file
from .names import build_benchmark_dataset
from .wikipron import load_category

G2P_CATEGORIES = ["vietnamese", "south_asian", "middle_eastern", "african"]


def load_full_language_corpus(categories: list[str] = G2P_CATEGORIES) -> pd.DataFrame:
    """Every WikiPron entry for the given categories, tagged with `lang_id`.

    Unlike `build_benchmark_dataset`, this is the *full* per-language
    vocabulary (no capitalization filtering, no sampling) - the raw material
    G2P training draws from.
    """
    frames = [load_category(category) for category in categories]
    df = pd.concat(frames, ignore_index=True)
    df["lang_id"] = df["source_file"].apply(lang_id_from_source_file)
    return df


def benchmark_test_set(*, n_per_category: int = 125, seed: int = 42) -> pd.DataFrame:
    """Reproduce the exact benchmark item set already scored in Phase 1.

    Same categories/n_per_category/seed as the full Phase 1 run
    (`scripts/run_benchmark.py --n-per-category 125`) - `build_benchmark_dataset`
    is deterministic given the same arguments, so this reconstructs the same
    rows without needing to read the results CSV.
    """
    df = build_benchmark_dataset(categories=G2P_CATEGORIES, n_per_category=n_per_category, seed=seed)
    df = df.copy()
    df["lang_id"] = df["source_file"].apply(lang_id_from_source_file)
    return df


def build_g2p_dataset(
    *, n_per_category: int = 125, val_fraction: float = 0.1, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Build the G2P train/val/test split.

    `test` is the reproduced Phase 1 benchmark set. `train`/`val` are drawn
    from every other WikiPron entry in the same four categories, with any
    row whose (lang_id, word) matches a test item excluded first so the
    model never trains on a test word - a coincidental spelling match across
    languages is also excluded via the same (lang_id, word) key, not just
    word alone.

    WikiPron legitimately has multiple rows for the same (lang_id, word)
    with different IPA (alternate pronunciations - ~22% of the English file,
    e.g. two transcriptions of "'Murica"), so the train/val split is done on
    unique (lang_id, word) keys rather than individual rows - otherwise the
    same word could land in both splits under different transcriptions,
    letting the model "see" a val word during training and inflating the
    validation signal used for early stopping/model selection in Colab.

    Returned frames share columns [lang_id, word, ipa].
    """
    test_df = benchmark_test_set(n_per_category=n_per_category, seed=seed)
    test_keys = test_df[["lang_id", "word"]].drop_duplicates()

    corpus = load_full_language_corpus(G2P_CATEGORIES)
    merged = corpus.merge(test_keys, on=["lang_id", "word"], how="left", indicator=True)
    trainval_df = merged[merged["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)

    unique_keys = trainval_df[["lang_id", "word"]].drop_duplicates().reset_index(drop=True)
    val_keys = unique_keys.sample(frac=val_fraction, random_state=seed)
    split_marker = trainval_df.merge(
        val_keys.assign(_is_val=True), on=["lang_id", "word"], how="left"
    )
    val_df = split_marker[split_marker["_is_val"] == True].drop(columns="_is_val").reset_index(drop=True)  # noqa: E712
    train_df = split_marker[split_marker["_is_val"].isna()].drop(columns="_is_val").reset_index(drop=True)

    columns = ["lang_id", "word", "ipa"]
    return {
        "train": train_df[columns],
        "val": val_df[columns],
        "test": test_df[columns],
    }
