import pandas as pd

import pronunciation_benchmark.data.g2p_dataset as g2p_module
from pronunciation_benchmark.data.g2p_dataset import (
    benchmark_test_set,
    build_g2p_dataset,
    load_full_language_corpus,
)


def _wikipron_like_df(words, source_file="vie_latn_hanoi_narrow_filtered.tsv"):
    return pd.DataFrame(
        {
            "word": words,
            "ipa": [f"ipa_{w}" for w in words],
            "source_file": source_file,
            "category": "fake",
        }
    )


def test_load_full_language_corpus_tags_lang_id(monkeypatch):
    def fake_load_category(category):
        if category == "vietnamese":
            return _wikipron_like_df(["Nguyen", "Tran"], source_file="vie_latn_hanoi_narrow_filtered.tsv")
        return _wikipron_like_df(["shabd"], source_file="hin_deva_broad_filtered.tsv")

    monkeypatch.setattr(g2p_module, "load_category", fake_load_category)

    df = load_full_language_corpus(["vietnamese", "south_asian"])

    assert set(df["lang_id"]) == {"vie", "hin"}
    assert len(df) == 3


def test_benchmark_test_set_tags_lang_id(monkeypatch):
    def fake_build_benchmark_dataset(categories, n_per_category, seed):
        return _wikipron_like_df(["Nguyen"], source_file="vie_latn_hanoi_narrow_filtered.tsv")

    monkeypatch.setattr(g2p_module, "build_benchmark_dataset", fake_build_benchmark_dataset)

    df = benchmark_test_set(n_per_category=1, seed=1)

    assert list(df["lang_id"]) == ["vie"]
    assert list(df["word"]) == ["Nguyen"]


def test_build_g2p_dataset_excludes_test_words_from_train_and_val(monkeypatch):
    monkeypatch.setattr(
        g2p_module,
        "build_benchmark_dataset",
        lambda categories, n_per_category, seed: _wikipron_like_df(["Nguyen", "Tran"]),
    )
    monkeypatch.setattr(
        g2p_module,
        "load_category",
        lambda category: _wikipron_like_df([f"word{i}" for i in range(20)] + ["Nguyen", "Tran"]),
    )

    splits = build_g2p_dataset(n_per_category=2, val_fraction=0.2, seed=7)

    test_words = set(splits["test"]["word"])
    assert test_words == {"Nguyen", "Tran"}
    assert not test_words & set(splits["train"]["word"])
    assert not test_words & set(splits["val"]["word"])


def test_build_g2p_dataset_train_val_test_cover_corpus_without_overlap(monkeypatch):
    monkeypatch.setattr(
        g2p_module,
        "build_benchmark_dataset",
        lambda categories, n_per_category, seed: _wikipron_like_df(["Nguyen"]),
    )
    all_words = [f"word{i}" for i in range(30)] + ["Nguyen"]

    def fake_load_category(category):
        # Only "vietnamese" carries data; the other three G2P_CATEGORIES
        # return empty frames, so the corpus is exactly `all_words` with no
        # cross-category duplicate words (mirroring how real WikiPron
        # per-language files don't share vocabulary).
        if category == "vietnamese":
            return _wikipron_like_df(all_words)
        return _wikipron_like_df([])

    monkeypatch.setattr(g2p_module, "load_category", fake_load_category)

    splits = build_g2p_dataset(n_per_category=1, val_fraction=0.2, seed=3)

    train_words = set(splits["train"]["word"])
    val_words = set(splits["val"]["word"])
    test_words = set(splits["test"]["word"])

    assert not (train_words & val_words)
    assert not (train_words & test_words)
    assert not (val_words & test_words)
    # every corpus row lands in exactly one split
    assert train_words | val_words | test_words == set(all_words)
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(all_words)


def test_build_g2p_dataset_keeps_duplicate_word_groups_in_one_split(monkeypatch):
    # WikiPron legitimately has multiple rows for the same word (alternate
    # pronunciations) - "dup" appears three times here, "unique0..N" once each.
    monkeypatch.setattr(
        g2p_module,
        "build_benchmark_dataset",
        lambda categories, n_per_category, seed: _wikipron_like_df([]),
    )
    words = ["dup", "dup", "dup"] + [f"unique{i}" for i in range(20)]

    def fake_load_category(category):
        return _wikipron_like_df(words) if category == "vietnamese" else _wikipron_like_df([])

    monkeypatch.setattr(g2p_module, "load_category", fake_load_category)

    splits = build_g2p_dataset(n_per_category=1, val_fraction=0.3, seed=5)

    train_words = set(splits["train"]["word"])
    val_words = set(splits["val"]["word"])
    # "dup" must be entirely in train or entirely in val, never split across both.
    assert not ("dup" in train_words and "dup" in val_words)


def test_build_g2p_dataset_is_deterministic_given_same_seed(monkeypatch):
    monkeypatch.setattr(
        g2p_module,
        "build_benchmark_dataset",
        lambda categories, n_per_category, seed: _wikipron_like_df(["Nguyen"]),
    )
    all_words = [f"word{i}" for i in range(20)] + ["Nguyen"]
    monkeypatch.setattr(g2p_module, "load_category", lambda category: _wikipron_like_df(all_words))

    splits1 = build_g2p_dataset(n_per_category=1, val_fraction=0.25, seed=11)
    splits2 = build_g2p_dataset(n_per_category=1, val_fraction=0.25, seed=11)

    assert list(splits1["train"]["word"]) == list(splits2["train"]["word"])
    assert list(splits1["val"]["word"]) == list(splits2["val"]["word"])
