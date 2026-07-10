import pandas as pd
import pytest

import pronunciation_benchmark.data.names as names_module
from pronunciation_benchmark.data.names import (
    CASED_SCRIPT_CATEGORIES,
    build_benchmark_dataset,
    extract_names,
)


def _wikipron_like_df(words):
    return pd.DataFrame(
        {
            "word": words,
            "ipa": [f"ipa_{w}" for w in words],
            "source_file": "fake.tsv",
            "category": "fake",
        }
    )


def test_extract_names_filters_capitalized_for_cased_category():
    df = _wikipron_like_df(["Nguyễn", "và", "Sơn", "là"])

    result = extract_names("vietnamese", df)

    assert list(result["word"]) == ["Nguyễn", "Sơn"]


def test_extract_names_returns_everything_for_uncased_category():
    df = _wikipron_like_df(["शब्द", "नाम", "भाषा"])

    result = extract_names("south_asian", df)

    assert len(result) == len(df)
    assert list(result["word"]) == list(df["word"])


def test_extract_names_uncased_category_keeps_ascii_uppercase_words_too():
    # Even if a stray ASCII-uppercase token shows up in an uncased-script
    # category's data, nothing should be filtered out since there's no
    # applicable proper-noun signal for these languages.
    df = _wikipron_like_df(["ABC", "xyz"])

    result = extract_names("middle_eastern", df)

    assert len(result) == 2


def test_cased_script_categories_matches_known_latin_script_categories():
    assert CASED_SCRIPT_CATEGORIES == {"vietnamese", "african"}


def test_extract_names_drops_single_character_entries():
    df = _wikipron_like_df(["Sơn", "X", "M", "Nguyễn"])

    result = extract_names("vietnamese", df)

    assert list(result["word"]) == ["Sơn", "Nguyễn"]


def test_extract_names_min_length_applies_to_uncased_categories_too():
    df = _wikipron_like_df(["A", "शब्द"])

    result = extract_names("south_asian", df)

    assert list(result["word"]) == ["शब्द"]


def test_build_benchmark_dataset_samples_up_to_n_per_category(monkeypatch):
    def fake_load_category(category):
        if category == "vietnamese":
            return _wikipron_like_df([f"Name{i}" for i in range(10)])
        return _wikipron_like_df([f"word{i}" for i in range(10)])

    monkeypatch.setattr(names_module, "load_category", fake_load_category)

    df = build_benchmark_dataset(categories=["vietnamese", "south_asian"], n_per_category=3, seed=1)

    assert len(df) == 6  # 3 from each category
    assert set(df["category"]) == {"fake"}  # category column comes through from source data


def test_build_benchmark_dataset_caps_at_available_count_when_fewer_than_requested(monkeypatch):
    def fake_load_category(category):
        return _wikipron_like_df(["OnlyOne"])

    monkeypatch.setattr(names_module, "load_category", fake_load_category)

    df = build_benchmark_dataset(categories=["vietnamese"], n_per_category=100, seed=1)

    assert len(df) == 1


def test_build_benchmark_dataset_is_deterministic_given_same_seed(monkeypatch):
    def fake_load_category(category):
        return _wikipron_like_df([f"Name{i}" for i in range(20)])

    monkeypatch.setattr(names_module, "load_category", fake_load_category)

    df1 = build_benchmark_dataset(categories=["vietnamese"], n_per_category=5, seed=7)
    df2 = build_benchmark_dataset(categories=["vietnamese"], n_per_category=5, seed=7)

    assert list(df1["word"]) == list(df2["word"])
