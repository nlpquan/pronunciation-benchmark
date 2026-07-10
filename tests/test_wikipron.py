import pandas as pd
import pytest

from pronunciation_benchmark.data.wikipron import LANGUAGE_PRESETS, load_category, parse_tsv


def test_parse_tsv_preserves_phoneme_segmentation(tmp_path):
    path = tmp_path / "sample.tsv"
    path.write_text("hello\th ə l oʊ\nworld\tw ɜː l d\n", encoding="utf-8")

    df = parse_tsv(path)

    assert list(df.columns) == ["word", "ipa"]
    assert len(df) == 2
    assert df.loc[0, "word"] == "hello"
    assert df.loc[0, "ipa"] == "h ə l oʊ"


def test_parse_tsv_rejects_malformed_row(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("onlyoneword\n", encoding="utf-8")

    with pytest.raises(ValueError):
        parse_tsv(path)


def test_parse_tsv_empty_file_returns_empty_dataframe(tmp_path):
    path = tmp_path / "empty.tsv"
    path.write_text("", encoding="utf-8")

    df = parse_tsv(path)

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == ["word", "ipa"]


def test_load_category_rejects_unknown_category():
    with pytest.raises(KeyError):
        load_category("atlantean")


def test_language_presets_cover_project_categories():
    assert "vietnamese" in LANGUAGE_PRESETS
    for filenames in LANGUAGE_PRESETS.values():
        assert filenames, "each category must map to at least one WikiPron file"


def test_medical_and_oov_categories_share_the_english_source_file():
    assert LANGUAGE_PRESETS["medical"] == ["eng_latn_us_broad_filtered.tsv"]
    assert LANGUAGE_PRESETS["oov"] == ["eng_latn_us_broad_filtered.tsv"]
