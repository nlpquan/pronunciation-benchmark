import pandas as pd
import pytest

import pronunciation_benchmark.benchmark.runner as runner_module
from pronunciation_benchmark.benchmark.runner import (
    lang_id_from_source_file,
    results_to_dataframe,
    run_benchmark,
    summarize_by_provider_and_category,
)
from pronunciation_benchmark.tts.base import TTSResult


def _dataset(rows):
    return pd.DataFrame(rows, columns=["word", "ipa", "category", "source_file"])


class _FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def synthesize(self, text, voice, *, language_code=None):
        self.calls.append((text, voice, language_code))
        if self.error:
            raise self.error
        return self.response


def test_lang_id_from_source_file_takes_iso_prefix():
    assert lang_id_from_source_file("vie_latn_hanoi_narrow_filtered.tsv") == "vie"
    assert lang_id_from_source_file("hin_deva_broad_filtered.tsv") == "hin"


def test_run_benchmark_scores_a_successful_item(monkeypatch):
    fake_client = _FakeClient(response=TTSResult(audio_bytes=b"x", audio_format="mp3", provider="openai", voice="alloy"))
    monkeypatch.setattr(runner_module, "get_client", lambda provider: fake_client)
    monkeypatch.setattr(runner_module, "extract_phonemes", lambda result, lang_id: ["a", "b", "c"])

    dataset = _dataset([("Nguyễn", "a b c", "vietnamese", "vie_latn_hanoi_narrow_filtered.tsv")])
    results = run_benchmark(dataset, ["openai"])

    assert len(results) == 1
    result = results[0]
    assert result.error is None
    assert result.hypothesis == ["a", "b", "c"]
    assert result.per == 0.0
    assert fake_client.calls == [("Nguyễn", "alloy", None)]


def test_run_benchmark_records_unavailable_provider_without_crashing(monkeypatch):
    def raise_missing_key(provider):
        raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")

    monkeypatch.setattr(runner_module, "get_client", raise_missing_key)

    dataset = _dataset([("Nguyễn", "a b c", "vietnamese", "vie_latn_hanoi_narrow_filtered.tsv")])
    results = run_benchmark(dataset, ["openai"])

    assert len(results) == 1
    assert results[0].hypothesis is None
    assert results[0].per is None
    assert "OPENAI_API_KEY" in results[0].error


def test_run_benchmark_records_unconfigured_voice(monkeypatch):
    fake_client = _FakeClient(response=TTSResult(audio_bytes=b"x", audio_format="wav", provider="azure", voice="v"))
    monkeypatch.setattr(runner_module, "get_client", lambda provider: fake_client)

    # "azure" has no voice configured for a made-up language.
    dataset = _dataset([("word", "a b c", "made_up", "xyz_script_style.tsv")])
    results = run_benchmark(dataset, ["azure"])

    assert len(results) == 1
    assert results[0].error is not None
    assert "no azure voice configured" in results[0].error
    assert fake_client.calls == []


def test_run_benchmark_one_bad_item_does_not_abort_the_run(monkeypatch):
    fake_client = _FakeClient(error=RuntimeError("synthesis failed"))
    monkeypatch.setattr(runner_module, "get_client", lambda provider: fake_client)
    monkeypatch.setattr(runner_module, "extract_phonemes", lambda result, lang_id: ["a"])

    dataset = _dataset(
        [
            ("word1", "a b", "vietnamese", "vie_latn_hanoi_narrow_filtered.tsv"),
            ("word2", "a b", "vietnamese", "vie_latn_hanoi_narrow_filtered.tsv"),
        ]
    )
    results = run_benchmark(dataset, ["openai"])

    assert len(results) == 2
    assert all(r.error is not None for r in results)
    assert all(r.per is None for r in results)


def test_run_benchmark_strips_suprasegmentals_from_reference_and_hypothesis(monkeypatch):
    fake_client = _FakeClient(response=TTSResult(audio_bytes=b"x", audio_format="mp3", provider="openai", voice="alloy"))
    monkeypatch.setattr(runner_module, "get_client", lambda provider: fake_client)
    # Recognizer output also carries a tone-ish token that should be stripped.
    monkeypatch.setattr(runner_module, "extract_phonemes", lambda result, lang_id: ["ŋ", "w", "i", "ə", "n", "˧˧"])

    dataset = _dataset([("Nguyễn", "ŋ w i ə n ˦ˀ˥", "vietnamese", "vie_latn_hanoi_narrow_filtered.tsv")])
    results = run_benchmark(dataset, ["openai"])

    result = results[0]
    assert result.reference == ["ŋ", "w", "i", "ə", "n"]
    assert result.hypothesis == ["ŋ", "w", "i", "ə", "n"]
    assert result.per == 0.0


def test_results_to_dataframe_and_summary_roundtrip(monkeypatch):
    fake_client = _FakeClient(response=TTSResult(audio_bytes=b"x", audio_format="mp3", provider="openai", voice="alloy"))
    monkeypatch.setattr(runner_module, "get_client", lambda provider: fake_client)
    monkeypatch.setattr(runner_module, "extract_phonemes", lambda result, lang_id: ["a", "b"])

    dataset = _dataset(
        [
            ("word1", "a b", "vietnamese", "vie_latn_hanoi_narrow_filtered.tsv"),
            ("word2", "a x", "vietnamese", "vie_latn_hanoi_narrow_filtered.tsv"),
        ]
    )
    results = run_benchmark(dataset, ["openai"])
    df = results_to_dataframe(results)

    assert list(df.columns) == ["word", "category", "lang_id", "provider", "reference", "hypothesis", "per", "error"]
    assert len(df) == 2

    summary = summarize_by_provider_and_category(df)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["provider"] == "openai"
    assert row["category"] == "vietnamese"
    assert row["n_items"] == 2
    # word1: exact match (0 edits/2 ref). word2: "a x" vs "a b" = 1 sub/2 ref.
    # Edit-weighted corpus PER = 1 / 4.
    assert row["corpus_per"] == pytest.approx(0.25)


def test_summarize_handles_no_successful_results():
    df = results_to_dataframe(
        [
            runner_module.BenchmarkResult(
                word="w", category="c", lang_id="xx", provider="openai",
                reference=["a"], error="boom",
            )
        ]
    )
    summary = summarize_by_provider_and_category(df)
    assert summary.empty
