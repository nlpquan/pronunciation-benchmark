"""End-to-end benchmark runner: dataset -> TTS -> phoneme extraction -> PER.

Ties together every Phase 1 module into one pipeline: for each benchmark
item and each configured TTS provider, synthesizes audio, extracts the
phoneme sequence actually produced, strips suprasegmentals from both sides
(see scoring.normalize - PER is segmental-only by design), and scores PER
against WikiPron's reference.

Failures are per-(item, provider) and never abort the run: an unavailable
provider (e.g. missing API key), an unconfigured voice for a language, or
a synthesis/recognition error are all recorded as an `error` on that one
BenchmarkResult so a partial run still produces results for everything
that did work.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..asr.phoneme_recognizer import extract_phonemes
from ..scoring.normalize import strip_suprasegmentals
from ..scoring.per import corpus_phoneme_error_rate, phoneme_error_rate, tokenize_ipa
from ..tts import get_client
from ..tts.base import TTSClient
from .voice_config import resolve_voice


def lang_id_from_source_file(source_file: str) -> str:
    """WikiPron filenames are prefixed with their ISO 639-3 code, e.g.
    "vie_latn_hanoi_narrow_filtered.tsv" -> "vie".

    This is the code used for voice config lookup, result grouping, etc.
    It usually also matches Allosaurus's `lang_id`, but not always - see
    ALLOSAURUS_LANG_ID below for the exceptions.
    """
    return source_file.split("_")[0]


# WikiPron identifies Arabic/Persian/Swahili by their macrolanguage code
# ("ara", "fas", "swa"), but Allosaurus's bundled inventory only registers
# the specific-language code it actually has phone data for ("arb" Standard
# Arabic, "pes" Iranian Persian, "swh" Swahili) - confirmed against
# venv/Lib/site-packages/allosaurus/pretrained/uni2005/inventory/index.json.
# Passing the WikiPron code straight through raises
# `AssertionError: Language ara is not available!` from allosaurus's
# is_available() check. hin/ben/tam/urd/vie/yor need no mapping - Allosaurus
# registers those under the same ISO 639-3 code WikiPron uses.
ALLOSAURUS_LANG_ID: dict[str, str] = {
    "ara": "arb",
    "fas": "pes",
    "swa": "swh",
}


def allosaurus_lang_id(lang_id: str) -> str:
    """Map a WikiPron/dataset lang_id to the lang_id Allosaurus expects."""
    return ALLOSAURUS_LANG_ID.get(lang_id, lang_id)


@dataclass
class BenchmarkResult:
    word: str
    category: str
    lang_id: str
    provider: str
    reference: list[str]
    hypothesis: list[str] | None = None
    per: float | None = None
    error: str | None = None


def _build_clients(providers: list[str]) -> tuple[dict[str, TTSClient], dict[str, str]]:
    """Construct one client per provider, separating out any that fail
    (typically a missing API key) so the main loop can skip them cheaply
    instead of retrying the same failure on every dataset item.
    """
    clients: dict[str, TTSClient] = {}
    unavailable: dict[str, str] = {}
    for provider in providers:
        try:
            clients[provider] = get_client(provider)
        except Exception as exc:  # noqa: BLE001 - want to record any construction failure, not just RuntimeError
            unavailable[provider] = f"{type(exc).__name__}: {exc}"
    return clients, unavailable


def run_benchmark(dataset: pd.DataFrame, providers: list[str]) -> list[BenchmarkResult]:
    """Run every dataset item through every provider.

    `dataset` must have the columns produced by
    `data.names.build_benchmark_dataset`: word, ipa, category, source_file.
    Returns one BenchmarkResult per (item, provider) pair.
    """
    clients, unavailable = _build_clients(providers)
    results: list[BenchmarkResult] = []

    for row in dataset.itertuples(index=False):
        lang_id = lang_id_from_source_file(row.source_file)
        reference = strip_suprasegmentals(tokenize_ipa(row.ipa))

        for provider in providers:
            if provider in unavailable:
                results.append(
                    BenchmarkResult(
                        word=row.word, category=row.category, lang_id=lang_id,
                        provider=provider, reference=reference, error=unavailable[provider],
                    )
                )
                continue

            voice_config = resolve_voice(provider, lang_id)
            if voice_config is None:
                results.append(
                    BenchmarkResult(
                        word=row.word, category=row.category, lang_id=lang_id,
                        provider=provider, reference=reference,
                        error=f"no {provider} voice configured for lang_id={lang_id!r}",
                    )
                )
                continue
            voice, language_code = voice_config

            try:
                tts_result = clients[provider].synthesize(row.word, voice=voice, language_code=language_code)
                hypothesis = strip_suprasegmentals(
                    extract_phonemes(tts_result, lang_id=allosaurus_lang_id(lang_id))
                )
            except Exception as exc:  # noqa: BLE001 - one bad item shouldn't abort the run
                results.append(
                    BenchmarkResult(
                        word=row.word, category=row.category, lang_id=lang_id,
                        provider=provider, reference=reference,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            results.append(
                BenchmarkResult(
                    word=row.word, category=row.category, lang_id=lang_id, provider=provider,
                    reference=reference, hypothesis=hypothesis,
                    per=phoneme_error_rate(reference, hypothesis),
                )
            )

    return results


def results_to_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "word": r.word,
                "category": r.category,
                "lang_id": r.lang_id,
                "provider": r.provider,
                "reference": " ".join(r.reference),
                "hypothesis": " ".join(r.hypothesis) if r.hypothesis is not None else None,
                "per": r.per,
                "error": r.error,
            }
            for r in results
        ]
    )


def summarize_by_provider_and_category(df: pd.DataFrame) -> pd.DataFrame:
    """Corpus-level PER (edit-weighted, not a mean of per-item PER - see
    scoring.per.corpus_phoneme_error_rate) for every (provider, category)
    combination that has at least one successfully scored item.
    """
    scored = df[df["per"].notna()]
    rows = []
    for (provider, category), group in scored.groupby(["provider", "category"]):
        pairs = [(ref.split(), hyp.split()) for ref, hyp in zip(group["reference"], group["hypothesis"])]
        rows.append(
            {
                "provider": provider,
                "category": category,
                "n_items": len(group),
                "corpus_per": corpus_phoneme_error_rate(pairs),
            }
        )
    return pd.DataFrame(rows)
