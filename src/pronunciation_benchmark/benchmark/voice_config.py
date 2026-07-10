"""Per-(provider, language) voice selection for the benchmark runner.

OpenAI and ElevenLabs voices are multilingual and auto-detect language from
the input text (see their `synthesize()` docstrings), so one default voice
works for every language. Azure and Google voices are locale-scoped and
need an explicit (voice, language_code) pair per language - filled in below
for every (provider, language) combination that provider actually supports
(validated for Vietnamese via scripts/smoke_test_tts.py; the rest via
Microsoft's language-support docs and a live query of Google's
`texttospeech.googleapis.com/v1/voices` endpoint). `resolve_voice` returns
None for an unconfigured (provider, lang_id) pair rather than guessing, and
the runner skips that combination instead of failing the run - this is used
deliberately below for languages a provider genuinely doesn't support,
rather than picking an unrelated locale as a stand-in.

Two lang_ids have no case-sensitive dialect signal in their source WikiPron
file (ar_arab_broad.tsv / urd_arab_broad.tsv aren't tagged to a country), so
the locale below is a reasonable-but-arbitrary pick rather than a verified
match: Arabic uses ar-SA (Azure) / ar-XA (Google's only Arabic locale,
documented by Google as Modern Standard Arabic). Urdu uses ur-PK (Azure) /
ur-IN (Google, which doesn't offer ur-PK at all).

"eng" backs the `medical` and `oov` categories (see data/names.py) - both
are English-only, so a single en-US voice covers both.
"""

from __future__ import annotations

DEFAULT_VOICES: dict[str, str] = {
    "openai": "alloy",
    "elevenlabs": "21m00Tcm4TlvDq8ikWAM",
}

# lang_id (ISO 639-3, matches WikiPron filename prefixes and Allosaurus's
# lang_id) -> (voice, language_code)
LOCALE_VOICES: dict[str, dict[str, tuple[str, str]]] = {
    "azure": {
        "vie": ("vi-VN-HoaiMyNeural", "vi-VN"),
        "hin": ("hi-IN-SwaraNeural", "hi-IN"),
        "ben": ("bn-IN-TanishaaNeural", "bn-IN"),
        "tam": ("ta-IN-PallaviNeural", "ta-IN"),
        "urd": ("ur-PK-UzmaNeural", "ur-PK"),
        "ara": ("ar-SA-ZariyahNeural", "ar-SA"),
        "fas": ("fa-IR-DilaraNeural", "fa-IR"),
        "swa": ("sw-KE-ZuriNeural", "sw-KE"),
        "eng": ("en-US-JennyNeural", "en-US"),  # medical/oov categories
        # "yor" intentionally omitted: Azure Speech has no Yoruba TTS voice
        # as of this check (confirmed against Microsoft's language-support
        # docs, not just a single failed lookup) - resolve_voice() returns
        # None and the runner skips azure/yor rather than guessing a voice.
    },
    "google": {
        "vie": ("vi-VN-Standard-A", "vi-VN"),
        "hin": ("hi-IN-Standard-A", "hi-IN"),
        "ben": ("bn-IN-Standard-A", "bn-IN"),
        "tam": ("ta-IN-Standard-A", "ta-IN"),
        "urd": ("ur-IN-Standard-A", "ur-IN"),
        "ara": ("ar-XA-Standard-A", "ar-XA"),
        # Swahili has no Standard/Wavenet/Neural2 tier on Google, only
        # Chirp3-HD (confirmed via a live voices:list API call).
        "swa": ("sw-KE-Chirp3-HD-Kore", "sw-KE"),
        "eng": ("en-US-Standard-A", "en-US"),  # medical/oov categories
        # "fas" and "yor" intentionally omitted: confirmed via a live
        # texttospeech.googleapis.com/v1/voices query that Google Cloud TTS
        # has no Persian or Yoruba voices at all (no fa-* or yo-* language
        # code in the catalog) - resolve_voice() returns None for these and
        # the runner skips google/fas and google/yor rather than guessing.
    },
}


def resolve_voice(provider: str, lang_id: str) -> tuple[str, str | None] | None:
    """Return (voice, language_code) for a provider/language, or None if unconfigured."""
    if provider in DEFAULT_VOICES:
        return DEFAULT_VOICES[provider], None
    return LOCALE_VOICES.get(provider, {}).get(lang_id)
