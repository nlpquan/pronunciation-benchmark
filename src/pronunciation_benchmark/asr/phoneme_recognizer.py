"""Audio -> phoneme extraction via Allosaurus universal phone recognition.

Allosaurus (https://github.com/xinjli/allosaurus) is an unconstrained phone
recognizer: it decodes whatever phones its acoustic model hears, unlike
forced alignment (which presupposes the reference sequence is correct and
only locates time boundaries for it). Unconstrained recognition is the
property this benchmark needs - if a TTS system mispronounces a name, only
an unconstrained recognizer can surface that in its output.

`lang_id` optionally narrows the phone inventory to a target language's
allophone set for higher accuracy than the universal "ipa" inventory, using
Allosaurus's ISO 639-3 codes. These usually match WikiPron's language codes
(e.g. "vie", "hin"; see data/wikipron.py's LANGUAGE_PRESETS), but not always
- callers going from a WikiPron/dataset lang_id should map it through
benchmark.runner.allosaurus_lang_id() first (e.g. WikiPron's "ara"/"fas"/
"swa" macrolanguage codes need Allosaurus's "arb"/"pes"/"swh").
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import soundfile as sf
from allosaurus.app import read_recognizer

from ..tts.base import TTSResult

_recognizer = None


def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = read_recognizer()
    return _recognizer


def _to_pcm16_wav_bytes(audio_bytes: bytes) -> bytes:
    """Decode arbitrary audio bytes (mp3, wav, ...) to a PCM16 WAV file.

    Allosaurus only reads WAV, via Python's stdlib `wave` module, so every
    provider's output is normalized to that regardless of its original
    format (soundfile's bundled libsndfile handles mp3 decoding directly,
    no ffmpeg needed).
    """
    data, samplerate = sf.read(io.BytesIO(audio_bytes))
    buffer = io.BytesIO()
    sf.write(buffer, data, samplerate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def extract_phonemes(result: TTSResult, *, lang_id: str = "ipa") -> list[str]:
    """Recognize the phoneme sequence actually present in a TTS result's audio.

    Returns a list of phoneme tokens in the same space-tokenized shape as
    `scoring.per.tokenize_ipa`, so it can be scored directly against
    WikiPron reference IPA.
    """
    wav_bytes = _to_pcm16_wav_bytes(result.audio_bytes)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        temp_path = f.name

    try:
        phones = _get_recognizer().recognize(temp_path, lang_id=lang_id)
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return phones.split()
