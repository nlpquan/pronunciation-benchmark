import io
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

import pronunciation_benchmark.asr.phoneme_recognizer as recognizer_module
from pronunciation_benchmark.asr.phoneme_recognizer import _to_pcm16_wav_bytes, extract_phonemes
from pronunciation_benchmark.tts.base import TTSResult


def _sine_wave_bytes(samplerate: int = 16000, duration: float = 0.1) -> bytes:
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    data = (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    buffer = io.BytesIO()
    sf.write(buffer, data, samplerate, format="WAV")
    return buffer.getvalue()


class _FakeRecognizer:
    def __init__(self, output: str = "a b c"):
        self.output = output
        self.received_path: str | None = None
        self.received_lang_id: str | None = None

    def recognize(self, path, lang_id="ipa"):
        self.received_path = path
        self.received_lang_id = lang_id
        return self.output


def test_to_pcm16_wav_bytes_produces_valid_pcm16_wav():
    converted = _to_pcm16_wav_bytes(_sine_wave_bytes())

    with wave.open(io.BytesIO(converted)) as wf:
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() > 0


def test_extract_phonemes_tokenizes_recognizer_output(monkeypatch):
    fake = _FakeRecognizer(output="ŋ w i ə n")
    monkeypatch.setattr(recognizer_module, "_get_recognizer", lambda: fake)

    result = TTSResult(audio_bytes=_sine_wave_bytes(), audio_format="wav", provider="test", voice="v")
    phonemes = extract_phonemes(result, lang_id="vie")

    assert phonemes == ["ŋ", "w", "i", "ə", "n"]
    assert fake.received_lang_id == "vie"


def test_extract_phonemes_defaults_to_universal_ipa_inventory(monkeypatch):
    fake = _FakeRecognizer()
    monkeypatch.setattr(recognizer_module, "_get_recognizer", lambda: fake)

    result = TTSResult(audio_bytes=_sine_wave_bytes(), audio_format="wav", provider="test", voice="v")
    extract_phonemes(result)

    assert fake.received_lang_id == "ipa"


def test_extract_phonemes_cleans_up_temp_file(monkeypatch):
    fake = _FakeRecognizer()
    monkeypatch.setattr(recognizer_module, "_get_recognizer", lambda: fake)

    result = TTSResult(audio_bytes=_sine_wave_bytes(), audio_format="wav", provider="test", voice="v")
    extract_phonemes(result)

    assert fake.received_path is not None
    assert not Path(fake.received_path).exists()
