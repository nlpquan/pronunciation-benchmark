"""Manual smoke test for the audio -> phoneme extraction step.

Loads the real TTS audio saved by scripts/smoke_test_tts.py, runs Allosaurus
phoneme recognition on each, and scores the result against WikiPron's ground
truth IPA for the same word - so you can eyeball both the raw phonemes and
the resulting PER end-to-end, not just confirm the pipeline runs.

The first run downloads Allosaurus's pretrained model (network access
required); on this machine that needs the Kaspersky SSL cert bundle - see
memory/env_kaspersky_ssl_interception.md.

Usage: python scripts/smoke_test_asr.py
(run scripts/smoke_test_tts.py first if data/raw/smoke_test/ is empty)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pronunciation_benchmark.asr.phoneme_recognizer import extract_phonemes  # noqa: E402
from pronunciation_benchmark.data.wikipron import load_tsv  # noqa: E402
from pronunciation_benchmark.scoring.normalize import strip_suprasegmentals  # noqa: E402
from pronunciation_benchmark.scoring.per import phoneme_error_rate, tokenize_ipa  # noqa: E402
from pronunciation_benchmark.tts.base import TTSResult  # noqa: E402

AUDIO_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "smoke_test"
TEST_WORD = "Nguyễn"
LANG_ID = "vie"
WIKIPRON_FILE = "vie_latn_hanoi_narrow_filtered.tsv"

AUDIO_FILES = {
    "openai": ("openai.mp3", "mp3"),
    "elevenlabs": ("elevenlabs.mp3", "mp3"),
    "azure": ("azure.wav", "wav"),
    "google": ("google.mp3", "mp3"),
}


def main() -> None:
    wikipron_df = load_tsv(WIKIPRON_FILE)
    matches = wikipron_df[wikipron_df["word"] == TEST_WORD]
    if matches.empty:
        print(f"No WikiPron ground truth found for {TEST_WORD!r} in {WIKIPRON_FILE}")
        return
    reference = strip_suprasegmentals(tokenize_ipa(matches.iloc[0]["ipa"]))
    print(f"Reference ({TEST_WORD}), suprasegmentals stripped: {' '.join(reference)}\n")

    results = []
    for provider, (filename, audio_format) in AUDIO_FILES.items():
        path = AUDIO_DIR / filename
        if not path.exists():
            print(f"[SKIP] {provider}: no audio at {path} - run scripts/smoke_test_tts.py first")
            continue

        tts_result = TTSResult(
            audio_bytes=path.read_bytes(), audio_format=audio_format, provider=provider, voice=""
        )
        try:
            hypothesis = strip_suprasegmentals(extract_phonemes(tts_result, lang_id=LANG_ID))
        except Exception as exc:  # noqa: BLE001 - smoke test, want to see any failure
            print(f"[FAIL] {provider}: {type(exc).__name__}: {exc}")
            continue

        per = phoneme_error_rate(reference, hypothesis)
        print(f"[ OK ] {provider}: {' '.join(hypothesis)}  (PER={per:.2f})")
        results.append(provider)

    print()
    print(f"{len(results)}/{len(AUDIO_FILES)} providers processed.")


if __name__ == "__main__":
    main()
