"""Manual smoke test for the TTS provider clients against real APIs.

Synthesizes a short Vietnamese test word with every provider that has
credentials set in .env, saving each result under data/raw/smoke_test/ so
you can listen and confirm it actually sounds right. Skips (doesn't fail)
any provider whose env vars aren't set - run this with however many
providers you've configured.

Usage: python scripts/smoke_test_tts.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pronunciation_benchmark.tts import get_client  # noqa: E402

load_dotenv()

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "smoke_test"
TEST_TEXT = "Nguyễn"

# (provider, required env vars, voice, language_code)
# ElevenLabs/Azure/Google voice IDs are account- or provider-specific;
# defaults below are public/standard voices that work on any account.
# Override via env var if you want a different voice.
PROVIDER_CONFIGS = [
    ("openai", ["OPENAI_API_KEY"], "alloy", None),
    ("elevenlabs", ["ELEVENLABS_API_KEY"], os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"), None),
    ("azure", ["AZURE_TTS_KEY", "AZURE_TTS_REGION"], "vi-VN-HoaiMyNeural", "vi-VN"),
    ("google", ["GOOGLE_TTS_API_KEY"], "vi-VN-Standard-A", "vi-VN"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for provider, required_vars, voice, language_code in PROVIDER_CONFIGS:
        missing = [v for v in required_vars if not os.environ.get(v)]
        if missing:
            print(f"[SKIP] {provider}: missing {', '.join(missing)}")
            continue

        try:
            client = get_client(provider)
            result = client.synthesize(TEST_TEXT, voice=voice, language_code=language_code)
        except Exception as exc:  # noqa: BLE001 - smoke test, want to see any failure
            print(f"[FAIL] {provider}: {type(exc).__name__}: {exc}")
            results.append((provider, False))
            continue

        out_path = OUTPUT_DIR / f"{provider}.{result.audio_format}"
        out_path.write_bytes(result.audio_bytes)
        print(f"[ OK ] {provider}: {len(result.audio_bytes)} bytes -> {out_path}")
        results.append((provider, True))

    print()
    passed = sum(1 for _, ok in results if ok)
    print(f"{passed}/{len(results)} attempted providers succeeded.")
    if not results:
        print("No providers configured - fill in .env and retry.")


if __name__ == "__main__":
    main()
