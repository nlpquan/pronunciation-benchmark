"""ElevenLabs text-to-speech client.

REST reference: POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}
Auth: xi-api-key header
Response body is raw audio bytes (application/octet-stream).
"""

from __future__ import annotations

import requests

from .base import TTSClient, TTSResult, require_env

API_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


class ElevenLabsTTSClient(TTSClient):
    provider_name = "elevenlabs"

    def __init__(self, api_key: str | None = None, model_id: str = "eleven_multilingual_v2"):
        self.api_key = api_key or require_env("ELEVENLABS_API_KEY")
        self.model_id = model_id

    def synthesize(
        self, text: str, voice: str, *, language_code: str | None = None
    ) -> TTSResult:
        # `voice` is an ElevenLabs voice_id (not a language-scoped name);
        # eleven_multilingual_v2 auto-detects language from the input text,
        # so language_code is accepted for interface consistency but unused.
        response = requests.post(
            API_URL_TEMPLATE.format(voice_id=voice),
            headers={"xi-api-key": self.api_key},
            json={"text": text, "model_id": self.model_id},
            params={"output_format": "mp3_44100_128"},
            timeout=60,
        )
        response.raise_for_status()
        return TTSResult(
            audio_bytes=response.content,
            audio_format="mp3",
            provider=self.provider_name,
            voice=voice,
        )
