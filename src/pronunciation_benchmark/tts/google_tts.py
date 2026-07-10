"""Google Cloud text-to-speech client.

REST reference: POST https://texttospeech.googleapis.com/v1/text:synthesize
Auth: API key as a query param (?key=...). Google's REST docs describe OAuth
Bearer auth too, but a plain API key (from Cloud Console / AI Studio) works
for this endpoint and needs no service-account setup - simpler for a
zero-budget project.
Response body is JSON with base64-encoded audio in `audioContent`.
"""

from __future__ import annotations

import base64

import requests

from .base import TTSClient, TTSResult, require_env

API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"


class GoogleTTSClient(TTSClient):
    provider_name = "google"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or require_env("GOOGLE_TTS_API_KEY")

    def synthesize(
        self, text: str, voice: str, *, language_code: str | None = None
    ) -> TTSResult:
        if not language_code:
            raise ValueError("GoogleTTSClient.synthesize requires language_code (e.g. 'vi-VN')")

        response = requests.post(
            API_URL,
            params={"key": self.api_key},
            json={
                "input": {"text": text},
                "voice": {"languageCode": language_code, "name": voice},
                "audioConfig": {"audioEncoding": "MP3"},
            },
            timeout=60,
        )
        response.raise_for_status()
        audio_bytes = base64.b64decode(response.json()["audioContent"])
        return TTSResult(
            audio_bytes=audio_bytes,
            audio_format="mp3",
            provider=self.provider_name,
            voice=voice,
        )
