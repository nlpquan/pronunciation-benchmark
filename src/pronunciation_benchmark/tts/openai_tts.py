"""OpenAI text-to-speech client.

REST reference: POST https://api.openai.com/v1/audio/speech
Auth: Authorization: Bearer <API key>
Response body is raw audio bytes (format set via response_format).
"""

from __future__ import annotations

import requests

from .base import TTSClient, TTSResult, require_env

API_URL = "https://api.openai.com/v1/audio/speech"


class OpenAITTSClient(TTSClient):
    provider_name = "openai"

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini-tts"):
        self.api_key = api_key or require_env("OPENAI_API_KEY")
        self.model = model

    def synthesize(
        self, text: str, voice: str = "alloy", *, language_code: str | None = None
    ) -> TTSResult:
        # OpenAI's TTS voices are multilingual and auto-detect language from
        # the input text, so language_code is accepted for interface
        # consistency but unused here.
        # mp3 rather than wav: OpenAI's wav response sets the RIFF/data chunk
        # size fields to a placeholder (0xFFFFFFFF) instead of the true size
        # (a streaming-response artifact), which confuses tools that trust
        # the WAV header's declared length instead of computing from actual
        # file size. mp3 has no such header field, so it sidesteps the issue.
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": text,
                "voice": voice,
                "response_format": "mp3",
            },
            timeout=60,
        )
        response.raise_for_status()
        return TTSResult(
            audio_bytes=response.content,
            audio_format="mp3",
            provider=self.provider_name,
            voice=voice,
        )
