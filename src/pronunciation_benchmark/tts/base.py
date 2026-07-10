"""Common interface for TTS provider clients."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


def require_env(name: str) -> str:
    """Read a required environment variable, raising a clear error if unset."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. Set it in .env (see .env.example)."
        )
    return value


@dataclass
class TTSResult:
    audio_bytes: bytes
    audio_format: str
    provider: str
    voice: str


class TTSClient(ABC):
    provider_name: str

    @abstractmethod
    def synthesize(
        self, text: str, voice: str, *, language_code: str | None = None
    ) -> TTSResult:
        """Synthesize `text` with the given voice, returning raw audio bytes.

        `language_code` (e.g. "vi-VN") is required by providers whose voice
        selection is locale-scoped (Azure, Google); ignored by providers with
        multilingual voices that auto-detect language (OpenAI, ElevenLabs).
        """
