"""Factory for TTS provider clients, keyed by provider name."""

from __future__ import annotations

from .azure_tts import AzureTTSClient
from .base import TTSClient
from .elevenlabs_tts import ElevenLabsTTSClient
from .google_tts import GoogleTTSClient
from .openai_tts import OpenAITTSClient

PROVIDERS: dict[str, type[TTSClient]] = {
    "openai": OpenAITTSClient,
    "elevenlabs": ElevenLabsTTSClient,
    "azure": AzureTTSClient,
    "google": GoogleTTSClient,
}


def get_client(provider: str, **kwargs) -> TTSClient:
    try:
        cls = PROVIDERS[provider]
    except KeyError:
        raise KeyError(f"Unknown TTS provider {provider!r}; choose from {sorted(PROVIDERS)}") from None
    return cls(**kwargs)
