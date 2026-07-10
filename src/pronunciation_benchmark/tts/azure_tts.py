"""Azure AI Speech text-to-speech client.

REST reference: POST https://{region}.tts.speech.microsoft.com/cognitiveservices/v1
Auth: Ocp-Apim-Subscription-Key header (resource key directly; no token
exchange needed - the STS bearer-token flow is an alternative Azure supports
but isn't used here since the subscription key works directly for this
endpoint).
Body: SSML XML. Response body is raw audio bytes.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

import requests

from .base import TTSClient, TTSResult, require_env

API_URL_TEMPLATE = "https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"


class AzureTTSClient(TTSClient):
    provider_name = "azure"

    def __init__(self, api_key: str | None = None, region: str | None = None):
        self.api_key = api_key or require_env("AZURE_TTS_KEY")
        self.region = region or require_env("AZURE_TTS_REGION")

    def synthesize(
        self, text: str, voice: str, *, language_code: str | None = None
    ) -> TTSResult:
        if not language_code:
            raise ValueError("AzureTTSClient.synthesize requires language_code (e.g. 'vi-VN')")

        ssml = (
            f"<speak version='1.0' xml:lang='{escape(language_code)}'>"
            f"<voice xml:lang='{escape(language_code)}' name='{escape(voice)}'>"
            f"{escape(text)}"
            f"</voice></speak>"
        )
        response = requests.post(
            API_URL_TEMPLATE.format(region=self.region),
            headers={
                "Ocp-Apim-Subscription-Key": self.api_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
                "User-Agent": "pronunciation-benchmark",
            },
            data=ssml.encode("utf-8"),
            timeout=60,
        )
        response.raise_for_status()
        return TTSResult(
            audio_bytes=response.content,
            audio_format="wav",
            provider=self.provider_name,
            voice=voice,
        )
