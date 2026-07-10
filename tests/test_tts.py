import base64

import pytest

import pronunciation_benchmark.tts.azure_tts as azure_module
import pronunciation_benchmark.tts.elevenlabs_tts as elevenlabs_module
import pronunciation_benchmark.tts.google_tts as google_module
import pronunciation_benchmark.tts.openai_tts as openai_module
from pronunciation_benchmark.tts import get_client
from pronunciation_benchmark.tts.azure_tts import AzureTTSClient
from pronunciation_benchmark.tts.base import require_env
from pronunciation_benchmark.tts.elevenlabs_tts import ElevenLabsTTSClient
from pronunciation_benchmark.tts.google_tts import GoogleTTSClient
from pronunciation_benchmark.tts.openai_tts import OpenAITTSClient
from pronunciation_benchmark.tts.registry import PROVIDERS


class _FakeResponse:
    def __init__(self, content=b"", json_body=None, status=200):
        self.content = content
        self._json_body = json_body
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_body


def test_require_env_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.delenv("SOME_MISSING_VAR", raising=False)
    with pytest.raises(RuntimeError, match="SOME_MISSING_VAR"):
        require_env("SOME_MISSING_VAR")


def test_require_env_returns_value(monkeypatch):
    monkeypatch.setenv("SOME_VAR", "value123")
    assert require_env("SOME_VAR") == "value123"


def test_openai_synthesize_sends_expected_request(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResponse(content=b"FAKEWAV")

    monkeypatch.setattr(openai_module.requests, "post", fake_post)

    client = OpenAITTSClient(api_key="sk-test")
    result = client.synthesize("hello", voice="alloy")

    assert captured["url"] == "https://api.openai.com/v1/audio/speech"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["input"] == "hello"
    assert captured["json"]["voice"] == "alloy"
    assert result.audio_bytes == b"FAKEWAV"
    assert result.provider == "openai"


def test_elevenlabs_synthesize_sends_expected_request(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, params=None, timeout=None, **kwargs):
        captured.update(url=url, headers=headers, json=json, params=params)
        return _FakeResponse(content=b"FAKEMP3")

    monkeypatch.setattr(elevenlabs_module.requests, "post", fake_post)

    client = ElevenLabsTTSClient(api_key="el-test")
    result = client.synthesize("hello", voice="voice123")

    assert captured["url"] == "https://api.elevenlabs.io/v1/text-to-speech/voice123"
    assert captured["headers"]["xi-api-key"] == "el-test"
    assert captured["json"]["text"] == "hello"
    assert result.audio_bytes == b"FAKEMP3"
    assert result.provider == "elevenlabs"


def test_azure_synthesize_builds_ssml_and_sends_expected_request(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, data=None, timeout=None, **kwargs):
        captured.update(url=url, headers=headers, data=data)
        return _FakeResponse(content=b"FAKERIFF")

    monkeypatch.setattr(azure_module.requests, "post", fake_post)

    client = AzureTTSClient(api_key="az-test", region="eastus")
    result = client.synthesize("hello", voice="vi-VN-HoaiMyNeural", language_code="vi-VN")

    assert captured["url"] == "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1"
    assert captured["headers"]["Ocp-Apim-Subscription-Key"] == "az-test"
    ssml = captured["data"].decode("utf-8")
    assert "vi-VN-HoaiMyNeural" in ssml
    assert "hello" in ssml
    assert result.audio_bytes == b"FAKERIFF"


def test_azure_synthesize_requires_language_code(monkeypatch):
    client = AzureTTSClient(api_key="az-test", region="eastus")
    with pytest.raises(ValueError, match="language_code"):
        client.synthesize("hello", voice="vi-VN-HoaiMyNeural")


def test_azure_ssml_escapes_special_characters(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, data=None, timeout=None, **kwargs):
        captured["data"] = data
        return _FakeResponse(content=b"x")

    monkeypatch.setattr(azure_module.requests, "post", fake_post)

    client = AzureTTSClient(api_key="az-test", region="eastus")
    client.synthesize("A & B < C", voice="v", language_code="en-US")

    ssml = captured["data"].decode("utf-8")
    assert "&amp;" in ssml
    assert "&lt;" in ssml


def test_google_synthesize_sends_expected_request_and_decodes_base64(monkeypatch):
    captured = {}
    audio_bytes = b"FAKEAUDIO"

    def fake_post(url, params=None, json=None, timeout=None, **kwargs):
        captured.update(url=url, params=params, json=json)
        return _FakeResponse(json_body={"audioContent": base64.b64encode(audio_bytes).decode("ascii")})

    monkeypatch.setattr(google_module.requests, "post", fake_post)

    client = GoogleTTSClient(api_key="gkey")
    result = client.synthesize("hello", voice="vi-VN-Standard-A", language_code="vi-VN")

    assert captured["url"] == "https://texttospeech.googleapis.com/v1/text:synthesize"
    assert captured["params"] == {"key": "gkey"}
    assert captured["json"]["voice"] == {"languageCode": "vi-VN", "name": "vi-VN-Standard-A"}
    assert result.audio_bytes == audio_bytes


def test_google_synthesize_requires_language_code():
    client = GoogleTTSClient(api_key="gkey")
    with pytest.raises(ValueError, match="language_code"):
        client.synthesize("hello", voice="vi-VN-Standard-A")


def test_registry_get_client_dispatches_by_provider_name():
    client = get_client("openai", api_key="sk-test")
    assert isinstance(client, OpenAITTSClient)


def test_registry_rejects_unknown_provider():
    with pytest.raises(KeyError):
        get_client("not-a-real-provider")


def test_registry_covers_all_project_candidate_providers():
    assert set(PROVIDERS) == {"openai", "elevenlabs", "azure", "google"}
