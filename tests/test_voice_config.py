from pronunciation_benchmark.benchmark.voice_config import resolve_voice


def test_openai_default_voice_used_regardless_of_language():
    assert resolve_voice("openai", "vie") == ("alloy", None)
    assert resolve_voice("openai", "hin") == ("alloy", None)


def test_elevenlabs_default_voice_used_regardless_of_language():
    voice, language_code = resolve_voice("elevenlabs", "yor")
    assert language_code is None
    assert voice


def test_azure_returns_configured_locale_voice():
    assert resolve_voice("azure", "vie") == ("vi-VN-HoaiMyNeural", "vi-VN")


def test_google_returns_configured_locale_voice():
    assert resolve_voice("google", "vie") == ("vi-VN-Standard-A", "vi-VN")


def test_azure_configured_for_all_categories_except_yoruba():
    for lang_id in ["hin", "ben", "tam", "urd", "ara", "fas", "swa", "eng"]:
        voice, language_code = resolve_voice("azure", lang_id)
        assert voice
        assert language_code


def test_google_configured_for_all_categories_except_persian_and_yoruba():
    for lang_id in ["hin", "ben", "tam", "urd", "ara", "swa", "eng"]:
        voice, language_code = resolve_voice("google", lang_id)
        assert voice
        assert language_code


def test_azure_returns_none_for_yoruba():
    # Azure Speech has no Yoruba TTS voice at all.
    assert resolve_voice("azure", "yor") is None


def test_google_returns_none_for_persian_and_yoruba():
    # Google Cloud TTS has no Persian or Yoruba voices at all.
    assert resolve_voice("google", "fas") is None
    assert resolve_voice("google", "yor") is None


def test_unknown_provider_returns_none():
    assert resolve_voice("not-a-provider", "vie") is None
