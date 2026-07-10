import pronunciation_benchmark.data.lexicons as lexicons_module
from pronunciation_benchmark.data.lexicons import load_common_words, load_medical_terms


class _FakeResponse:
    def __init__(self, payload=None, text=""):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _category_page(titles, cmcontinue=None):
    payload = {"query": {"categorymembers": [{"title": t} for t in titles]}}
    if cmcontinue:
        payload["continue"] = {"cmcontinue": cmcontinue}
    return payload


def test_load_medical_terms_filters_to_clean_alpha_words(monkeypatch, tmp_path):
    monkeypatch.setattr(lexicons_module, "CACHE_DIR", tmp_path)
    pages = [
        _category_page(["pneumonia", "ABC", "A and E", "arthritis-like", "X"]),
        _category_page([]),
    ]
    monkeypatch.setattr(
        lexicons_module.requests, "get", lambda *a, **k: _FakeResponse(pages.pop(0))
    )

    terms = load_medical_terms()

    # "ABC" (all-caps initialism), "A and E" (multi-word phrase),
    # "arthritis-like" (hyphenated compound), and "X" (below min_length) are
    # all dropped - only clean single alphabetic words survive.
    assert terms == {"pneumonia"}


def test_load_medical_terms_paginates_across_categories(monkeypatch, tmp_path):
    monkeypatch.setattr(lexicons_module, "CACHE_DIR", tmp_path)
    pages = [
        _category_page(["pneumonia"], cmcontinue="next"),
        _category_page(["arthritis"]),
        _category_page(["ibuprofen"]),
    ]
    monkeypatch.setattr(
        lexicons_module.requests, "get", lambda *a, **k: _FakeResponse(pages.pop(0))
    )

    terms = load_medical_terms()

    assert terms == {"pneumonia", "arthritis", "ibuprofen"}


def test_load_medical_terms_reads_from_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(lexicons_module, "CACHE_DIR", tmp_path)
    (tmp_path / "medical_terms.txt").write_text("cached_term", encoding="utf-8")

    def _fail_if_called(*a, **k):
        raise AssertionError("should not hit the network when a cache file exists")

    monkeypatch.setattr(lexicons_module.requests, "get", _fail_if_called)

    assert load_medical_terms() == {"cached_term"}


def test_load_common_words_lowercases_and_dedupes(monkeypatch, tmp_path):
    monkeypatch.setattr(lexicons_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        lexicons_module.requests,
        "get",
        lambda *a, **k: _FakeResponse(text="The\nof\nTHE\n\n"),
    )

    words = load_common_words()

    assert words == {"the", "of"}


def test_load_common_words_reads_from_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(lexicons_module, "CACHE_DIR", tmp_path)
    (tmp_path / "common_words.txt").write_text("the\nof", encoding="utf-8")

    def _fail_if_called(*a, **k):
        raise AssertionError("should not hit the network when a cache file exists")

    monkeypatch.setattr(lexicons_module.requests, "get", _fail_if_called)

    assert load_common_words() == {"the", "of"}
