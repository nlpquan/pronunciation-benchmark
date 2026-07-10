import pandas as pd
import pytest

import pronunciation_benchmark.data.wikidata as wikidata_module
from pronunciation_benchmark.data.wikidata import (
    COUNTRY_PRESETS,
    load_category,
    query_names_by_country,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _sparql_payload(pairs):
    return {
        "results": {
            "bindings": [
                {
                    "person": {"type": "uri", "value": f"http://www.wikidata.org/entity/{qid}"},
                    "nativeName": {"xml:lang": lang, "type": "literal", "value": name},
                }
                for qid, lang, name in pairs
            ]
        }
    }


def test_query_names_by_country_parses_bindings(monkeypatch):
    payload = _sparql_payload([("Q1", "vi", "Nguyễn Văn A"), ("Q2", "vi", "Trần Thị B")])
    monkeypatch.setattr(
        wikidata_module.requests, "get", lambda *a, **k: _FakeResponse(payload)
    )

    df = query_names_by_country("Q881", "vi")

    assert list(df.columns) == ["name", "wikidata_id", "country_qid", "lang_tag"]
    assert len(df) == 2
    assert df.loc[0, "name"] == "Nguyễn Văn A"
    assert df.loc[0, "wikidata_id"] == "Q1"
    assert df.loc[0, "country_qid"] == "Q881"
    assert df.loc[0, "lang_tag"] == "vi"


def test_query_names_by_country_empty_results(monkeypatch):
    monkeypatch.setattr(
        wikidata_module.requests, "get", lambda *a, **k: _FakeResponse(_sparql_payload([]))
    )

    df = query_names_by_country("Q881", "vi")

    assert df.empty
    assert list(df.columns) == ["name", "wikidata_id", "country_qid", "lang_tag"]


def test_load_category_rejects_unknown_category():
    with pytest.raises(KeyError):
        load_category("narnian")


def test_load_category_reads_from_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(wikidata_module, "CACHE_DIR", tmp_path)
    cached = pd.DataFrame(
        {"name": ["Cached Name"], "wikidata_id": ["Q1"], "country_qid": ["Q881"], "lang_tag": ["vi"]}
    )
    cached.to_csv(tmp_path / "vietnamese.csv", index=False, encoding="utf-8")

    def _fail_if_called(*a, **k):
        raise AssertionError("should not hit the network when a cache file exists")

    monkeypatch.setattr(wikidata_module.requests, "get", _fail_if_called)

    df = load_category("vietnamese")

    assert list(df["name"]) == ["Cached Name"]


def test_load_category_writes_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(wikidata_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(wikidata_module, "COUNTRY_PRESETS", {"vietnamese": [("Q881", "vi", "Vietnam")]})
    payload = _sparql_payload([("Q1", "vi", "Nguyễn Văn A")])
    monkeypatch.setattr(
        wikidata_module.requests, "get", lambda *a, **k: _FakeResponse(payload)
    )
    monkeypatch.setattr(wikidata_module.time, "sleep", lambda *_: None)

    df = load_category("vietnamese")

    assert (tmp_path / "vietnamese.csv").exists()
    assert list(df["name"]) == ["Nguyễn Văn A"]


def test_country_presets_cover_project_categories():
    for category, countries in COUNTRY_PRESETS.items():
        assert countries, f"{category} must map to at least one country"
        for qid, lang_tag, label in countries:
            assert qid.startswith("Q")
            assert lang_tag
            assert label
