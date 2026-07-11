import pandas as pd

from pronunciation_benchmark.leaderboard.data import (
    best_tts_vs_g2p,
    score_g2p_predictions,
    score_tts_results,
)


def _tts_results_df():
    return pd.DataFrame(
        [
            {"provider": "openai", "lang_id": "vie", "category": "vietnamese", "reference": "a b c", "hypothesis": "a b c", "per": 0.0},
            {"provider": "openai", "lang_id": "vie", "category": "vietnamese", "reference": "a b", "hypothesis": "a x", "per": 0.5},
            {"provider": "azure", "lang_id": "vie", "category": "vietnamese", "reference": "a b c", "hypothesis": "a b", "per": 1 / 3},
            {"provider": "azure", "lang_id": "yor", "category": "african", "reference": "a b", "hypothesis": "", "per": None},
        ]
    )


def test_score_tts_results_excludes_unscored_rows():
    results = _tts_results_df()
    scores = score_tts_results(results, ["provider", "lang_id"])
    # azure/yor has per=None (unavailable voice) and must be dropped, not counted as 0.
    assert not ((scores["provider"] == "azure") & (scores["lang_id"] == "yor")).any()


def test_score_tts_results_corpus_per_is_edit_weighted_not_averaged():
    results = _tts_results_df()
    scores = score_tts_results(results, ["provider", "lang_id"])
    row = scores[(scores["provider"] == "openai") & (scores["lang_id"] == "vie")].iloc[0]
    # total edits (0 + 1) / total ref length (3 + 2) = 0.2, not mean(0.0, 0.5) = 0.25.
    assert row["n_items"] == 2
    assert row["corpus_per"] == 0.2


def test_score_tts_results_can_group_by_single_column():
    results = _tts_results_df()
    scores = score_tts_results(results, ["provider"])
    assert set(scores["provider"]) == {"openai", "azure"}


def test_score_g2p_predictions_strips_suprasegmentals():
    predictions = pd.DataFrame(
        [
            {"lang_id": "vie", "word": "a", "reference_ipa": "a ˧˧", "predicted_ipa": "a ˧˧"},
            {"lang_id": "vie", "word": "b", "reference_ipa": "b ˦ˀ˥", "predicted_ipa": "b"},
        ]
    )
    scores = score_g2p_predictions(predictions)
    row = scores[scores["lang_id"] == "vie"].iloc[0]
    # tone tokens are stripped before scoring, so both rows should match exactly.
    assert row["corpus_per"] == 0.0


def test_best_tts_vs_g2p_picks_lowest_per_provider_and_signs_improvement():
    tts_scores = pd.DataFrame(
        [
            {"provider": "openai", "lang_id": "vie", "n_items": 10, "corpus_per": 0.7},
            {"provider": "azure", "lang_id": "vie", "n_items": 10, "corpus_per": 0.9},
        ]
    )
    g2p_scores = pd.DataFrame([{"lang_id": "vie", "n_items": 10, "corpus_per": 0.1}])

    comparison = best_tts_vs_g2p(tts_scores, g2p_scores)
    row = comparison[comparison["lang_id"] == "vie"].iloc[0]

    assert row["best_tts_provider"] == "openai"
    assert row["best_tts_per"] == 0.7
    assert row["improvement"] == 0.6
