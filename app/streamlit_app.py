"""Public leaderboard UI: TTS pronunciation-accuracy rankings + the G2P-vs-TTS
comparison from Phase 2.

Reads the two published result CSVs (data/results/benchmark_results.csv,
data/results/g2p_predictions.csv) and scores them with
pronunciation_benchmark.leaderboard.data - the exact same functions
scripts/compare_g2p_vs_tts.py uses - so the numbers shown here can never
silently drift from the numbers in WRITEUP.md.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from pronunciation_benchmark.leaderboard.data import (
    best_tts_vs_g2p,
    load_g2p_predictions,
    load_tts_results,
    score_g2p_predictions,
    score_tts_results,
)

RESULTS_PATH = Path(__file__).resolve().parents[1] / "data" / "results" / "benchmark_results.csv"
PREDICTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "results" / "g2p_predictions.csv"

# Fixed categorical color assignment (never cycled) - blue/aqua/yellow for the
# three TTS providers, violet for the fine-tuned G2P model, matching the
# project's palette slot order.
PROVIDER_COLORS = {
    "openai": "#2a78d6",
    "azure": "#1baf7a",
    "google": "#eda100",
}
G2P_COLOR = "#4a3aa7"

PROVIDER_LABELS = {"openai": "OpenAI TTS", "azure": "Azure TTS", "google": "Google Cloud TTS"}
CATEGORY_LABELS = {
    "vietnamese": "Vietnamese",
    "south_asian": "South Asian",
    "middle_eastern": "Middle Eastern",
    "african": "African",
    "medical": "Medical / Pharma (English)",
    "oov": "Common OOV (English)",
}
LANG_LABELS = {
    "vie": "Vietnamese",
    "hin": "Hindi",
    "ben": "Bengali",
    "tam": "Tamil",
    "urd": "Urdu",
    "ara": "Arabic",
    "fas": "Persian",
    "yor": "Yoruba",
    "swa": "Swahili",
    "eng": "English",
}


@st.cache_data
def load_data():
    results = load_tts_results(RESULTS_PATH)
    predictions = load_g2p_predictions(PREDICTIONS_PATH) if PREDICTIONS_PATH.exists() else None
    return results, predictions


def provider_scale(providers: list[str]) -> alt.Scale:
    return alt.Scale(domain=providers, range=[PROVIDER_COLORS[p] for p in providers])


st.set_page_config(page_title="Pronunciation Benchmark Leaderboard", page_icon="\U0001F5E3", layout="wide")

st.title("Open Pronunciation Benchmark for Voice AI")
st.markdown(
    "Phoneme Error Rate (PER) of commercial TTS providers on non-Western names and "
    "words, scored against dictionary-sourced IPA ground truth from "
    "[WikiPron](https://github.com/CUNY-CL/wikipron). Lower PER is better. "
    "Full methodology and findings are in this repository's `WRITEUP.md`."
)

if not RESULTS_PATH.exists():
    st.error(f"No results found at `{RESULTS_PATH}`. Run `scripts/run_benchmark.py` first.")
    st.stop()

results, predictions = load_data()
providers_present = sorted(results["provider"].dropna().unique())

# --- Filters (one row, above the charts) ---
filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    category_choice = st.multiselect(
        "Category",
        options=list(CATEGORY_LABELS.keys()),
        default=list(CATEGORY_LABELS.keys()),
        format_func=lambda c: CATEGORY_LABELS.get(c, c),
    )
with filter_col2:
    lang_choice = st.multiselect(
        "Language",
        options=sorted(results["lang_id"].unique()),
        default=sorted(results["lang_id"].unique()),
        format_func=lambda code: LANG_LABELS.get(code, code),
    )

filtered = results[results["category"].isin(category_choice) & results["lang_id"].isin(lang_choice)]

if filtered.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# --- Overall leaderboard: providers ranked by corpus PER ---
st.header("Provider ranking")

overall = score_tts_results(filtered, ["provider"]).sort_values("corpus_per")
overall["provider_label"] = overall["provider"].map(PROVIDER_LABELS)

rank_chart = (
    alt.Chart(overall)
    .mark_bar(cornerRadiusEnd=4)
    .encode(
        x=alt.X("corpus_per:Q", title="Corpus PER (lower is better)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("provider_label:N", sort="-x", title=None),
        color=alt.Color("provider:N", scale=provider_scale(providers_present), legend=None),
        tooltip=[
            alt.Tooltip("provider_label:N", title="Provider"),
            alt.Tooltip("corpus_per:Q", title="Corpus PER", format=".3f"),
            alt.Tooltip("n_items:Q", title="Items scored"),
        ],
    )
    .properties(height=32 * len(overall) + 20)
)
st.altair_chart(rank_chart, use_container_width=True)
st.caption(
    "Corpus PER = total edits / total reference phonemes across all scored items "
    "(edit-weighted, not a mean of per-item PER)."
)

# --- Breakdown by category ---
st.header("By category")

by_category = score_tts_results(filtered, ["provider", "category"])
by_category["category_label"] = by_category["category"].map(CATEGORY_LABELS)
by_category["provider_label"] = by_category["provider"].map(PROVIDER_LABELS)

category_chart = (
    alt.Chart(by_category)
    .mark_bar()
    .encode(
        x=alt.X("provider_label:N", title=None, axis=None),
        y=alt.Y("corpus_per:Q", title="Corpus PER", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("provider:N", scale=provider_scale(providers_present), title="Provider"),
        column=alt.Column("category_label:N", title=None, header=alt.Header(labelAngle=-30, labelAlign="right")),
        tooltip=[
            alt.Tooltip("category_label:N", title="Category"),
            alt.Tooltip("provider_label:N", title="Provider"),
            alt.Tooltip("corpus_per:Q", title="Corpus PER", format=".3f"),
            alt.Tooltip("n_items:Q", title="Items scored"),
        ],
    )
    .properties(width=90)
)
st.altair_chart(category_chart, use_container_width=False)

with st.expander("Table view"):
    table = by_category[["category_label", "provider_label", "n_items", "corpus_per"]].sort_values(
        ["category_label", "corpus_per"]
    )
    st.dataframe(
        table.rename(columns={"category_label": "Category", "provider_label": "Provider", "n_items": "Items", "corpus_per": "Corpus PER"}),
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    "Some (provider, language) pairs are absent by provider limitation, not filtered out: "
    "Azure has no Yoruba voice; Google has neither Persian nor Yoruba. ElevenLabs is excluded "
    "entirely (no funded account)."
)

# --- G2P vs TTS comparison ---
st.header("Fine-tuned G2P vs. best TTS provider")

if predictions is None:
    st.info(
        f"No G2P predictions found at `{PREDICTIONS_PATH}`. Run the training notebook "
        "(`notebooks/g2p_finetune_kaggle.ipynb`) and drop its output CSV there to show this section."
    )
else:
    tts_by_provider_lang = score_tts_results(results, ["provider", "lang_id"])
    g2p_scores = score_g2p_predictions(predictions)
    comparison = best_tts_vs_g2p(tts_by_provider_lang, g2p_scores).dropna(subset=["g2p_per"])
    comparison["lang_label"] = comparison["lang_id"].map(LANG_LABELS)
    comparison["best_tts_label"] = comparison["best_tts_provider"].map(PROVIDER_LABELS)

    long = pd.concat(
        [
            comparison[["lang_label", "best_tts_per"]].rename(columns={"best_tts_per": "corpus_per"}).assign(series="Best TTS"),
            comparison[["lang_label", "g2p_per"]].rename(columns={"g2p_per": "corpus_per"}).assign(series="Fine-tuned G2P"),
        ]
    )

    comparison_chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("series:N", title=None, axis=None),
            y=alt.Y("corpus_per:Q", title="Corpus PER", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Best TTS", "Fine-tuned G2P"], range=["#2a78d6", G2P_COLOR]),
                title=None,
            ),
            column=alt.Column("lang_label:N", title=None, header=alt.Header(labelAngle=-30, labelAlign="right")),
            tooltip=[alt.Tooltip("lang_label:N", title="Language"), alt.Tooltip("series:N"), alt.Tooltip("corpus_per:Q", format=".3f")],
        )
        .properties(width=70)
    )
    st.altair_chart(comparison_chart, use_container_width=False)

    with st.expander("Table view"):
        display_cols = comparison[
            ["lang_label", "best_tts_label", "best_tts_per", "g2p_per", "improvement", "g2p_n_items"]
        ].sort_values("lang_label")
        st.dataframe(
            display_cols.rename(
                columns={
                    "lang_label": "Language",
                    "best_tts_label": "Best TTS provider",
                    "best_tts_per": "Best TTS PER",
                    "g2p_per": "G2P PER",
                    "improvement": "Improvement",
                    "g2p_n_items": "Items",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.warning(
        "**This is not a head-to-head pronunciation contest.** The TTS PER measures a full "
        "round-trip: text → synthesized audio → Allosaurus recognizes phonemes from that "
        "audio → scored against ground truth. The G2P PER measures text → predicted "
        "phonemes directly, with no audio synthesis or audio-based recognition step at all — a "
        "strictly easier task, since it removes an entire lossy channel (synthesis quality *and* "
        "recognizer accuracy). The honest framing: a dedicated G2P model predicts the correct "
        "phoneme sequence far more reliably than the current TTS-then-recognize pipeline achieves, "
        "which quantifies how much of the TTS pronunciation gap is “the model doesn't know how "
        "to say this” vs. potentially recoverable with better grapheme-to-phoneme conditioning — "
        "just not literally “beats TTS providers at speaking.”"
    )

st.divider()
with st.expander("Methodology & limitations"):
    st.markdown(
        """
- **Scoring**: PER = (substitutions + deletions + insertions) / reference length,
  computed with edit-distance alignment over phoneme tokens, and corpus-aggregated
  (sum of edits / sum of reference length) rather than averaged per item.
- **Segmental only**: tone and stress marking is stripped from both sides before
  scoring, since the phoneme recognizer (Allosaurus) identifies phones, not pitch
  contour. A TTS system that gets every phone right but the wrong tone still scores
  perfectly here — a deliberate scope decision, not an oversight.
- **Sample sizes are modest** (125 items/category, 500 for the G2P test set) —
  enough to see a large, consistent effect, not enough for tight confidence
  intervals on the smallest sub-populations (Swahili, n=2 for two providers).
- **South Asian and Middle Eastern categories aren't literal proper nouns** — those
  scripts have no case distinction to filter general vocabulary down to names the
  way capitalization does for Vietnamese/African, so these categories test general
  non-Western vocabulary pronunciation rather than person names specifically.
- **Provider coverage gaps are real, not bugs**: Azure has no Yoruba voice; Google
  has neither Persian nor Yoruba. ElevenLabs is excluded entirely (no funded
  account) though its client remains in the codebase.
        """
    )

st.caption("The dataset, evaluation harness, and training notebooks behind this leaderboard are open-source.")
