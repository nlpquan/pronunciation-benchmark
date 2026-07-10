# Open Pronunciation Benchmark for Voice AI

An open benchmark measuring pronunciation accuracy of TTS providers on non-Western
and multilingual names, scored via Phoneme Error Rate (PER) against WikiPron ground
truth IPA transcriptions.

See [PROJECT.md](PROJECT.md) for the full plan (phases, dataset sources, scoring
approach, tech stack).

## Status
Repo scaffolding only — no pipeline implemented yet.

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
cp .env.example .env         # then fill in TTS API keys
```

## Structure
```
src/pronunciation_benchmark/
  data/      # dataset loaders (Wikidata, WikiPron)
  tts/       # TTS provider API clients (ElevenLabs, OpenAI, Azure, Google)
  scoring/   # G2P/forced-alignment + Phoneme Error Rate computation
  models/    # fine-tuned G2P model (Phase 2)
app/         # Streamlit leaderboard UI
data/raw/, data/processed/  # gitignored; see data loaders for how to regenerate
notebooks/   # Colab training notebooks (Phase 2)
tests/
```
