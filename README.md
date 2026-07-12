---
title: Pronunciation Benchmark
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Open Pronunciation Benchmark for Voice AI

An open benchmark measuring pronunciation accuracy of TTS providers on non-Western
and multilingual names, scored via Phoneme Error Rate (PER) against WikiPron ground
truth IPA transcriptions. A fine-tuned G2P model is included as a comparison point.

See [WRITEUP.md](WRITEUP.md) for methodology and findings.

## Status
All three phases complete: benchmark harness, fine-tuned G2P model, and this
public leaderboard.

## Leaderboard
```bash
pip install -r requirements.txt
pip install -e .
streamlit run app/streamlit_app.py
```
Or via Docker (same image used for the Hugging Face Spaces deployment):
```bash
docker build -t pronunciation-benchmark .
docker run -p 7860:7860 pronunciation-benchmark
```

## Setup (full harness - dataset build, TTS calls, G2P training)
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
  data/        # dataset loaders (Wikidata, WikiPron)
  tts/         # TTS provider API clients (ElevenLabs, OpenAI, Azure, Google)
  scoring/     # G2P/forced-alignment + Phoneme Error Rate computation
  leaderboard/ # shared PER aggregation, used by scripts/ and app/
  models/      # fine-tuned G2P model (Phase 2)
app/           # Streamlit leaderboard UI (app/requirements.txt is Docker-only, app-scoped)
data/raw/, data/processed/  # gitignored; see data loaders for how to regenerate
data/results/  # published results (benchmark_results.csv, g2p_predictions.csv) - tracked in git
notebooks/     # Colab/Kaggle training notebooks (Phase 2)
tests/
```
