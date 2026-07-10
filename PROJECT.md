# Open Pronunciation Benchmark for Voice AI

## The Pitch
An open benchmark measuring pronunciation accuracy of TTS (text-to-speech) providers
on non-Western and multilingual names, with a public leaderboard — addressing a
real, unsolved measurement gap in voice AI (no agreed-upon benchmark currently exists
for this problem).

## Goal
Build a portfolio project that:
1. Demonstrates real speech/audio ML skills (complementing existing RAG + multimodal
   fine-tuning projects)
2. Produces a public, citable artifact (not just a closed demo)
3. Shows rigorous evaluation methodology (phoneme error rate against ground truth,
   not self-graded metrics)

## Background / Context
- Builder has 2 existing AI projects: a RAG system (FinSight AI) and a multimodal
  text/voice tone detector (Depression Companion) — both using Python, PyTorch,
  FastAPI, Streamlit, Docker, HuggingFace ecosystem.
- This project should reuse the same stack where possible to keep ramp-up low —
  the new ground here is the *domain* (phonetics/IPA/G2P), not new frameworks.
- Zero budget target, same as prior projects — free tiers only (Colab for training,
  free/trial tiers for TTS APIs, Hugging Face Spaces or similar for hosting).

---

## Phase 1 — Core Benchmark (target: 2-3 weeks)

**Dataset**
- Build a test set of 200-500 names/words across categories:
  - Vietnamese names
  - Other non-Western names (South Asian, African, Middle Eastern)
  - Medical/pharma terms
  - Common out-of-vocabulary (OOV) words
- Source from public datasets rather than starting from zero:
  - Wikidata (names)
  - **WikiPron** (pre-built IPA ground truth transcriptions for many languages —
    use this to avoid manually transcribing IPA from scratch)

**Systems under test**
- Run each name/word through 3-4 TTS APIs. Candidates (check current free/trial
  tier availability before committing):
  - ElevenLabs
  - OpenAI TTS
  - Azure TTS
  - Google Cloud TTS

**Scoring**
- Use a G2P model or forced-alignment ASR to extract the *actual* phonemes each
  TTS system produced from its audio output.
- Compute Phoneme Error Rate (PER) against the WikiPron ground-truth IPA.
- This scoring pipeline is the core technical contribution — invest the most
  care here.

---

## Phase 2 — Model Layer (target: 2-3 weeks)

- Fine-tune a small G2P model on WikiPron data, focused specifically on
  Vietnamese and other underrepresented names/languages.
  - Candidate architectures: T5-small or ByT5 (standard for G2P tasks)
  - Train on Google Colab free tier (GPU compute, doesn't touch Claude Code usage)
- Compare fine-tuned model's phoneme predictions against raw TTS output —
  this "before/after" comparison is the key result to highlight.

---

## Phase 3 — Ship It Publicly (target: 1 week)

- Public leaderboard: simple Streamlit or Next.js page ranking TTS providers
  by PER, broken down by language / name-origin category.
- Open-source the benchmark dataset + evaluation harness on GitHub.
- Short methodology write-up: approach, findings, limitations. This becomes
  the "one piece of work you're proud of" link for job applications.
- Deploy (Hugging Face Spaces + Docker, matching FinSight AI's deployment
  pattern) so there's a live, clickable demo — not just a repo.

---

## Tech Stack (reusing existing skills where possible)
- **Language/ML**: Python, PyTorch, HuggingFace Transformers
- **Backend**: FastAPI (if an API layer is needed)
- **Frontend**: Streamlit (leaderboard UI)
- **Deployment**: Docker, Hugging Face Spaces
- **New domain knowledge needed**: phonetics, IPA notation, G2P modeling,
  WikiPron dataset format

## Definition of Done (for resume purposes)
By the end, should be able to write concrete (non-placeholder) resume bullets like:
- "Built an open benchmark evaluating TTS pronunciation accuracy on [N] non-Western
  languages across [N] providers, using phoneme error rate against IPA ground truth."
- "Fine-tuned a G2P model, improving pronunciation accuracy on underrepresented
  names by [X]% over baseline TTS output."
- "Open-sourced the evaluation harness and public leaderboard, [live link]."

## Working Notes / Session Planning
- Scope each Claude Code session to ONE sub-task (e.g., "set up repo structure,"
  "write the WikiPron data loader," "build the PER scoring function") rather than
  trying to build multiple phases in one sitting.
- Use `/clear` between unrelated tasks to keep context lean.
- Default to Sonnet for routine coding; reserve Opus (if available) for
  architecture/design decisions only.
