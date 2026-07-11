# Open Pronunciation Benchmark for Voice AI — Methodology & Findings

## Motivation

There's no widely agreed-upon benchmark for how well text-to-speech (TTS) systems
pronounce non-Western names and words. Anecdotally, TTS providers do worse on
languages and scripts outside their primary training distribution, but that
claim is rarely backed by a reproducible measurement. This project builds
that measurement: a phoneme-level benchmark scoring TTS pronunciation
accuracy against real dictionary-sourced IPA ground truth, across nine
languages and six word categories, followed by a fine-tuned
grapheme-to-phoneme (G2P) model as a "how much room is there to improve"
comparison point.

## Phase 1: The Benchmark

### Dataset

Benchmark items are sourced directly from [WikiPron](https://github.com/CUNY-CL/wikipron),
which scrapes `<word, IPA>` pairs from Wiktionary — this guarantees IPA
ground truth by construction, with no manual transcription and no linkage
step to get wrong.

The original design (see `PROJECT.md`) planned to link real personal names
from Wikidata to WikiPron entries by exact string match. That was tested
empirically before being adopted: match rate was ~72% for Vietnamese (whose
given/family names are literally built from common single-syllable
dictionary words, a quirk of Vietnamese orthography) but only 13–17% for
South Asian and Middle Eastern names, and under 1% for African names —
personal names are proper nouns a general-vocabulary dictionary mostly
doesn't contain, the same reason an English word list and an English
baby-name list barely overlap. Given that, benchmark items are sampled
directly from WikiPron's vocabulary instead:

| Category | Languages | Selection signal |
|---|---|---|
| Vietnamese | vie | Capitalized first letter (proper-noun proxy) |
| South Asian | hin, ben, tam, urd | None — Devanagari/Bengali/Tamil/Perso-Arabic scripts have no case distinction, so these draw from general vocabulary, not filtered names |
| Middle Eastern | ara, fas | Same as above |
| African | yor, swa | Capitalized first letter |
| Medical/pharma | eng | Intersected against Wiktionary's `en:Medicine`/`en:Pharmacology` categories |
| Common OOV words | eng | Excluded from a public top-10k common-English-word list |

125 items were sampled per category (750 total, near the top of PROJECT.md's
200–500 target range once counted per-category rather than in aggregate).

### Systems Under Test

OpenAI TTS, Azure Speech, and Google Cloud TTS. ElevenLabs was excluded —
the account's trial credit expired, and every call returns
`402 Payment Required`; that's a billing gap, not a code issue, and the
client is still in the codebase for whenever it's funded again.

Azure/Google both require an explicit (voice, locale) pair per language,
unlike OpenAI/ElevenLabs which auto-detect language from input text. Two
real gaps came out of filling that in, both confirmed live rather than
assumed: **Azure has no Yoruba voice at all**, and **Google has neither
Persian nor Yoruba**. Those (provider, language) pairs are skipped with a
recorded reason rather than silently substituting an unrelated locale.

### Scoring

Each TTS system's audio output is run through
[Allosaurus](https://github.com/xinjli/allosaurus), an unconstrained
universal phone recognizer, to get back the phoneme sequence the audio
*actually* contains — as opposed to forced alignment, which would presuppose
the reference is correct and only locate time boundaries for it.
Unconstrained recognition is what lets a TTS mispronunciation actually show
up as an error rather than being aligned away.

Phoneme Error Rate (PER) — substitutions + deletions + insertions, divided
by reference length, the same edit-distance metric as Word Error Rate but
over phoneme tokens — is computed between that recognized sequence and
WikiPron's reference IPA. Scoring is **segmental only**: tone and stress
marking is stripped from both sides before comparison, since Allosaurus
recognizes phones, not pitch contour, and comparing a segmental hypothesis
against a reference that encodes lexical tone (Vietnamese's Chao tone
letters, Yoruba's tone accents) would inflate PER with errors no phone
recognizer could ever avoid, regardless of pronunciation accuracy.

### Results

Corpus-level PER by provider and category (lower is better; edit-weighted
across all items in a category, not a mean of per-item PER, which would
over-weight short words):

| Category | OpenAI | Azure | Google |
|---|---|---|---|
| Vietnamese | 0.732 | 0.888 | 0.758 |
| South Asian (hi/bn/ta/ur) | 0.747 | 0.875 | 0.737 |
| Middle Eastern (ar/fa) | 0.665 | 0.748 | 0.727 |
| African (yo/sw) | 0.804 | 0.700* | 0.600* |
| Medical (English) | **0.421** | **0.575** | **0.455** |
| Common OOV (English) | **0.427** | **0.568** | **0.482** |

\* Azure/Google's African rows are Swahili-only (n=2) — Yoruba is
unsupported by both, so these two numbers aren't statistically meaningful
on their own; OpenAI's African row (n=125) mixes Yoruba and Swahili since it
has no per-language config to skip anything.

The headline finding: **every provider does meaningfully worse on every
non-Western language category than on English**, by a wide margin (roughly
0.65–0.9 PER vs. 0.42–0.58 PER). That gap — TTS systems getting worse than
half the phonemes wrong on Vietnamese, South Asian, and Middle Eastern
vocabulary even from well-resourced commercial providers — is the core
result this benchmark was built to surface.

## Phase 2: Fine-Tuned G2P as a Comparison Point

### Approach

A single multilingual [ByT5-small](https://huggingface.co/google/byt5-small)
model was fine-tuned on WikiPron `word -> IPA` pairs for the nine non-English
languages above, with the input prefixed by a language tag
(`f"[{lang_id}] {word}"`) so one model handles every language rather than
training nine separate ones — this also lets very low-resource languages
(Swahili has only 370 WikiPron entries total) benefit from cross-lingual
transfer with the larger languages in the same training run. ByT5 was
chosen over T5 because it's byte-level: it needs no script-specific
tokenizer vocabulary, which matters when training data spans Latin,
Devanagari, Bengali, Tamil, and Perso-Arabic scripts in one model — a
SentencePiece vocabulary (T5) would likely fragment the non-Latin scripts
into inefficient byte-fallback tokens.

**Held-out test set, verified leak-free.** The critical requirement for a
fair comparison is that the G2P model never trains on the exact words
already scored against TTS output in Phase 1. The test split is
byte-for-byte the same 500 items from `benchmark_results.csv` (reproduced
deterministically via the same `build_benchmark_dataset(..., seed=42)`
call), with every matching `(language, word)` pair excluded from the
~113k-row train/validation pool before it's split. Verified empirically:
zero overlap between train, validation, and test.

Training ran on a Kaggle T4×2 GPU (3 epochs, ~113k training pairs). One
implementation pitfall worth recording: an initial attempt used fp16 mixed
precision, which silently failed for this model family — T5/ByT5's
internals overflow fp16's numeric range, so the gradient scaler skipped
every optimizer step to avoid NaN weights. Training *looked* like it was
running (loss values were logged, checkpoints were saved) but the model
never actually updated from its pretrained state. The fix was training in
plain fp32.

### Results

| Language | Best TTS PER | G2P PER | Improvement |
|---|---|---|---|
| Tamil | 0.833 (OpenAI) | 0.000 | 0.833 |
| Hindi | 0.743 (Google) | 0.021 | 0.722 |
| Vietnamese | 0.732 (OpenAI) | 0.081 | 0.651 |
| Persian | 0.698 (OpenAI) | 0.110 | 0.588 |
| Yoruba | 0.802 (OpenAI) | 0.218 | 0.585 |
| Bengali | 0.614 (Google) | 0.062 | 0.552 |
| Arabic | 0.649 (OpenAI) | 0.149 | 0.500 |
| Urdu | 0.712 (OpenAI) | 0.212 | 0.500 |
| Swahili | 0.600 (Google, n=2) | 0.200 (n=2) | 0.400 |

The fine-tuned model's PER is dramatically lower than the best TTS
provider's PER in every language it covers.

**This is not a head-to-head pronunciation contest, and shouldn't be read
as one.** The TTS PER measures a full round-trip: text → synthesized audio
→ Allosaurus recognizes phonemes from that audio → scored against ground
truth. The G2P PER measures text → predicted phonemes directly, with no
audio synthesis or audio-based recognition step at all — a strictly easier
task, since it removes an entire lossy channel (synthesis quality *and*
recognizer accuracy). The honest framing: **a dedicated G2P model predicts
the correct phoneme sequence far more reliably than the current
TTS-then-recognize pipeline achieves** — which is still a useful,
concrete result (it quantifies how much of the TTS pronunciation gap is
"the model doesn't know how to say this" vs. potentially recoverable with
better grapheme-to-phoneme conditioning), just not literally "beats TTS
providers at speaking."

## Limitations

- **Sample sizes are modest** (125 items/category, 500 for the G2P test
  set) — enough to see a large, consistent effect, not enough for tight
  confidence intervals on the smaller sub-populations (Swahili n=2 for two
  providers, in particular).
- **South Asian and Middle Eastern categories aren't literal proper nouns.**
  Their scripts have no case distinction, so there's no cheap signal to
  filter WikiPron's general vocabulary down to names the way capitalization
  does for Vietnamese/African — these categories test general non-Western
  vocabulary pronunciation, not specifically person names.
  ElevenLabs is excluded from every result (no funded account) though the
  client remains in the codebase.
- **Provider coverage gaps are real, not bugs**: Azure has no Yoruba voice;
  Google has neither Persian nor Yoruba. Both are documented rather than
  worked around with a substitute locale.
- **PER is segmental-only** — tone/stress is stripped before scoring, so a
  TTS system that gets every phone right but the wrong tone would still
  score perfectly here. This was a deliberate scope decision (Allosaurus
  doesn't reliably recognize tone), not an oversight, but it means the
  benchmark doesn't capture the full pronunciation picture for tonal
  languages like Vietnamese.
- **The G2P comparison isn't apples-to-apples**, as detailed above.

## Reproducing This

- `scripts/run_benchmark.py --n-per-category 125` — full TTS benchmark run.
- `scripts/build_g2p_dataset.py` — builds the G2P train/val/test split.
- `notebooks/g2p_finetune.ipynb` (Colab) or `notebooks/g2p_finetune_kaggle.ipynb`
  (Kaggle) — fine-tunes ByT5-small; both produce `g2p_predictions.csv`.
- `scripts/compare_g2p_vs_tts.py` — the before/after comparison above.
- `pytest` — 88 tests covering every module (dataset loaders, TTS clients,
  scoring, phoneme extraction, voice config, G2P dataset prep).
