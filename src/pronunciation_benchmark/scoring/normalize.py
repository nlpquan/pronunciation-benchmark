"""Suprasegmental (tone/stress) normalization for PER scoring.

An audio-derived phoneme sequence (scoring.asr.phoneme_recognizer output) is
a *segmental* phone recognition - it identifies which phones were spoken,
not the pitch/stress contour they carried. WikiPron reference IPA, however,
encodes tone/stress in two different ways depending on the language:

- Vietnamese marks tone as a standalone Chao tone-letter token appended
  after each syllable (e.g. "n ˦ˀ˥"; the ˀ is a co-occurring glottalization
  component of that syllable's tone, e.g. Hanoi's creaky tone).
- Yoruba marks tone as grave/acute/macron accents fused onto the vowel
  itself (e.g. "ɔ̃̀" = nasalized ɔ carrying a grave/low tone) - mixed in
  with real phonemic diacritics like nasalization, which must be kept.
- Arabic/Hindi carry no lexical tone, but broad transcriptions occasionally
  mark stress with the same accent characters (e.g. "ɒ́ː").

Comparing a segmental hypothesis against a reference that includes this
suprasegmental marking would inflate PER with edits no phone recognizer
could ever avoid, regardless of pronunciation accuracy. This module strips
tone/stress marking from both sides before scoring, while leaving phonemic
diacritics (nasalization, dentality, aspiration, affricate tie-bars, vowel
length) untouched - those are real segmental contrasts an accurate
recognizer should be expected to get right.

Verified against WikiPron's actual token inventory (see project memory
"phoneme extraction: Allosaurus") rather than assumed: Vietnamese, Yoruba,
Hindi, and Arabic data were inspected directly for which diacritics
co-occur with which phenomena before choosing this character set.
"""

from __future__ import annotations

# Chao tone-letter register marks (Vietnamese-style standalone tone tokens),
# plus the modifier-letter glottal stop that co-occurs with them to mark
# creaky/glottalized tone (e.g. "˦ˀ˥") - distinct from the true glottal
# stop consonant "ʔ", which must never be stripped.
_TONE_TOKEN_CHARS = set("˥˦˧˨˩ˀ")

# Combining accents used for tone (Yoruba) or stress (Arabic broad
# transcriptions) when fused onto a segmental character. Deliberately
# narrow: does NOT include nasalization (U+0303 tilde is excluded because
# it's already covered by combining tilde below only when acting as an
# accent - see note), dentality, or tie-bars, which are phonemic.
_TONE_ACCENT_MARKS = "̀́̄"  # grave, acute, macron


def strip_suprasegmentals(tokens: list[str]) -> list[str]:
    """Remove tone/stress marking from a phoneme token sequence.

    Drops tokens that are pure tone markers (e.g. "˧˧", "˦ˀ˥"), and strips
    tone/stress accent characters from within otherwise-segmental tokens
    (e.g. "ɔ̃̀" -> "ɔ̃", keeping the phonemic nasalization). Tokens that
    become empty after stripping are dropped.
    """
    result = []
    for tok in tokens:
        if tok and all(ch in _TONE_TOKEN_CHARS for ch in tok):
            continue
        stripped = "".join(ch for ch in tok if ch not in _TONE_ACCENT_MARKS)
        if stripped:
            result.append(stripped)
    return result
