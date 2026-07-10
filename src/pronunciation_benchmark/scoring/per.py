"""Phoneme Error Rate (PER) scoring.

PER is the ASR-style edit-distance metric (as used for Word Error Rate)
applied to phoneme sequences instead of words:

    PER = (substitutions + deletions + insertions) / len(reference)

`reference` is the ground-truth phoneme sequence (e.g. from WikiPron);
`hypothesis` is the phoneme sequence recognized/predicted from a TTS
system's audio output. Alignment is computed via standard Levenshtein
dynamic programming.
"""

from __future__ import annotations

from dataclasses import dataclass


def tokenize_ipa(ipa: str) -> list[str]:
    """Split a WikiPron-style space-segmented IPA string into phoneme tokens."""
    return ipa.split()


@dataclass
class AlignmentResult:
    substitutions: int
    deletions: int
    insertions: int
    matches: int
    reference_length: int

    @property
    def edits(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def per(self) -> float:
        if self.reference_length == 0:
            return 0.0 if self.edits == 0 else float("inf")
        return self.edits / self.reference_length


def align(reference: list[str], hypothesis: list[str]) -> AlignmentResult:
    """Compute the minimum-edit-distance alignment between two phoneme sequences.

    When multiple alignments share the same minimum edit distance, the
    specific substitution/deletion/insertion split is not unique - but the
    total edit count (and therefore PER) is, since it's read off the DP
    table's final cell regardless of which optimal path is backtracked.
    """
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # deletion: reference[i-1] missing from hypothesis
                    dp[i][j - 1],  # insertion: hypothesis[j-1] extra
                    dp[i - 1][j - 1],  # substitution
                )

    i, j = n, m
    substitutions = deletions = insertions = matches = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            matches += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return AlignmentResult(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        matches=matches,
        reference_length=n,
    )


def phoneme_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    """Single-utterance PER: (S + D + I) / len(reference)."""
    return align(reference, hypothesis).per


def corpus_phoneme_error_rate(pairs: list[tuple[list[str], list[str]]]) -> float:
    """Aggregate PER across many (reference, hypothesis) pairs.

    Computed as sum(edits) / sum(len(reference)) across all pairs - the
    standard way to report a corpus-level PER, as opposed to averaging each
    utterance's PER (which over-weights short utterances/names).
    """
    total_edits = 0
    total_ref_len = 0
    for reference, hypothesis in pairs:
        result = align(reference, hypothesis)
        total_edits += result.edits
        total_ref_len += result.reference_length
    if total_ref_len == 0:
        return 0.0
    return total_edits / total_ref_len
