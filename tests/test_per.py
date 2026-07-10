import math

from pronunciation_benchmark.scoring.per import (
    align,
    corpus_phoneme_error_rate,
    phoneme_error_rate,
    tokenize_ipa,
)


def test_tokenize_ipa_splits_on_spaces():
    assert tokenize_ipa("h ə l oʊ") == ["h", "ə", "l", "oʊ"]


def test_tokenize_ipa_empty_string():
    assert tokenize_ipa("") == []


def test_identical_sequences_have_zero_per():
    ref = ["k", "æ", "t"]
    assert phoneme_error_rate(ref, ref) == 0.0


def test_kitten_sitting_classic_edit_distance():
    # Classic Levenshtein example: kitten -> sitting = 3 edits (2 substitutions, 1 insertion).
    reference = list("kitten")
    hypothesis = list("sitting")
    result = align(reference, hypothesis)

    assert result.edits == 3
    assert result.per == 3 / 6


def test_pure_substitution():
    result = align(["a", "b", "c"], ["a", "x", "c"])
    assert result.substitutions == 1
    assert result.deletions == 0
    assert result.insertions == 0
    assert result.matches == 2
    assert result.per == 1 / 3


def test_pure_deletion_hypothesis_shorter():
    result = align(["a", "b", "c"], ["a", "c"])
    assert result.deletions == 1
    assert result.substitutions == 0
    assert result.insertions == 0
    assert result.per == 1 / 3


def test_pure_insertion_hypothesis_longer():
    result = align(["a", "c"], ["a", "b", "c"])
    assert result.insertions == 1
    assert result.substitutions == 0
    assert result.deletions == 0
    assert result.per == 1 / 2


def test_completely_disjoint_sequences():
    result = align(["a", "b"], ["x", "y", "z"])
    # 2 substitutions + 1 insertion is the minimum-edit path.
    assert result.edits == 3
    assert result.per == 3 / 2


def test_empty_reference_and_empty_hypothesis_is_zero_per():
    assert phoneme_error_rate([], []) == 0.0


def test_empty_reference_nonempty_hypothesis_is_infinite_per():
    assert math.isinf(phoneme_error_rate([], ["a", "b"]))


def test_empty_hypothesis_nonempty_reference_is_full_deletion():
    result = align(["a", "b", "c"], [])
    assert result.deletions == 3
    assert result.per == 1.0


def test_corpus_per_is_edit_weighted_not_simple_average():
    # Utterance 1: 1 edit / 10 ref phonemes = 0.1 PER
    # Utterance 2: 1 edit / 1 ref phoneme = 1.0 PER
    # Simple average would be 0.55; edit-weighted corpus PER should be 2/11.
    long_ref = ["p"] * 10
    long_hyp = ["p"] * 9 + ["x"]
    short_ref = ["p"]
    short_hyp = ["x"]

    result = corpus_phoneme_error_rate([(long_ref, long_hyp), (short_ref, short_hyp)])

    assert result == 2 / 11


def test_corpus_per_empty_input_is_zero():
    assert corpus_phoneme_error_rate([]) == 0.0
