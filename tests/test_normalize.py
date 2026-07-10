from pronunciation_benchmark.scoring.normalize import strip_suprasegmentals


def test_drops_standalone_vietnamese_tone_token():
    # "n" + tone token, as in WikiPron's entry for "Nguyễn".
    assert strip_suprasegmentals(["ŋ", "w", "i", "ə", "n", "˦ˀ˥"]) == ["ŋ", "w", "i", "ə", "n"]


def test_drops_plain_tone_register_token():
    assert strip_suprasegmentals(["a", "˧˧"]) == ["a"]


def test_keeps_true_glottal_stop_consonant():
    # "ʔ" (U+0294) is a real consonant, distinct from the modifier "ˀ"
    # (U+02C0) that marks glottalized/creaky tone - must not be dropped.
    assert strip_suprasegmentals(["ʔ", "a"]) == ["ʔ", "a"]


def test_strips_tone_accent_but_keeps_nasalization_yoruba():
    assert strip_suprasegmentals(["ɔ̃̀", "ã̀", "ĩ́"]) == ["ɔ̃", "ã", "ĩ"]


def test_strips_stress_accent_but_keeps_length_and_pharyngealization_arabic():
    assert strip_suprasegmentals(["d̪ˤ", "ɒ́ː"]) == ["d̪ˤ", "ɒː"]


def test_leaves_affricates_and_unreleased_diacritics_untouched():
    tokens = ["t͡ɕ", "k̟̚", "k̚"]
    assert strip_suprasegmentals(tokens) == tokens


def test_leaves_hindi_phonemic_diacritics_untouched():
    tokens = ["d͡ʒ", "t̪ʰ", "ɑ̃ː"]
    assert strip_suprasegmentals(tokens) == tokens


def test_empty_input_returns_empty_list():
    assert strip_suprasegmentals([]) == []
