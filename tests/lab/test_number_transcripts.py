"""The transcript test set itself: big, labelled word by word, and consistent."""

from lab.week1.number_transcripts import ASK, NONE, NUMBER, SAME, load_transcripts

CASES = load_transcripts()


def test_the_set_is_big_and_covers_every_hard_kind():
    kinds = {case.kind for case in CASES}
    assert len(CASES) >= 750 and len({case.id for case in CASES}) == len(CASES)
    for kind in ("correction", "correction in another style", "two orders", "wait inside", "broken then whole",
                 "filler inside", "restart", "stammer", "unfinished", "amount next to it", "no number", "real", "hard"):
        assert kind in kinds, kind


def test_every_label_is_a_real_part_of_the_sentence():
    for case in CASES:
        assert " ".join(text for text, _ in case.pieces).replace(" ,", ",") == case.heard or case.kind in ("real", "hard", "calls"), case.id
        assert case.expect in (ASK, NONE) or (case.expect.isdigit() and len(case.expect) == 4), case.id
        if case.expect == NONE:
            assert not case.number_words, case.id
        elif case.expect != ASK:
            assert case.number_words, case.id  # a number to act on is always somewhere in the sentence


def test_a_repeated_mention_is_the_same_number():
    for case in CASES:
        if case.same_number:
            assert any(role == NUMBER for _, role in case.pieces) and any(role == SAME for _, role in case.pieces)
