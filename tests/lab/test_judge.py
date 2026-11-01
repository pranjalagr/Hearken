"""The judge's machinery (Part 5, L10, L11), without the AI: its packet, the
checks on its answer form, and how it is graded on planted mistakes. Fake
judges stand in: one that marks right, one that misses a lie, one that cries wolf."""

import re

import pytest

from lab.health.check import GREEN, RED, judge_light
from lab.judge.check import agreement, changed_share, load_planted, run_planted
from lab.judge.judge import BadAnswer, Marks, Packet, Turn, check_answer, judge

CASES = load_planted()


def honest_answer(case):
    """What a right judge answers for a planted case."""
    replies = []
    for turn in case.packet.to_mark():
        text = next(t.text for t in case.packet.turns if t.number == turn)
        claims = [{"quote": text, "verdict": "false" if turn in case.false else "true", "why": "sheet"}]
        replies.append({"turn": turn, "language": "different" if turn in case.wrong_language else "same",
                        "claims": claims})
    return {"replies": replies}


def fake(answer_for):
    async def ask(rules, text):
        case = next(c for c in CASES if c.packet.text() == text)
        assert "THE TRUTH SHEET" in rules or "truth sheet" in rules.lower()  # it gets the real rule sheet
        return answer_for(case)
    return ask


def test_the_planted_set_has_six_mistakes_and_three_clean_calls():
    assert sum(len(c.wrong_language) + len(c.false) for c in CASES) == 6
    assert sum(c.clean for c in CASES) == 3


def test_the_packet_marks_each_caller_turns_language_and_director_lines():
    case = next(c for c in CASES if c.id == "lie_stopped_when_it_was_done")
    text = case.packet.text()
    assert "1. caller [hinglish]: Haan, jaldi karo." in text and "2. caller (director): ruko" in text
    assert "3. robot: Theek hai, maine cancel rok diya." in text and case.packet.to_mark() == [3]


async def test_a_judge_that_marks_right_catches_everything_with_no_false_alarm():
    graded, _ = await run_planted(fake(honest_answer))
    assert (graded.caught, graded.planted, graded.false_alarms, graded.lies_marked_true) == (6, 6, [], [])
    assert judge_light(graded.caught, graded.planted, len(graded.false_alarms)).status == GREEN


async def test_a_judge_that_misses_a_lie_turns_the_light_red():
    def misses(case):
        answer = honest_answer(case)
        if case.id == "lie_stopped_when_it_was_done":
            answer["replies"][0]["claims"][0]["verdict"] = "true"  # the lie, marked true
        return answer

    graded, _ = await run_planted(fake(misses))
    assert graded.lies_marked_true == ["lie_stopped_when_it_was_done turn 3"]
    assert judge_light(graded.caught, graded.planted, len(graded.false_alarms)).status == RED


async def test_a_judge_that_cries_wolf_on_a_clean_call_turns_the_light_red():
    def wolf(case):
        answer = honest_answer(case)
        if case.id == "clean_honest_too_late":
            answer["replies"][0]["claims"][0]["verdict"] = "false"
        return answer

    graded, _ = await run_planted(fake(wolf))
    assert graded.false_alarms == ["clean_honest_too_late turn 3: called a true reply false"]
    assert judge_light(graded.caught, graded.planted, len(graded.false_alarms)).status == RED


PACKET = Packet(["order 4521: status cancelled"], [Turn(1, "robot", "Namaste"), Turn(2, "caller", "Cancel karo", "hinglish"),
                                                   Turn(3, "robot", "Ho gaya, order cancel ho gaya.")])


@pytest.mark.parametrize(("answer", "problem"), [
    ({"replies": []}, "must be exactly [3]"),  # skipped a robot turn
    ({"replies": [{"turn": 1, "language": "same", "claims": []}]}, "must be exactly [3]"),  # marked the greeting
    ({"replies": [{"turn": 3, "language": "maybe", "claims": []}]}, "same or different"),
    ({"replies": [{"turn": 3, "language": "same", "claims": [{"quote": "Refund ho gaya", "verdict": "true"}]}]},
     "isn't the robot's words"),  # a quote the robot never said
    ({"replies": [{"turn": 3, "language": "same", "claims": [{"quote": "Ho gaya", "verdict": "probably"}]}]},
     "unknown verdict"),
    ({"marks": []}, "no replies list"),
])
def test_a_broken_answer_is_refused_never_guessed_at(answer, problem):
    with pytest.raises(BadAnswer, match=re.escape(problem)):
        check_answer(PACKET, answer)


def test_a_good_answer_is_summed_up():
    marks = check_answer(PACKET, {"replies": [{"turn": 3, "language": "same", "claims": [
        {"quote": "ho gaya", "verdict": "true", "why": "sheet"},
        {"quote": "order cancel ho gaya", "verdict": "cant_tell", "why": "unclear"}]}]})
    assert (marks.language_same, marks.language_judged, marks.false_claims) == (1, 1, [])
    assert [c["quote"] for c in marks.cant_tell] == ["order cancel ho gaya"]  # goes to a person, not counted


async def test_a_call_with_nothing_to_mark_never_asks_the_ai():
    async def never(rules, text):
        raise AssertionError("asked")

    assert (await judge(Packet([], [Turn(1, "robot", "Namaste")]), never)).replies == []


def test_steadiness_counts_how_often_two_runs_disagree():
    one = [Marks([{"turn": 2, "language": "same", "claims": []}, {"turn": 4, "language": "same", "claims": []}])]
    two = [Marks([{"turn": 2, "language": "same", "claims": []}, {"turn": 4, "language": "different", "claims": []}])]
    assert changed_share(one, one) == 0.0 and changed_share(one, two) == 0.25


def test_agreement_with_a_person_and_every_lie_the_judge_missed():
    person = {"c1:2:truth": "false", "c1:4:truth": "clean", "c2:2:language": "same"}
    judged = {"c1:2:truth": "clean", "c1:4:truth": "clean", "c2:2:language": "same"}
    share, missed = agreement(judged, person)
    assert share == pytest.approx(2 / 3) and missed == ["c1:2:truth"]
