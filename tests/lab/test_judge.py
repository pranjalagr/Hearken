"""The judge's machinery (Part 5, L10, L11), without the AI: its packet, the
checks on its answer form, and how it is graded on planted mistakes. Fake
judges stand in: one that marks right, one that misses a lie, one that cries wolf."""

import json
import re

import httpx
import pytest

from common.groq import groq_client
from lab.health.check import GREEN, RED, judge_light
from lab.judge.check import agreement, changed_share, load_planted, run_planted
from lab.judge.judge import (
    FORM,
    GROQ_MODEL,
    VERDICTS,
    BadAnswer,
    Marks,
    Packet,
    Turn,
    check_answer,
    groq_judge,
    judge,
    paced,
    strict_form,
)

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


# ---- the Groq judge, on a fake network under the real request code ------------------


def fake_groq(*replies):
    """A Groq that answers each request with the next (status, body, headers)."""
    sent, queue = [], list(replies)

    def handle(request):
        sent.append(request)
        status, body, headers = queue.pop(0)
        return httpx.Response(status, json=body, headers=headers)

    return groq_client("not-a-real-key", 120, httpx.MockTransport(handle)), sent


def chat_reply(content):
    return {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 1234}}


ANSWER = {"replies": [{"turn": 3, "language": "same", "claims": []}]}


async def test_the_groq_judge_asks_in_strict_form_and_reads_the_answer():
    client, sent = fake_groq((200, chat_reply(json.dumps(ANSWER)), {}))
    usage = []
    ask = groq_judge(client, on_usage=usage.append)
    assert await ask("the rules", "the call") == ANSWER and usage == [{"total_tokens": 1234}]
    body = json.loads(sent[0].content)
    assert str(sent[0].url) == "https://api.groq.com/openai/v1/chat/completions"
    assert sent[0].headers["authorization"] == "Bearer not-a-real-key"  # the key rides in the client only
    assert (body["model"], body["reasoning_effort"], body["include_reasoning"]) == (GROQ_MODEL, "high", False)
    assert body["messages"] == [{"role": "system", "content": "the rules"}, {"role": "user", "content": "the call"}]
    assert body["response_format"]["json_schema"]["strict"] is True


def test_the_strict_form_closes_every_object_and_leaves_the_gemini_form_alone():
    strict = strict_form(FORM)
    claim = strict["properties"]["replies"]["items"]["properties"]["claims"]["items"]
    assert strict["additionalProperties"] is False and claim["additionalProperties"] is False
    assert claim["required"] == ["quote", "verdict", "why"] and claim["properties"]["verdict"]["enum"] == list(VERDICTS)
    assert "additionalProperties" not in FORM  # Gemini's form is unchanged


async def test_a_busy_groq_is_asked_again_after_the_wait_it_asks_for():
    waited = []

    async def sleep(seconds):
        waited.append(seconds)

    client, sent = fake_groq((429, {"error": {"message": "slow down"}}, {"retry-after": "37.5"}),
                             (200, chat_reply(json.dumps(ANSWER)), {}))
    assert await groq_judge(client, sleep=sleep)("r", "t") == ANSWER
    assert waited == [37.5] and len(sent) == 2  # Groq's 37.5 s, not our first 10 s, which would only fail again


async def test_a_groq_daily_limit_fails_at_once():
    async def sleep(seconds):
        raise AssertionError("waited")

    client, sent = fake_groq((429, {"error": {"message": "tokens per day"}}, {"retry-after": "51000"}))
    with pytest.raises(httpx.HTTPStatusError, match="Groq said 429"):
        await groq_judge(client, sleep=sleep)("r", "t")
    assert len(sent) == 1


async def test_a_refused_request_fails_at_once_with_groqs_reason():
    client, sent = fake_groq((400, {"error": {"message": "json_validate_failed"}}, {}))
    with pytest.raises(httpx.HTTPStatusError, match="Groq said 400: .*json_validate_failed"):
        await groq_judge(client)("r", "t")
    assert len(sent) == 1


async def test_an_answer_that_isnt_json_is_refused():
    client, _ = fake_groq((200, chat_reply("I think turn 3 is fine."), {}))
    with pytest.raises(BadAnswer, match="not a JSON answer"):
        await groq_judge(client)("r", "t")


async def test_a_broken_answer_from_groq_is_asked_for_again_then_refused():
    broken = (400, {"error": {"code": "json_validate_failed", "failed_generation": ""}}, {})
    told = []
    client, sent = fake_groq(broken, (200, chat_reply(json.dumps(ANSWER)), {}))
    assert await groq_judge(client, on_retry=lambda e, w: told.append(str(e)))("r", "t") == ANSWER
    assert len(sent) == 2 and told == ["its answer broke the form"]
    client, sent = fake_groq(broken, broken, broken)
    with pytest.raises(BadAnswer, match="broke the form 3 times"):
        await groq_judge(client)("r", "t")
    assert len(sent) == 3


async def test_judge_requests_are_spaced_by_the_gap():
    now, waited, asked = [100.0], [], []

    async def sleep(seconds):
        waited.append(seconds)
        now[0] += seconds

    async def ask(rules, text):
        asked.append(now[0])
        now[0] += 3  # the answer takes 3 s
        return {}

    spaced = paced(ask, 20, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        await spaced("r", "t")
    assert asked == [100, 120, 140] and waited == [17, 17]  # 20 s from start to start, never more


async def test_an_unusable_answer_counts_against_the_judge_and_the_check_goes_on():
    async def refuses_one(rules, text):
        case = next(c for c in CASES if c.packet.text() == text)
        if case.id == "lie_stopped_when_it_was_done":
            raise BadAnswer("its answer broke the form 3 times")
        return honest_answer(case)

    graded, marks = await run_planted(refuses_one)
    assert len(marks) == len(CASES)  # every other case was still judged
    assert graded.refused == ["lie_stopped_when_it_was_done: its answer broke the form 3 times"]
    assert (graded.caught, graded.planted) == (5, 6) and "lie_stopped_when_it_was_done turn 3: no usable answer" in graded.missed
    assert judge_light(graded.caught, graded.planted, 0, len(graded.refused)).status == RED
