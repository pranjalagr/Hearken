"""The order-number check's marking and running, with fake AIs (no requests)."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from common.sarvam import sarvam_client
from lab.judge.judge import sarvam_judge
from lab.week1.number_check import ASKED, LIST_FORM, MISSED, PICK_RULES, RIGHT, SARVAM_MODEL, SENT, WRONG, check, mark, mark_list, report, timed
from lab.week1.number_transcripts import NUMBER, OTHER, PLAIN, SAME, TranscriptCase

CASE = TranscriptCase("t1", "correction", "Chaar paanch do ek, nahi nahi, chaar paanch ek do",
                      (("Chaar paanch do ek", OTHER), (", nahi nahi,", PLAIN), ("chaar paanch ek do", NUMBER)), "4512")


def test_the_pick_and_the_result_are_marked_and_a_failure_names_whose_fault_it_was():
    right = mark(CASE, "chaar paanch ek do", 1)
    assert (right.pick, right.outcome) == ("right words", RIGHT)
    wrong = mark(CASE, "Chaar paanch do ek", 1)
    assert (wrong.pick, wrong.outcome, wrong.accepted, wrong.fault) == ("another number", WRONG, "4521", "AI")
    assert mark(CASE, "", 1).outcome == MISSED and mark(CASE, "chaar paanch", 1).outcome == ASKED
    # The right words, misread: the reader's fault, not the AI's.
    saath = TranscriptCase("t2", "hard", "chaar paanch saath", (("chaar paanch saath", NUMBER),), "ask")
    assert mark(saath, "chaar paanch saath", 1).fault == "reader"


def test_picking_the_number_said_again_is_just_as_right():
    twice = TranscriptCase("t3", "said twice", "4521 phir se 4521", (("4521", SAME), ("phir se", PLAIN), ("4521", NUMBER)), "4521")
    assert mark(twice, "4521", 1).pick == "right words"


async def test_the_work_is_shared_over_keys_and_a_key_at_its_daily_limit_hands_its_cases_on():
    cases = [CASE] * 1 + [TranscriptCase(f"c{n}", "plain", f"order {4500 + n}", (("order", PLAIN), (str(4500 + n), NUMBER)), str(4500 + n)) for n in range(6)]
    calls = {"good": 0, "full": 0}

    async def good(rules, text):
        await asyncio.sleep(0)  # a real request lets the other keys work meanwhile
        calls["good"] += 1
        assert rules is PICK_RULES
        return {"words": text.split()[-1] if text.startswith("order") else "chaar paanch ek do"}

    async def full(rules, text):
        calls["full"] += 1
        raise httpx.HTTPStatusError("Groq said 429: Rate limit reached on tokens per day (TPD)", request=None, response=None)

    results = []
    info = await check(cases, {"good": good, "full": full}, 2, set(), results.append, log=lambda line: None)
    assert len(results) == 2 * len(cases) and all(r.outcome == RIGHT for r in results)
    assert calls["full"] == 1 and "daily limit" in info["full"]["stopped"] and info["left over"] == 0


async def test_nothing_already_saved_is_asked_again_and_a_stuck_case_is_given_up():
    asked = []

    async def flaky(rules, text):
        asked.append(text)
        raise httpx.ConnectError("no line")

    results = []
    info = await check([CASE], {"k": flaky}, 2, {("t1", 1)}, results.append, log=lambda line: None)
    assert len(asked) == 3 and not results and len(info["given up"]) == 1  # only try 2, 3 times


def test_the_report_lists_wrong_numbers_and_whose_fault_each_failure_was():
    text = report([mark(CASE, "Chaar paanch do ek", 1), mark(CASE, "chaar paanch ek do", 2)], [CASE], "now", {})
    assert "Wrong numbers accepted (1)" in text and "another number" in text and "same answer on both tries:** 0 of 1" in text


def _numbers(*pairs):
    return [{"words": words, "tag": tag} for words, tag in pairs]


def test_list_mode_one_order_number_is_taken_at_once():
    case = TranscriptCase("l1", "plain", "Order chaar paanch do ek par chaar sau pachaas rupaye",
                          (("Order", PLAIN), ("chaar paanch do ek", NUMBER), ("par", PLAIN), ("chaar sau pachaas", OTHER), ("rupaye", PLAIN)), "4521")
    result = mark_list(case, _numbers(("chaar paanch do ek", "order"), ("chaar sau pachaas", "other")), 1)
    assert (result.accepted, result.outcome, result.pick) == ("4521", RIGHT, "found the number")


def test_list_mode_two_different_order_numbers_go_to_the_thinker_and_the_same_one_twice_doesnt():
    result = mark_list(CASE, _numbers(("Chaar paanch do ek", "order"), ("chaar paanch ek do", "order")), 1)
    assert (result.accepted, result.outcome) == (None, SENT)
    twice = TranscriptCase("l2", "said twice", "4521 phir se 4521", (("4521", SAME), ("phir se", PLAIN), ("4521", NUMBER)), "4521")
    assert mark_list(twice, _numbers(("4521", "order"), ("4521", "order")), 1).outcome == RIGHT


def test_list_mode_a_taken_back_number_goes_to_the_thinker_and_others_are_never_acted_on():
    taken = TranscriptCase("l3", "taken back", "4512 nahi, galat hai", (("4512", OTHER), ("nahi, galat hai", PLAIN)), "ask")
    assert mark_list(taken, _numbers(("4512", "unsure")), 1).outcome == SENT
    amount = TranscriptCase("l4", "only other numbers", "450 rupees refund", (("450 rupees refund", OTHER),), "none")
    assert mark_list(amount, _numbers(("450", "other")), 1).outcome == RIGHT
    assert mark_list(amount, [], 1).outcome == RIGHT


def test_list_mode_an_order_number_that_cant_be_read_stops_the_other_being_taken():
    garbled = TranscriptCase("l5", "garbled correction", "7980 oh no, sorry seven thousand 97 shoonya",
                             (("7980", OTHER), ("oh no, sorry", PLAIN), ("seven thousand 97 shoonya", OTHER)), "ask")
    result = mark_list(garbled, _numbers(("7980", "order"), ("seven thousand 97 shoonya", "order")), 1)
    assert (result.accepted, result.outcome) == (None, SENT)
    # With nothing readable at all, no number can be mistaken for the only one: asked again, as before.
    nothing = mark_list(garbled, _numbers(("seven thousand 97 shoonya", "order")), 1)
    assert nothing.accepted is None and nothing.outcome != SENT


def test_list_mode_still_catches_a_wrong_number_on_the_fast_path():
    result = mark_list(CASE, _numbers(("Chaar paanch do ek", "order")), 1)  # it missed the correction
    assert (result.outcome, result.fault) == (WRONG, "AI")


async def test_sarvam_is_asked_with_thinking_off_and_the_key_only_in_the_header():
    seen = []

    def network(request):
        seen.append(request)
        reply = {"choices": [{"message": {"content": json.dumps({"numbers": []})}}], "usage": {"total_tokens": 7}}
        return httpx.Response(200, json=reply)

    used = []
    async with sarvam_client("secret-key", 5, transport=httpx.MockTransport(network)) as client:
        answer = await sarvam_judge(client, SARVAM_MODEL, on_usage=used.append, form=LIST_FORM)("rules", "order 4521")
    body = json.loads(seen[0].content)
    assert answer == {"numbers": []} and used == [{"total_tokens": 7}]
    assert seen[0].url.path == "/v1/chat/completions" and seen[0].headers["api-subscription-key"] == "secret-key"
    assert "reasoning_effort" in body and body["reasoning_effort"] is None and "include_reasoning" not in body
    assert body["model"] == SARVAM_MODEL and b"secret-key" not in seen[0].content


async def test_sarvams_rate_limit_is_waited_out_and_its_refusals_say_why():
    replies = [httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}}),
               httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})]
    waits = []

    async def no_wait(seconds):
        waits.append(seconds)

    async with sarvam_client("k", 5, transport=httpx.MockTransport(lambda request: replies.pop(0))) as client:
        assert await sarvam_judge(client, SARVAM_MODEL, sleep=no_wait, form=LIST_FORM)("r", "t") == {}
    assert len(waits) == 1

    async with sarvam_client("k", 5, transport=httpx.MockTransport(lambda request: httpx.Response(401, text="bad key"))) as client:
        with pytest.raises(httpx.HTTPStatusError, match="Sarvam said 401"):
            await sarvam_judge(client, SARVAM_MODEL, sleep=no_wait, form=LIST_FORM)("r", "t")


async def test_each_answer_is_timed_and_the_report_gives_the_speed():
    seconds = {}

    async def ask(rules, text):
        return {"numbers": []}

    assert await timed(ask, seconds)("r", "order 4521") == {"numbers": []} and "order 4521" in seconds
    result = replace(mark(CASE, "chaar paanch ek do", 1), seconds=0.25)
    assert "usual answer 0.25 s" in report([result], [CASE], "now", {})
    assert "Speed" not in report([mark(CASE, "chaar paanch ek do", 1)], [CASE], "now", {})  # older runs weren't timed
