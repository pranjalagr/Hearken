"""Decision 2's Flash-Lite check, without the network: the situations, what
the Talker reads, retries, and the report."""

from datetime import datetime

import pytest
from google.genai import errors

from lab.week1.flash_lite_check import (
    MODEL,
    STYLE_PROMPT,
    Answer,
    ask_all,
    load_situations,
    packet,
    report,
    with_retries,
)


def test_there_are_20_situations_each_ending_with_the_caller():
    situations = load_situations()
    assert [s.id for s in situations] == list(range(1, 21))
    assert all(s.turns[-1][0] == "caller" and s.good for s in situations)


def test_no_situation_copies_a_style_example():
    """Otherwise the check would measure copying, not the model's Hinglish."""
    style = STYLE_PROMPT.read_text(encoding="utf-8")
    assert not [s.id for s in load_situations() if s.caller_line in style]


def test_the_talker_reads_the_call_then_the_notes():
    situation = load_situations()[4]  # id 5: a status note
    assert packet(situation) == (
        "The call so far:\n"
        "You: Ek second, check kar raha hoon.\n"
        "Caller: Toh kya hua mera order?\n"
        "\n"
        "NOTES:\n"
        "- found: order 4530\n"
        "- status: Aapka order mil gaya hai, restaurant ne abhi banana shuru nahi kiya.\n"
        "\n"
        "Your next reply:"
    )
    assert "NOTES: none yet" in packet(load_situations()[0])


def test_a_bad_situations_file_is_refused(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- id: 1\n  good: x\n  turns:\n    - robot: hi\n  notes: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="last turn must be the caller"):
        load_situations(path)


async def test_replies_are_asked_one_by_one_and_a_failure_is_recorded_not_fatal():
    situations = load_situations()[:3]
    pauses = []

    async def ask(text):
        if "4530" in text or "kya madad" not in text:
            return Answer(f"  reply to: {text.splitlines()[-3]}  ", 0)
        raise RuntimeError("network down")

    async def sleep(seconds):
        pauses.append(seconds)

    replies = await ask_all(situations, ask, pause_s=1.0, sleep=sleep)
    assert [r.situation.id for r in replies] == [1, 2, 3]
    assert replies[0].text.startswith("reply to:") and not replies[0].text.endswith(" ")
    assert replies[1].error == "RuntimeError: network down" and replies[1].text == ""
    assert pauses == [1.0, 1.0]  # between calls, not before the first


def api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": "x", "status": "S"}})


async def test_too_many_requests_and_server_errors_are_retried():
    waited, calls = [], []

    async def sleep(seconds):
        waited.append(seconds)

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise api_error(429 if len(calls) == 1 else 503)
        return "ok"

    told = []
    retried = lambda error, wait: told.append((error.code, wait))
    assert await with_retries(flaky, waits=(10, 20, 40), sleep=sleep, on_retry=retried) == "ok"
    assert waited == [10, 20] and told == [(429, 10), (503, 20)]  # every retry is told, never silent


async def test_a_bad_key_fails_at_once_and_retries_have_an_end():
    async def sleep(seconds):
        pass

    async def bad_key():
        raise api_error(403)

    with pytest.raises(errors.APIError):
        await with_retries(bad_key, sleep=sleep)

    calls = []

    async def always_busy():
        calls.append(1)
        raise api_error(429)

    with pytest.raises(errors.APIError):
        await with_retries(always_busy, waits=(1, 2), sleep=sleep)
    assert len(calls) == 3  # the first try and two retries


async def test_the_report_has_the_model_date_rule_and_one_row_per_situation():
    situations = load_situations()

    async def ask(text):
        return Answer("Ji | zaroor", 0)

    async def no_sleep(seconds):
        pass

    replies = await ask_all(situations, ask, sleep=no_sleep)
    text = report(replies, MODEL, datetime(2026, 11, 1, 10, 30))
    assert f"`{MODEL}`, run on 01 Nov 2026 at 10:30" in text
    assert "18 or more of 20 natural" in text and "gemini-3.8-flash" in text
    rows = [line for line in text.splitlines() if line.startswith("| ") and line[2].isdigit()]
    assert len(rows) == 20
    assert "Ji \\| zaroor" in rows[0]  # a | in a reply can't break the table
    assert "Errors:** 0" in text


def api_error_saying(code, message):
    return errors.APIError(code, {"error": {"code": code, "message": message, "status": "S"}})


async def test_a_daily_limit_fails_at_once_and_a_short_asked_wait_is_honoured():
    """On 1 Nov 2026 the judge check spent 60 s retrying a daily cap with 14 hours left."""
    waited = []

    async def sleep(seconds):
        waited.append(seconds)

    async def daily_cap():
        raise api_error_saying(429, "Quota exceeded. Please retry in 14h34m57.7s.")

    with pytest.raises(errors.APIError):
        await with_retries(daily_cap, waits=(10, 20), sleep=sleep)
    assert waited == []

    calls = []

    async def per_minute():
        calls.append(1)
        if len(calls) == 1:
            raise api_error_saying(429, "Please retry in 54s.")
        return "ok"

    assert await with_retries(per_minute, waits=(10, 20), sleep=sleep) == "ok"
    assert waited == [54]  # our 10 s would only have failed again


async def test_a_stuck_request_is_tried_again():
    waited, calls = [], []

    async def sleep(seconds):
        waited.append(seconds)

    async def stuck_once():
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError  # what the async Google client raises at its time limit
        return "ok"

    assert await with_retries(stuck_once, waits=(10,), sleep=sleep) == "ok" and waited == [10]
