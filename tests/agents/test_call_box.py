"""The call box (Part 2 decision 14), in real time with short limits."""

import asyncio
import time

import pytest

from agents.common.call_box import (
    CRASH,
    DONE,
    NO_AUDIO,
    TIME_CAP,
    UNKNOWN,
    BoxClosing,
    CallBox,
    CallBoxSettings,
)

pytestmark = pytest.mark.usefixtures("no_freezes")

FAST = CallBoxSettings(no_audio_s=0.2, max_call_s=5.0, change_wait_s=0.3)


_BEFORE: set[asyncio.Task] = set()


@pytest.fixture(autouse=True)
async def tasks_before_the_call(no_freezes):
    """Remember what was running before the test (the freeze heartbeat, which
    is why this depends on no_freezes: it must already be running), so only
    the call's own tasks are checked for leftovers."""
    _BEFORE.clear()
    _BEFORE.update(asyncio.all_tasks())
    yield


def box_tasks() -> list[asyncio.Task]:
    """The call's tasks still alive after it ended (decision 14: must be none)."""
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and t not in _BEFORE]


async def keep_hearing(box: CallBox, every_s: float = 0.05) -> None:
    while True:
        box.heard_audio()
        await asyncio.sleep(every_s)


async def test_no_caller_audio_ends_the_call():
    start = time.perf_counter()
    result = await CallBox(FAST).run(lambda box: asyncio.Event().wait())
    assert result.reason == NO_AUDIO
    assert 0.15 <= time.perf_counter() - start < 1.0
    assert box_tasks() == []  # nothing of the call is left running


async def test_audio_keeps_the_call_alive_past_the_no_audio_limit():
    async def main(box):
        box.start("ears", keep_hearing(box))
        await asyncio.sleep(0.6)  # three times the no-audio limit
        box.end("robot:bye")

    result = await CallBox(FAST).run(main)
    assert result.reason == "robot:bye"


async def test_the_time_cap_ends_even_a_lively_call():
    async def main(box):
        box.start("ears", keep_hearing(box))
        await asyncio.Event().wait()

    result = await CallBox(CallBoxSettings(no_audio_s=0.2, max_call_s=0.4, change_wait_s=0.3)).run(main)
    assert result.reason == TIME_CAP


async def test_ending_cancels_every_task_of_the_call():
    cancelled = []

    async def thinker():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append("thinker")
            raise

    async def main(box):
        box.start("thinker", thinker(), crash_safe=True)
        await asyncio.sleep(0.05)
        box.end("caller:bye")

    result = await CallBox(FAST).run(main)
    assert result.reason == "caller:bye" and cancelled == ["thinker"]
    assert box_tasks() == []


async def test_a_change_already_sent_is_finished_after_the_call_ends():
    saved = []

    async def confirm():
        await asyncio.sleep(0.2)  # the backend is still saving when the caller hangs up
        saved.append("refund")
        return "ok"

    async def main(box):
        box.start("thinker", box.change("refund", confirm()), crash_safe=True)
        await asyncio.sleep(0.02)
        box.end("caller:hung up")

    start = time.perf_counter()
    result = await CallBox(FAST).run(main)
    assert saved == ["refund"]  # it was not cut in half
    assert result.finished_changes == ["refund"] and result.unknown_changes == []
    assert time.perf_counter() - start >= 0.18  # the box waited for it


async def test_a_change_that_never_answers_is_recorded_as_unknown():
    async def confirm():
        await asyncio.sleep(30)

    async def main(box):
        box.start("thinker", box.change("refund", confirm()), crash_safe=True)
        await asyncio.sleep(0.02)
        box.end("caller:hung up")

    start = time.perf_counter()
    result = await CallBox(FAST).run(main)
    assert result.unknown_changes == ["refund"] and result.finished_changes == []
    assert time.perf_counter() - start < 1.0  # bounded by change_wait_s, not 30 s


async def test_a_change_returns_its_answer_or_unknown_to_its_caller():
    answers = []

    async def quick():
        return "ok"

    async def slow():
        await asyncio.sleep(30)

    async def main(box):
        box.start("ears", keep_hearing(box))  # the caller is still there while we wait
        answers.append(await box.change("cancel", quick()))
        answers.append(await box.change("refund", slow()))
        box.end("done here")

    result = await CallBox(FAST).run(main)
    assert answers == ["ok", UNKNOWN]
    assert result.finished_changes == ["cancel"] and result.unknown_changes == ["refund"]


async def test_a_change_whose_line_breaks_is_unknown_and_the_error_reaches_its_caller():
    errors = []

    async def broken():
        raise ConnectionError("backend went away")

    async def main(box):
        try:
            await box.change("refund", broken())
        except ConnectionError as exc:
            errors.append(str(exc))
        box.end("done here")

    result = await CallBox(FAST).run(main)
    assert errors == ["backend went away"] and result.unknown_changes == ["refund"]


async def test_nothing_new_starts_once_the_box_is_closing():
    ran = []

    async def work():
        ran.append("ran")

    async def main(box):
        box.end("caller:bye")
        assert box.start("late", work()) is None
        with pytest.raises(BoxClosing):
            await box.change("late refund", work())

    result = await CallBox(FAST).run(main)
    assert ran == [] and result.finished_changes == [] and result.unknown_changes == []


async def test_a_crash_safe_task_crashing_does_not_end_the_call():
    told = []

    async def thinker():
        raise ValueError("bad step")

    async def main(box):
        box.start("ears", keep_hearing(box))
        box.start("thinker", thinker(), crash_safe=True)
        await asyncio.sleep(0.1)
        assert not box.closing  # the call goes on
        box.end("robot:handed over")

    result = await CallBox(FAST, on_crash=lambda name, exc: told.append((name, str(exc)))).run(main)
    assert result.reason == "robot:handed over"
    assert told == [("thinker", "bad step")]
    assert result.crashes == [("thinker", "ValueError('bad step')")]


async def test_any_other_crash_ends_the_call():
    async def ears():
        raise OSError("microphone gone")

    async def main(box):
        box.start("ears", ears())
        await asyncio.Event().wait()

    result = await CallBox(FAST).run(main)
    assert result.reason == CRASH and isinstance(result.error, OSError)
    assert box_tasks() == []


async def test_a_main_that_returns_ends_the_call_and_the_first_reason_wins():
    async def main(box):
        box.end("robot:bye")
        box.end("caller:bye")

    assert (await CallBox(FAST).run(main)).reason == "robot:bye"
    assert (await CallBox(FAST).run(lambda box: asyncio.sleep(0))).reason == DONE
