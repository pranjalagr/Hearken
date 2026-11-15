"""The freeze check itself (decision 14): it must catch a real block, name the
code that blocked, and stay quiet for normal async waiting."""

import asyncio
import time


async def test_a_blocking_call_is_caught_and_named(make_freeze_watch):
    watch = make_freeze_watch()
    watch.start()
    await asyncio.sleep(0.05)  # let the heartbeat start
    time.sleep(0.3)  # the classic mistake: a blocking sleep on the event loop
    await asyncio.sleep(0.05)
    freezes = await watch.stop()
    assert len(freezes) == 1
    assert "test_a_blocking_call_is_caught_and_named" in freezes[0]  # it points at the guilty code


async def test_awaiting_is_never_a_freeze(make_freeze_watch):
    watch = make_freeze_watch()
    watch.start()
    await asyncio.sleep(0.4)  # long, but the loop stays free the whole time
    assert await watch.stop() == []


async def test_work_in_a_thread_is_never_a_freeze(make_freeze_watch):
    watch = make_freeze_watch()
    watch.start()
    await asyncio.to_thread(time.sleep, 0.3)  # the right way to do slow blocking work
    assert await watch.stop() == []


async def test_each_stall_is_kept_with_when_how_long_and_whose(make_freeze_watch):
    # The lab writes these into a run's folder, so "the lab fell behind" comes with a cause.
    watch = make_freeze_watch()
    watch.start()
    await asyncio.sleep(0.05)
    time.sleep(0.3)
    await asyncio.sleep(0.05)
    await watch.stop()
    [stall] = watch.stalls
    assert stall.ours and stall.where and 250 <= stall.late_ms <= 600 and stall.wall.count(":") == 2
