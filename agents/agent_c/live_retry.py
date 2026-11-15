"""One try of a live request, with its time limit, and the few retries the plan allows
(decisions 19 and 30): the shared rules of common/retry.py, but no waiting between
tries. A caller is on the line: a "retry in 54s" is as good as a failure, so it fails
at once, as a daily limit does.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from common.retry import asks_too_long, worth_retrying


async def try_live(call: Callable[[], Awaitable[Any]], limit_s: float, retries: int,
                   on_retry: Callable[[BaseException], None] = lambda error: None) -> Any:
    """call() under limit_s, tried at most 1 + retries times. A timeout, a 429 or a 5xx is
    tried again; anything else (a bad key, a wrong name) and the last failure are raised.
    on_retry hears every retry, so none goes unseen (1 Nov 2026: silent retries hid
    where a judge check's 8 minutes went)."""
    for left in range(retries, -1, -1):
        try:
            return await asyncio.wait_for(call(), limit_s)
        except Exception as error:
            # wait_for raises TimeoutError, which common/retry counts as a stuck request.
            if left == 0 or not worth_retrying(error) or asks_too_long(error, 0.0):
                raise
            on_retry(error)
    raise AssertionError("unreachable")
