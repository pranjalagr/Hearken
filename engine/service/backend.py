"""The backend: the front desk of one call, and the only way an agent can
change the shop.

Every tool call is one write transaction, in this order:
    move the world forward (timeline) -> check -> act -> write the log -> commit
A refused call still commits, because its log row and the timeline moves must
be kept. A crash rolls everything back, so a half-done action can never exist.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from engine.contract.action import Action, ActionContext, clean_inputs
from engine.contract.pack import Pack
from engine.lab_controls.clock import Clock
from engine.lab_controls.switches import AFTER_SAVE, FAIL, REPLY_LOST, SLOW, Switch, Switchboard
from engine.lab_controls.timeline import Timeline, done_event, prepared_event
from engine.records.action_log import ActionLog, LogEntry
from engine.safety.lifetimes import Lifetimes
from engine.safety.tickets import TICKET_NEEDED, TicketBook
from engine.safety.tokens import DONE, PreparedAction, TokenBook, request_key
from engine.service.replies import ALREADY_DONE, ERROR, EXPIRED, OK, REFUSED, ToolReply
from engine.service.tool_catalog import CONFIRM, PREPARE_PREFIX, ToolSpec, build_catalog, confirm_inputs
from engine.storage.call_settings import SAFE, CallSettings
from engine.storage.database import Database, Tx

Sleep = Callable[[float], Awaitable[None]]


class Backend:
    def __init__(
        self,
        db: Database,
        pack: Pack,
        clock: Clock,
        settings: CallSettings,
        lifetimes: Lifetimes = Lifetimes(),
        sleep: Sleep = asyncio.sleep,  # tests can pass a fake to skip real waiting
    ) -> None:
        self._db = db
        self._clock = clock
        self._settings = settings
        self._lifetimes = lifetimes
        self._sleep = sleep
        self._actions = {action.name: action for action in pack.actions()}
        _check_action_names(self._actions)
        self._tokens = TokenBook()
        self._tickets = TicketBook()
        self._timeline = Timeline(pack)
        self._switches = Switchboard()
        self._log = ActionLog()
        self.tools: list[ToolSpec] = build_catalog(list(self._actions.values()), settings)

    @property
    def _safe(self) -> bool:
        return self._settings.mode == SAFE

    async def call_tool(self, tool: str, inputs: dict[str, Any] | None = None) -> ToolReply:
        """The single entry point for every tool call, from MCP or from tests."""
        inputs = inputs or {}
        try:
            return await self._route(tool, inputs)
        except Exception as exc:
            # The transaction already rolled back, so nothing was half saved.
            # Record the bug so the black box shows it, and answer honestly.
            await self._refuse_alone(tool, inputs, f"internal error: {exc!r}", outcome=ERROR)
            return ToolReply(ERROR, "The service had an internal error.")

    async def _route(self, tool: str, inputs: dict[str, Any]) -> ToolReply:
        if self._safe and tool == CONFIRM:
            return await self._confirm(inputs)
        if self._safe and tool.startswith(PREPARE_PREFIX):
            action = self._actions.get(tool.removeprefix(PREPARE_PREFIX))
            if action is not None and action.risky:
                return await self._prepare(tool, action, inputs)
        action = self._actions.get(tool)
        # In safe mode a risky action can only run through prepare -> confirm.
        if action is not None and not (self._safe and action.risky):
            return await self._act_now(tool, action, inputs)
        return await self._refuse_alone(tool, inputs, f"Unknown tool {tool}.")

    # ---- plain mode, and safe actions in any mode -------------------------------

    async def _act_now(self, tool: str, action: Action, raw: dict[str, Any]) -> ToolReply:
        inputs, problem = clean_inputs(action.inputs, raw)
        if problem:
            return await self._refuse_alone(tool, raw, problem, action=action.name)
        switch = await self._peek_switch(action.name)
        if switch and switch.waits_before_save:
            # Waits before the transaction starts, so it holds no lock (decision 9).
            await self._sleep(switch.seconds)
        async with self._db.transaction() as tx:
            now = self._clock.now()
            await self._timeline.apply_due(tx, now)  # the world moves first
            draft = LogEntry(at=now, tool=tool, action=action.name, inputs=inputs)
            verdict = await action.check(tx, inputs)
            if not verdict.allowed:
                return await self._refuse(tx, draft, verdict.reason)
            summary = await action.summarize(tx, inputs) if action.risky else ""
            switch = await self._switches.claim(tx, switch)
            if switch and switch.kind == FAIL:
                return await self._fail(tx, draft, switch)
            data = await action.do(tx, inputs, ActionContext(now=now, token=None))
            await self._timeline.fire_event(tx, done_event(action.name), now)
            await self._log.write(tx, replace(draft, outcome=OK, acted=True, switch_note=_note(switch)))
        reply = ToolReply(OK, f"Done: {summary}." if summary else "OK.", data)
        return await self._deliver(reply, switch)

    # ---- safe mode, step 1: prepare -----------------------------------------------

    async def _prepare(self, tool: str, action: Action, raw: dict[str, Any]) -> ToolReply:
        inputs, problem = clean_inputs(action.inputs, raw)
        if problem:
            return await self._refuse_alone(tool, raw, problem, action=action.name)
        key = request_key(action, inputs)
        async with self._db.transaction() as tx:
            now = self._clock.now()
            await self._timeline.apply_due(tx, now)
            draft = LogEntry(at=now, tool=tool, action=action.name, inputs=inputs)
            # Decision 14: after a lost reply an agent may start over with
            # prepare. Asking again for something already done must not create
            # a second token, or a partial refund could happen twice.
            done = await self._tokens.done_for_request(tx, key)
            if done is not None:
                await self._log.write(tx, replace(draft, outcome=ALREADY_DONE, token=done.token))
                return _already_done(done)
            verdict = await action.check(tx, inputs)
            if not verdict.allowed:
                return await self._refuse(tx, draft, verdict.reason)
            waiting = await self._tokens.waiting_for_request(tx, key, now)
            if waiting is not None:
                await self._log.write(
                    tx, replace(draft, outcome=OK, reason="same request: same token", token=waiting.token)
                )
                return _prepared(waiting)
            summary = await action.summarize(tx, inputs)
            prepared = await self._tokens.create(
                tx, action.name, inputs, key, summary, now, now + self._lifetimes.prepared_s
            )
            await self._timeline.fire_event(tx, prepared_event(action.name), now)
            await self._log.write(tx, replace(draft, outcome=OK, token=prepared.token))
        return _prepared(prepared)

    # ---- safe mode, step 2: confirm -----------------------------------------------

    async def _confirm(self, raw: dict[str, Any]) -> ToolReply:
        if self._settings.ticket_check and not raw.get("ticket"):
            # More useful than "Missing input: ticket": it tells the agent what to do next.
            return await self._refuse_alone(CONFIRM, raw, TICKET_NEEDED)
        inputs, problem = clean_inputs(confirm_inputs(self._settings.ticket_check), raw)
        if problem:
            return await self._refuse_alone(CONFIRM, raw, problem)
        token_id, ticket_id = inputs["token"], inputs.get("ticket")

        # A quick read-only look, only to learn which action this token is for,
        # so its breakdown switch is known before the transaction starts.
        async with self._db.transaction(write=False) as tx:
            peeked = await self._tokens.get(tx, token_id)
            switch = await self._switches.peek(tx, peeked.action) if peeked else None
        if switch and switch.waits_before_save:
            await self._sleep(switch.seconds)  # outside the transaction (decision 9)

        async with self._db.transaction() as tx:
            now = self._clock.now()
            await self._timeline.apply_due(tx, now)
            prepared = await self._tokens.get(tx, token_id)
            draft = LogEntry(
                at=now,
                tool=CONFIRM,
                action=prepared.action if prepared else None,
                inputs=inputs,
                token=token_id,
                ticket=ticket_id,
            )
            if prepared is None:
                return await self._refuse(tx, draft, f"Unknown token {token_id}. Call prepare first.")
            action = self._actions[prepared.action]

            # The order of these checks matters (decision 6).
            # 1. Done first, even if expired: a late retry after a lost reply
            #    must hear "already done", or the agent would prepare again and
            #    act twice.
            if prepared.state == DONE:
                await self._log.write(tx, replace(draft, outcome=ALREADY_DONE))
                return _already_done(prepared)
            # 2. Expired: the caller's yes would be about something said too long ago.
            if prepared.expired(now):
                reason = "This prepared action expired. Prepare it again and ask the caller again."
                await self._log.write(tx, replace(draft, outcome=EXPIRED, reason=reason))
                return ToolReply(EXPIRED, reason)
            # 3. The yes-ticket, when the ticket check is on (decision 8).
            if self._settings.ticket_check:
                problem = await self._tickets.problem_with(tx, ticket_id, prepared, now)
                if problem:
                    return await self._refuse(tx, draft, problem)
            # 4. The rules again: the world may have moved since prepare.
            verdict = await action.check(tx, prepared.inputs)
            if not verdict.allowed:
                return await self._refuse(tx, draft, verdict.reason)
            switch = await self._switches.claim(tx, switch)
            if switch and switch.kind == FAIL:
                return await self._fail(tx, draft, switch)
            # 5. Act. Everything from here to the end of the block is saved
            #    together, or not at all: the change, the token, the ticket, the log.
            data = await action.do(tx, prepared.inputs, ActionContext(now=now, token=prepared.token))
            reply = ToolReply(OK, f"Done: {prepared.summary}.", data)
            await self._tokens.mark_done(tx, prepared.token, {"data": data}, now)
            if self._settings.ticket_check:
                await self._tickets.mark_used(tx, ticket_id, now)
            await self._timeline.fire_event(tx, done_event(action.name), now)
            await self._log.write(tx, replace(draft, outcome=OK, acted=True, switch_note=_note(switch)))
        return await self._deliver(reply, switch)

    # ---- shared helpers ---------------------------------------------------------

    async def _peek_switch(self, action: str) -> Switch | None:
        async with self._db.transaction(write=False) as tx:
            return await self._switches.peek(tx, action)

    async def _refuse(self, tx: Tx, draft: LogEntry, reason: str) -> ToolReply:
        """Refuse inside an open transaction. Only the log row (and any timeline
        moves) get saved; no money moves."""
        await self._log.write(tx, replace(draft, outcome=REFUSED, reason=reason))
        return ToolReply(REFUSED, reason)

    async def _refuse_alone(
        self,
        tool: str,
        inputs: dict[str, Any],
        reason: str,
        action: str | None = None,
        outcome: str = REFUSED,
    ) -> ToolReply:
        """Refuse a call that never reached a transaction of its own (bad input,
        unknown tool, or a bug), still logging it and still moving the world."""
        async with self._db.transaction() as tx:
            now = self._clock.now()
            await self._timeline.apply_due(tx, now)
            entry = LogEntry(at=now, tool=tool, action=action, inputs=inputs, outcome=outcome, reason=reason)
            await self._log.write(tx, entry)
        return ToolReply(outcome, reason)

    async def _fail(self, tx: Tx, draft: LogEntry, switch: Switch) -> ToolReply:
        reason = "The service failed. Nothing was changed. You can try again."
        await self._log.write(tx, replace(draft, outcome=ERROR, reason=reason, switch_note=switch.note()))
        return ToolReply(ERROR, reason)

    async def _deliver(self, reply: ToolReply, switch: Switch | None) -> ToolReply:
        """Send the reply after the save, unless a breakdown switch delays or loses it."""
        if switch and switch.kind == SLOW and switch.delay_at == AFTER_SAVE:
            await self._sleep(switch.seconds)
        if switch and switch.kind == REPLY_LOST:
            # The change IS saved. Only the answer goes missing, like a network
            # timeout. This is exactly what tempts a plain agent to try again.
            await self._sleep(switch.seconds)
            return ToolReply(ERROR, "Timed out waiting for the service. The result is unknown.")
        return reply


def _prepared(prepared: PreparedAction) -> ToolReply:
    return ToolReply(
        OK,
        "Prepared, nothing changed yet. Read the summary to the caller, get a clear yes, then call confirm.",
        {"token": prepared.token, "summary": prepared.summary},
    )


def _already_done(prepared: PreparedAction) -> ToolReply:
    data = (prepared.reply or {}).get("data", {})
    return ToolReply(
        ALREADY_DONE,
        f"Already done earlier: {prepared.summary}. Do not do it again.",
        {"token": prepared.token, **data},
    )


def _note(switch: Switch | None) -> str:
    return switch.note() if switch else ""


def _check_action_names(actions: dict[str, Action]) -> None:
    # These names belong to the engine's own tools; a pack action using one
    # would be unreachable or would hijack confirm.
    for name in actions:
        if name == CONFIRM or name.startswith(PREPARE_PREFIX):
            raise ValueError(f"action name {name!r} is reserved by the engine")
