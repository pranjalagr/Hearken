"""The Guard (decisions 9, 17, 20-23): the only way a prepared change gets confirmed.

Plain code holds the backend's GuardDoor; no AI can reach it. When the Thinker
prepares a change, the Guard:
1. reads the yes-question the backend wrote (GuardDoor.question) and plays it itself:
   no AI writes or shortens it (decision 9);
2. sends a mark after it, and waits for the caller's "played" receipt: a "haan" that
   began before the question finished playing never counts (Agent A took the
   "haan haan" said over its own question as a yes, food_08, 15 Nov 2026);
3. labels the answer: a word list on the partial words ("ruko", "nahi"...) and a
   one-word judge; a no or a stop needs only one of them, a yes needs the judge's
   "yes", a finished sentence and the receipt timing;
4. on a real yes, gets a yes-ticket and presses confirm at once, shielded by the call
   box: a caller cutting in never cancels a change already sent (decision 17), and
   the outcome comes only from the backend's answer, as a [result] note;
5. asks again on "unclear": once in full, once short, then the keypad, then a human;
6. follows the retry table (decision 19): a confirm with no answer is sent again with
   the same token (the backend answers "already done" if the first one saved); an
   expired token is prepared again and asked again; still unsure -> the truth and
   one hand-over, never "done".

Every yes-moment is written to the diary (question, words, label, timing).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from agents.agent_c.call_job import CallJob
from agents.agent_c.guard.judge import CHANGED, NO, OTHER, STOP, UNCLEAR, YES, YesJudge, word_list_label
from agents.agent_c.guard.yes_check import YesCheck
from agents.agent_c.notebook import RESULT, YES_ANSWER, Note, Notebook
from agents.agent_c.talker.language import StickyLanguage
from agents.common.call_box import UNKNOWN, BoxClosing
from agents.common.diary import Diary
from agents.common.tools import Tools
from engine.contract.spoken import SpokenWords
from engine.safety.guard_door import TicketRefused
from engine.service.replies import ALREADY_DONE, EXPIRED, OK, REFUSED, ToolReply
from engine.service.tool_catalog import CONFIRM, PREPARE_PREFIX

DONE, REFUSED_OUTCOME, STOPPED, UNKNOWN_OUTCOME = "done", "refused", "stopped", "unknown"
FULL, AGAIN, KEYPAD = 1, 2, 3  # the asking ladder; past the keypad, a human
KEY_YES, KEY_NO = "1", "2"


class Door(Protocol):
    """The parts of the backend's GuardDoor the Guard uses (engine/safety/guard_door.py)."""

    async def question(self, token: str, language: str) -> str: ...

    async def issue_ticket(self, token: str) -> str: ...

    async def cancel_ticket(self, ticket: str, reason: str) -> bool: ...


class Box(Protocol):
    """The parts of the call box the Guard uses."""

    def start(self, name: str, work: Any, *, crash_safe: bool = False) -> Any: ...

    async def change(self, name: str, work: Any) -> Any: ...


@dataclass(frozen=True)
class GuardSettings:
    reask_wait_s: float  # after an answer, how long to let the Talker reply before asking again
    vad_start_s: float  # how much speech the voice detector needs before it says "speaking"
    stop_words: Mapping[str, str]  # word -> "no" or "stop"
    change_retries: int = 2  # decision 19: a confirm with no answer, sent again with the same token
    expired_reprepares: int = 1  # decision 19: an expired token is prepared again and asked again
    # A spoken yes counts only once the caller has stayed quiet this long after their turn ended
    # (settings.yaml guard.yes_hold_s says why). 0 confirms at once, as before 18 Nov 2026.
    yes_hold_s: float = 0.0
    # A yes held because the caller spoke again, with no new turn finished in this long: the
    # question is asked again, short, instead of waiting for ever.
    held_wait_s: float = 8.0

    @classmethod
    def from_settings(cls, section: dict[str, Any]) -> GuardSettings:
        return cls(float(section["reask_wait_s"]), float(section["vad_start_s"]), dict(section["stop_words"]),
                   int(section["change_retries"]), int(section["expired_reprepares"]),
                   float(section["yes_hold_s"]), float(section["held_wait_s"]))


@dataclass
class YesMoment:
    token: str
    action: str
    question: str  # the full question, in the call's language when it was first asked
    inputs: dict[str, Any] = field(default_factory=dict)  # what was prepared, to prepare it again if it expires
    reprepared: int = 0
    level: int = FULL
    asking: str = ""  # the words of the newest ask
    mark: str = ""
    in_flight: bool = False  # queued for the voice, not yet played to the end
    # What the voice has said since the newest ask was queued. The voice speaks a question
    # in pieces ("... kar doon?" then "Haan ya nahi?"): checking each piece alone for the
    # question's last words missed the end, so no mark was sent and every yes counted as
    # early (food_02, 16 Nov 2026).
    spoken: str = ""
    played_ms: int | None = None  # caller_ms when the newest ask finished playing
    ticket: str | None = None
    asks: int = 0
    answers: list[dict[str, Any]] = field(default_factory=list)
    # A yes the caller talked on after (18 Nov 2026): its words and start wait for the rest of
    # the sentence, and are judged together with it.
    held: str = ""
    held_start_ms: int | None = None


class Guard:
    def __init__(self, door: Door, tools: Tools, box: Box, notebook: Notebook, desk: CallJob, judge: YesJudge,
                 words: SpokenWords, language: StickyLanguage, say: Callable[[str], Awaitable[None]],
                 send_mark: Callable[[str], Awaitable[None]], settings: GuardSettings, diary: Diary,
                 hand_over: Callable[[str], Awaitable[Any]] | None = None,
                 yes_check: YesCheck | None = None) -> None:
        self._door = door
        self._yes_check = yes_check  # Jev's "clear yes?" before the judge (yes_check.py); None: the judge alone
        self._tools = tools
        self._box = box
        self._notebook = notebook
        self._desk = desk
        self._judge = judge
        self._words = words
        self._language = language
        self._say = say
        self._send_mark = send_mark
        self._settings = settings
        self._diary = diary
        self._hand_over = hand_over
        self.moment: YesMoment | None = None  # a yes-question being asked
        self.confirming: YesMoment | None = None  # a yes taken, its ticket being confirmed
        self._marks = 0
        self._turn_start_ms: int | None = None
        self._turns_begun = 0  # how many caller turns began: a yes waits to see if another one does
        self._answers_started = 0  # how many answers began being judged: only the newest may hold a yes
        self._cancelled: set[str] = set()  # tickets stopped by a "ruko"

    @property
    def asking(self) -> bool:
        """A yes-question is open: the note feed keeps the Talker's notes back meanwhile."""
        return self.moment is not None

    # ---- a prepared change arrives (the desk's on_prepared) -----------------------

    async def prepared(self, note: Note) -> None:
        token = str(note.data["data"]["token"])
        question = await self._door.question(token, self._language.value)
        self.moment = YesMoment(token, str(note.data["action"]), question, inputs=dict(note.data.get("inputs") or {}))
        await self._ask()

    # ---- what the call reports ---------------------------------------------------

    def turn_began(self, newest_caller_ms: int) -> None:
        """The voice detector says the caller started talking. It needs vad_start_s of
        speech first, so the words began that much before the newest piece heard."""
        self._turn_start_ms = max(0, newest_caller_ms - round(self._settings.vad_start_s * 1000))
        self._turns_begun += 1

    def partial(self, words: str) -> None:
        """Words still coming in. A listed "ruko"/"nahi" stops at once: a ticket being
        confirmed is cancelled before the backend's save (it waits ~0.8 s first)."""
        if word_list_label(words, self._settings.stop_words) is not None:
            self.stop_ticket("caller said stop")

    def stop_ticket(self, reason: str) -> None:
        """Cancel the ticket being confirmed, if there is one (a "ruko", or the
        stop-or-keep model saying stop): the backend refuses it if not yet saved."""
        confirming = self.confirming
        if confirming is not None and confirming.ticket is not None:
            self._box.start("guard:cancel", self._cancel_ticket(confirming.ticket, reason))

    def turn_finished(self, words: str) -> None:
        if self.moment is not None:
            self._box.start("guard:answer", self.answer(words, self._turn_start_ms))
        self._turn_start_ms = None

    def played(self, name: str, caller_ms: int) -> None:
        # Every step of the mark is in the diary: on 16 Nov 2026 real yeses kept counting
        # as early and the diary could not show which step was missing.
        current = self.moment.mark if self.moment is not None else None
        self._diary.decision("question played", mark=name, current=current, caller_ms=caller_ms)
        if self.moment is not None and name == self.moment.mark:
            self.moment.played_ms = caller_ms

    def cleared(self, dropped_marks: Iterable[str]) -> None:
        """The voice was cut and these marks never played: the caller never heard the end."""
        moment = self.moment
        if moment is not None and moment.mark in set(dropped_marks):
            moment.played_ms = None
            self._ask_soon()

    def speech_ended(self, cut: bool, spoken: str) -> None:
        """The voice stopped. If it was saying our question, mark its end on the wire,
        or, if the caller cut it off, ask it again: it was never fully heard."""
        moment = self.moment
        if moment is not None:
            self._diary.decision("voice stopped during a yes moment", cut=cut, in_flight=moment.in_flight,
                                 spoken=spoken[-80:], mark=moment.mark)
        if moment is None or not moment.in_flight:
            return
        if cut:
            # Cut off before or while it played (an interruption also drops what was
            # still queued for the voice): never fully heard, so asked again.
            moment.in_flight = False
            self._ask_soon()
            return
        moment.spoken += " " + spoken
        if _ends_with(moment.spoken, moment.asking):
            moment.in_flight = False
            self._diary.decision("question mark sent", mark=moment.mark)
            self._box.start("guard:mark", self._send_mark(moment.mark))

    def key(self, digit: str, caller_ms: int) -> None:
        moment = self.moment
        if moment is None or moment.played_ms is None or caller_ms < moment.played_ms:
            return
        if digit in (KEY_YES, KEY_NO):
            label = YES if digit == KEY_YES else NO
            self._box.start("guard:key", self._decide(label, f"key {digit}", early=False, start_ms=caller_ms))

    # ---- the answer ----------------------------------------------------------------

    async def answer(self, words: str, start_ms: int | None) -> None:
        moment = self.moment
        if moment is None:
            return
        loop = asyncio.get_running_loop()
        finished_at, begun = loop.time(), self._turns_begun
        self._answers_started += 1
        mine = self._answers_started
        if moment.held:
            # The rest of a sentence that began with a yes: judged as one answer, from its start.
            words, start_ms = f"{moment.held} {words}".strip(), moment.held_start_ms
            moment.held, moment.held_start_ms = "", None
        # A listed "ruko"/"nahi" first, as always: no AI between the caller and a stop.
        label, also_asks = word_list_label(words, self._settings.stop_words), False
        if label is None:
            label, also_asks = await self._label(moment, words)
        early = start_ms is None or moment.played_ms is None or start_ms < moment.played_ms
        if label == YES and not early and not await self._caller_stayed_quiet(finished_at, begun):
            # 18 Nov 2026, food_04: "Haan, sahi hai." was confirmed while the caller went on
            # "... Flat 2C, Rose Apartments", correcting the address the question said. A yes
            # counts only if the caller stays quiet after it; if they talk on, the yes waits for
            # the rest, and the whole sentence is judged (often "changed").
            if self.moment is moment and self._answers_started == mine:
                self._hold(moment, words, start_ms)
            else:
                # A later turn was already judged on its own (a slow judge): its words are the
                # caller's newest, so this yes is not kept for it. Never confirmed either.
                self._diary.decision("yes dropped: a later answer came first", words=words)
            return
        await self._decide(label, words, early, start_ms, also_asks)

    async def _label(self, moment: YesMoment, words: str) -> tuple[str, bool]:
        """The answer's label, and whether the caller also asked for something else. The yes
        check first (yes_check.py: food_10 lost three yeses said with "aur rider ka number"):
        a clear yes is a yes even with another request in it. Below its line, or with no
        answer from it, the one-word judge labels the answer, as before 20 Nov 2026."""
        if self._yes_check is not None:
            answer = await self._yes_check.clear_yes(moment.question, words)
            if answer is not None:
                self._diary.decision("yes check", words=words, yes=round(answer.yes, 3),
                                     also_asks=round(answer.also_asks, 3))
                if answer.yes >= self._yes_check.line:
                    return YES, answer.also_asks >= self._yes_check.line
        return await self._judge.label(moment.question, words), False

    async def _caller_stayed_quiet(self, finished_at: float, begun: int) -> bool:
        """Wait until yes_hold_s after the turn ended (the judge's own time counts towards it),
        then: did no new caller turn begin meanwhile?"""
        left = finished_at + self._settings.yes_hold_s - asyncio.get_running_loop().time()
        if left > 0:
            await asyncio.sleep(left)
        return self._turns_begun == begun

    def _hold(self, moment: YesMoment, words: str, start_ms: int | None) -> None:
        """Keep the yes's words for the caller's next turn (a turn that ends with no words, a
        cough, joins nothing, and the yes is judged again on its own). If no turn finishes at
        all, ask again, short, after held_wait_s: never wait for ever."""
        moment.held, moment.held_start_ms = words, start_ms
        self._diary.decision("yes held: caller talked on", words=words, hold_s=self._settings.yes_hold_s)
        self._box.start("guard:held", self._held_too_long(moment, words))

    async def _held_too_long(self, moment: YesMoment, words: str) -> None:
        await asyncio.sleep(self._settings.held_wait_s)
        if self.moment is moment and moment.held == words:
            # The started turn never finished: its "turn began" mark is stale, so cleared too.
            moment.held, moment.held_start_ms, self._turn_start_ms = "", None, None
            self._diary.decision("yes held too long: asked again", words=words)
            await self._ask(short=True)

    async def _decide(self, label: str, words: str, early: bool, start_ms: int | None, also_asks: bool = False) -> None:
        moment = self.moment
        if moment is None:
            return
        moment.answers.append({"words": words, "label": label, "early": early})
        self._diary.decision("yes moment", question=moment.asking, words=words, label=label, early=early,
                             played_ms=moment.played_ms, start_ms=start_ms, level=moment.level)
        if label == YES and early:
            # Decision 22: a "haan" that began before the question finished never counts.
            # Ask again, short: the caller may have meant it, but must say it after hearing it all.
            self._ask_soon(short=True)
        elif label == YES:
            await self._yes(moment, words, also_asks)
        elif label in (NO, STOP):
            await self._stopped(moment, label, words)
        elif label == CHANGED:
            await self._changed(moment, words)
        elif label == OTHER:
            # Answered by the Talker and the Thinker; asked again, not counted (decision 23).
            self._note_answer(f"other: {words}")
            self._ask_soon()
        else:  # unclear
            moment.level += 1
            if moment.level > KEYPAD:
                await self._stopped(moment, UNCLEAR, words, "no clear answer after asking three times and the keypad")
                await self._give_up("no clear yes or no, even on the keypad")
            else:
                self._ask_soon()

    async def _yes(self, moment: YesMoment, words: str, also_asks: bool = False) -> None:
        # "also_asks": the yes came with another request ("aur rider ka number"); the Thinker
        # answers that part while the change goes ahead (20 Nov 2026, food_10).
        self._note_answer(f"yes: {words}", {"also_asks": True} if also_asks else None)
        self.moment, self.confirming = None, moment
        try:
            moment.ticket = await self._door.issue_ticket(moment.token)
        except TicketRefused as refused:
            self.confirming = None
            if "expired" in str(refused):
                # The door's own words (engine/safety/guard_door.py): the token ran out while
                # the question was asked. Decision 19: prepare again, ask again.
                await self._prepare_again(moment)
            else:
                self._result(moment, REFUSED_OUTCOME, str(refused))
            return
        try:
            outcome = await self._confirm(moment)
        finally:
            self.confirming = None
        if outcome == EXPIRED:
            await self._prepare_again(moment)
        elif outcome == UNKNOWN_OUTCOME:
            await self._give_up("the change's result is unknown")

    async def _confirm(self, moment: YesMoment) -> str:
        """Pressed by code the instant the ticket exists (decision 17), shielded. With no
        answer it is sent again with the same token: the backend checks "already done"
        first, so a confirm that did save is never done twice."""
        job = self._desk.job
        if job is not None:
            job.change_sent()
        outcome, reason, answer = UNKNOWN_OUTCOME, "", {}
        try:
            for attempt in range(1 + self._settings.change_retries):
                try:
                    reply = await self._box.change(f"{CONFIRM}:{moment.action}",
                                                   self._tools.call(CONFIRM, {"token": moment.token, "ticket": moment.ticket}))
                    outcome, reason = self._outcome(moment, reply)
                    if outcome == DONE:
                        # The backend's own answer, as it came (decision 16): the refund a cancel
                        # made, the amount refunded. 18 Nov 2026, food_01: only "done" was kept, so
                        # nobody could say the ₹450 refund, and "decide" later said no tool could
                        # check it, and handed over.
                        answer = dict(reply.data)
                except BoxClosing:
                    outcome, reason = STOPPED, "the call ended before confirm was sent"
                except Exception as error:
                    outcome, reason = UNKNOWN_OUTCOME, f"no answer from the service ({type(error).__name__})"
                if outcome != UNKNOWN_OUTCOME:
                    break
                self._diary.error(f"confirm try {attempt + 1} has no answer: {reason}")
        finally:
            if job is not None:
                job.change_answered()
        if outcome != EXPIRED:
            self._result(moment, outcome, reason, answer)
        return outcome

    def _outcome(self, moment: YesMoment, reply: Any) -> tuple[str, str]:
        if reply is UNKNOWN or not isinstance(reply, ToolReply):
            return UNKNOWN_OUTCOME, "no answer in time"
        if reply.status in (OK, ALREADY_DONE):
            return DONE, ""
        if moment.ticket in self._cancelled:
            return STOPPED, "the caller said stop before it was saved"
        if reply.status == REFUSED:
            return REFUSED_OUTCOME, reply.message
        if reply.status == EXPIRED:
            return EXPIRED, reply.message
        return UNKNOWN_OUTCOME, reply.message  # an error: never "done" when unsure

    async def _prepare_again(self, moment: YesMoment) -> None:
        """Decision 19: the token expired before the yes was used: prepare the same change
        again and ask again, a limited number of times."""
        if moment.reprepared >= self._settings.expired_reprepares:
            self._result(moment, UNKNOWN_OUTCOME, "the change expired again")
            await self._give_up("the change kept expiring")
            return
        reply = await self._tools.call(PREPARE_PREFIX + moment.action, dict(moment.inputs))
        token = reply.data.get("token") if reply.status == OK else None
        if not token:
            self._result(moment, REFUSED_OUTCOME, reply.message)
            return
        self.moment = YesMoment(str(token), moment.action, await self._door.question(str(token), self._language.value),
                                inputs=moment.inputs, reprepared=moment.reprepared + 1)
        await self._ask()

    async def _give_up(self, why: str) -> None:
        """Still stuck: the truth is already in a [result] note; then one hand-over."""
        self._diary.decision("guard gives up", why=why)
        if self._hand_over is not None:
            await self._hand_over(why)

    async def _stopped(self, moment: YesMoment, label: str, words: str, why: str = "") -> None:
        self.moment = None
        if moment.ticket is not None:
            await self._cancel_ticket(moment.ticket, f"caller said {label}")
        self._note_answer(f"{label}: {words}" + (f" ({why})" if why else ""))
        self._result(moment, STOPPED, why or f"the caller said {label}")

    async def _changed(self, moment: YesMoment, words: str) -> None:
        """The caller corrected or replaced what was asked (the owner's choice, 18 Nov 2026):
        close the question, never ask it again, confirm nothing. The Thinker decides again with
        the caller's words and prepares the new change, with its own question. Nothing was saved:
        a change is saved only after a yes, so the old token is just never used. No result note:
        the Thinker would read a stale result at its next yes-question."""
        self.moment = None  # a re-ask already waiting sees this and stays quiet (_ask_after)
        self._diary.decision("yes moment changed", question=moment.asking, words=words)
        if moment.ticket is not None:
            await self._cancel_ticket(moment.ticket, "caller changed the request")
        self._note_answer(f"changed: {words}")

    async def _cancel_ticket(self, ticket: str, reason: str) -> None:
        if await self._door.cancel_ticket(ticket, reason):
            self._cancelled.add(ticket)
        else:
            self._diary.decision("ticket too late to stop", reason=reason)

    # ---- notes and asking --------------------------------------------------------------

    def _version(self) -> int:
        job = self._desk.job
        return job.version if job is not None else 0

    def _note_answer(self, text: str, data: Mapping[str, Any] | None = None) -> None:
        self._notebook.add(YES_ANSWER, text, self._version(), dict(data or {}))

    def _result(self, moment: YesMoment, outcome: str, reason: str, answer: Mapping[str, Any] | None = None) -> None:
        """The change's result note. A finished change carries the backend's answer, in its
        data ("backend") and in its words, so the Talker can say it and the Thinker sees it in
        "already done" ("cancel_order: done, status cancelled, refund ₹450")."""
        answer = dict(answer or {})
        self._diary.decision("change result", action=moment.action, outcome=outcome, reason=reason, backend=answer)
        facts = ", ".join(f"{key} {value}" for key, value in answer.items() if key != "order_no")
        self._notebook.add(RESULT, f"{moment.action}: {outcome}" + (f", {facts}" if facts else ""), self._version(),
                           {"action": moment.action, "outcome": outcome, "reason": reason, "backend": answer})

    def _ask_soon(self, short: bool = False) -> None:
        """Ask again once the Talker has had its say about the caller's words, and only
        if the caller is quiet then and nothing else asked meanwhile."""
        moment = self.moment
        if moment is not None:
            self._box.start("guard:ask", self._ask_after(self._settings.reask_wait_s, moment, moment.asks, short))

    async def _ask_after(self, wait_s: float, moment: YesMoment, asks: int, short: bool) -> None:
        await asyncio.sleep(wait_s)
        if self.moment is moment and moment.asks == asks and self._turn_start_ms is None:
            await self._ask(short)

    async def _ask(self, short: bool = False) -> None:
        moment = self.moment
        assert moment is not None
        language = self._language.value
        if short:
            moment.asking = self._words.phrase("line", "ask_short", language)
        elif moment.level == FULL:
            moment.asking = moment.question
        elif moment.level == AGAIN:
            moment.asking = self._words.phrase("line", "ask_again", language)
        else:
            moment.asking = self._words.phrase("line", "ask_keypad", language)
        self._marks += 1
        moment.asks += 1
        moment.mark = f"yes-{self._marks}"
        moment.in_flight, moment.played_ms, moment.spoken = True, None, ""
        await self._say(moment.asking)


def _ends_with(spoken: str, asked: str) -> bool:
    """Did this stretch of speech include the end of what was asked? The voice reports
    its words in pieces, so the last few words are compared, spaces ignored."""
    tail = "".join(asked.split()[-3:]).lower()
    return bool(tail) and tail in "".join(spoken.split()).lower()
