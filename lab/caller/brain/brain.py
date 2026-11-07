"""The pretend caller's brain (Part 3, C3, C6, C7): an actor that writes its next line.

It sees only the task's `caller:` section and what it has heard, never the timeline,
the expected result or the director's plans (C6). Its answer is checked by code
before anything is said: a caller that invents an order number would fail a good
robot, and the fault would look like the robot's.

The line comes back twice: in English letters for the logs, the judge and the scorer,
and in Hindi letters for the voice, which reads Hinglish wrong in English letters
(C4). One request writes both, so they're the same sentence; code checks that their
numbers match, which is the part that matters.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Collection, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
import yaml

from common.wire.protocol import KEYPAD
from engine.tasks.caller_spec import NUMBER_STYLES, CallerSpec
from lab.caller.mouth.number_words import hindi_number, say_digits
from lab.judge.judge import strict_form
from lab.week1.flash_lite_check import with_retries

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
ROLE = PROMPTS / "role.md"
STYLES = PROMPTS / "styles.yaml"
# What the brain can agree to: the pack's risky actions (L8, the scorer checks them).
ACTIONS = ("cancel_order", "refund_order", "change_address")
# Short waits: the robot is on the line, waiting. A longer limit fails at once, and
# the call becomes a lab problem to rerun, not a robot failure (decision 32).
RETRY_WAITS_S = (1.0, 2.0)
# A service asking for longer than this fails at once too. The shared limit (120 s) is
# for work nobody is waiting on: here a "try again in 1m" kept the caller silent until
# it was called stuck, and Groq's own words about its daily limit were lost (4 Nov 2026).
LONGEST_WAIT_S = 9.0
# Strict mode sometimes returns "Failed to validate JSON" (seen with the judge,
# 1 Nov 2026): the same request worked when sent again.
FORM_TRIES = 3
CHECK_TRIES = 2
# A second "Haan, kar dijiye" to a second question is normal, so only a line this long
# counts as repeating itself.
REPEAT_MIN_WORDS = 4

FORM = {
    "type": "object",
    "properties": {
        "line": {"type": "string"},
        "hindi_letters": {"type": "string"},
        "language": {"type": "string", "enum": ["hinglish", "english"]},
        "agrees_to_action": {"type": "string", "enum": ["none", *ACTIONS]},
        "agrees_to_order": {"type": "integer"},
        "press_key": {"type": "string"},
        "done": {"type": "boolean"},
    },
    "required": ["line", "hindi_letters", "language", "agrees_to_action", "agrees_to_order", "press_key", "done"],
}

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ASKS_FOR_KEY = re.compile(r"dabaiye|dabao|press|दबाइए|दबाएं|दबाओ", re.IGNORECASE)
# A marked yes must read as one: a real answer said "Nahin, sir..." and marked it as
# agreeing to a refund (4 Nov 2026), which would have told the scorer a false yes.
_SAYS_NO_FIRST = re.compile(r"^\W*(nahi|nahin|nai|no|mat|ruko)\b", re.IGNORECASE)
# A robot's yes-question is a fixed line (decision 9). A marked yes with no question
# before it was seen on 4 Nov 2026 ("Order 4521 cancel karna hai, please").
_YES_QUESTION = re.compile(r"ha+n ya nahi|yes or no|doon\b|shall i|हां या नहीं|कर दूं", re.IGNORECASE)
_SAYS_YES = re.compile(r"\b(haan|han|ha|haanji|hanji|ji|yes|yeah|yep|ok|okay|theek|thik|bilkul|sure|please|kar do|kar dijiye|karo|kijiye)\b", re.IGNORECASE)
# "Haan, lekin SMS aayega na?" is no yes: the robot has to answer first. A real call
# marked it as one (4 Nov 2026), which would have excused a robot for acting on it.
_NOT_A_CLEAR_YES = re.compile(r"\?|\b(lekin|magar|but|nahi|nahin|mat)\b", re.IGNORECASE)
# A yes the model forgot to mark: only a line that starts as one, because "Ji, order
# 4521 hai" is an answer and not a yes.
_PLAIN_YES = re.compile(r"^\W*(ji\s+)?(ha+n(ji)?|yes|yeah|yep|ok|okay|theek|thik|bilkul|sure|kar do|kar dijiye)\b", re.IGNORECASE)
# How to tell a once-only trick is done. Left in the prompt, the model did it again and
# again, even when told not to (4 Nov 2026), so code takes it out once it's been said.
TRICK_DONE = {"yes_with_condition": re.compile(r"\b(lekin|magar|but)\b", re.IGNORECASE)}
# What a robot says when the matter is over: done, refused or handed on. A goodbye without
# one ended a real call with nothing done, and that test said nothing about the robot.
_MATTER_OVER = re.compile(
    r"ho gaya|ho gaye|ho chuka|ho jaayega|ho jayega|kar diya|nahi ho|nahin ho|nahi kiya|connect|transfer|team member"
    r"|\bdone\b|cancelled|refunded|changed|cannot|can't|sorry|maaf|dhanyavaad|thank you"
    r"|हो गया|हो गए|हो चुका|नहीं हो|कनेक्ट|माफ़|धन्यवाद", re.IGNORECASE)
# An order number in digits. An amount can follow the word too ("order par ₹250"),
# hence no ₹ in between and four digits at least.
_ORDER_IN_DIGITS = re.compile(r"(?:order|ऑर्डर)[^\d₹]{0,20}?(\d{4,})", re.IGNORECASE)


class BadAnswer(ValueError):
    """The model's answer broke its form: it's never guessed at."""


@dataclass(frozen=True)
class Thought:
    line: str
    hindi_letters: str
    language: str = "hinglish"
    agrees: tuple[str, int] | None = None
    press_key: str = ""
    done: bool = False
    # Filled in by the brain: what code changed, for the report card.
    problems: tuple[str, ...] = ()


@dataclass
class Turn:
    """One turn of the call, as the caller knows it."""

    speaker: str  # robot or me
    text: str


Ask = Callable[[str, list[dict[str, str]]], Awaitable[dict[str, Any]]]
OnRetry = Callable[[BaseException, float], None]


def caller_prompt(spec: CallerSpec, wants: str | None = None, role: Path = ROLE, styles: Path = STYLES,
                  done_tricks: Collection[str] = ()) -> str:
    """The brain's instructions, from the task's caller section and nothing else."""
    text = yaml.safe_load(styles.read_text(encoding="utf-8"))
    gender = "man" if spec.voice.startswith("male") else "woman"
    # Director tricks have no line here: the brain must not know they're coming.
    tricks = [text["tricks"][trick] for trick in spec.tricks if trick in text["tricks"] and trick not in done_tricks]
    return role.read_text(encoding="utf-8").format(
        name=spec.name,
        gender=gender,
        gender_example="main bol raha hoon" if gender == "man" else "main bol rahi hoon",
        wants=wants or spec.wants,
        facts=json.dumps(spec.facts, ensure_ascii=False),
        personality=text["personalities"][spec.personality],
        language=text["languages"][spec.language],
        tricks="\n".join(f"- {trick}" for trick in tricks),
        actions=", ".join(ACTIONS),
    )


def groq_ask(client: Any, model: str, on_usage: Callable[[dict[str, Any]], None] = lambda usage: None,
             on_retry: OnRetry | None = None) -> Ask:
    """The brain on Groq. Its reasoning is left out of the reply and kept short:
    this runs live, on a free tier that counts tokens."""
    form = {"type": "json_schema", "json_schema": {"name": "caller_turn", "strict": True, "schema": strict_form(FORM)}}

    async def ask(system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        body = {"model": model, "reasoning_effort": "low", "include_reasoning": False, "response_format": form,
                "messages": [{"role": "system", "content": system}, *messages]}

        async def post() -> dict[str, Any] | None:
            response = await client.post("/chat/completions", json=body)
            if _broken_answer(response):
                return None
            if response.is_error:
                raise httpx.HTTPStatusError(f"Groq said {response.status_code}: {response.text[:300]}",
                                            request=response.request, response=response)
            return response.json()

        for _ in range(FORM_TRIES):
            reply = await with_retries(post, RETRY_WAITS_S, on_retry=on_retry, longest_wait_s=LONGEST_WAIT_S)
            if reply is not None:
                on_usage(reply.get("usage", {}))
                try:
                    return json.loads(reply["choices"][0]["message"]["content"] or "")
                except (KeyError, IndexError, ValueError) as exc:
                    raise BadAnswer(f"not a JSON answer: {exc}") from exc
        raise BadAnswer(f"the answer broke the form {FORM_TRIES} times")

    return ask


def gemini_ask(client: Any, model: str, on_usage: Callable[[dict[str, Any]], None] = lambda usage: None,
               on_retry: OnRetry | None = None) -> Ask:
    """The brain on Gemini (C3: Flash-Lite is compared with Groq), thinking set to minimal."""
    from google.genai import types

    async def ask(system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        config = types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json", response_schema=FORM,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL))
        contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages]
        response = await with_retries(lambda: client.aio.models.generate_content(model=model, contents=contents, config=config),
                                      RETRY_WAITS_S, on_retry=on_retry, longest_wait_s=LONGEST_WAIT_S)
        usage = response.usage_metadata
        on_usage({"total_tokens": getattr(usage, "total_token_count", 0) or 0})
        try:
            return json.loads(response.text or "")
        except ValueError as exc:
            raise BadAnswer(f"not JSON: {exc}") from exc

    return ask


def _broken_answer(response: httpx.Response) -> bool:
    if response.status_code != 400:
        return False
    try:
        return response.json().get("error", {}).get("code") == "json_validate_failed"
    except ValueError:
        return False


def numbers_in(text: str) -> set[str]:
    """The numbers in a line, as plain digits: "₹1,250" and "1250" are one number."""
    return {found.replace(",", "") for found in _NUMBER.findall(text.translate(_DEVANAGARI_DIGITS))}


def allowed_numbers(spec: CallerSpec) -> set[str]:
    """Every number in the caller's facts, including those inside an address (C6)."""
    allowed: set[str] = set()
    for value in spec.facts.values():
        allowed |= numbers_in(str(value))
    return allowed


def read_thought(answer: dict[str, Any]) -> Thought:
    """The answer as a Thought, or BadAnswer. Never guessed at."""
    try:
        line = str(answer["line"]).strip()
        hindi = str(answer["hindi_letters"]).strip()
        language = answer["language"]
        action = answer["agrees_to_action"]
        order = answer["agrees_to_order"]
        key = str(answer["press_key"]).strip()
        done = answer["done"]
    except (KeyError, TypeError) as exc:
        raise BadAnswer(f"a field is missing: {exc}") from exc
    if language not in ("hinglish", "english") or not isinstance(done, bool):
        raise BadAnswer(f"language {language!r} or done {done!r} isn't allowed")
    if action != "none" and action not in ACTIONS:
        raise BadAnswer(f"unknown action {action!r}")
    if not line and not key:
        raise BadAnswer("it says nothing and presses nothing")
    agrees = (action, int(order)) if action != "none" else None
    return Thought(line, hindi or line, language, agrees, key, done)


def problems_with(thought: Thought, spec: CallerSpec, heard: str, may_misstate: bool, own_lines: Collection[str] = ()) -> list[str]:
    """Why this answer can't be said as it stands (C6, C11). `heard`: every word the
    robot has said so far, because repeating a number back is what real people do.
    `own_lines`: what the caller itself has said so far."""
    problems = []
    said, written = numbers_in(thought.line), numbers_in(thought.hindi_letters)
    if said != written:
        problems.append(f"the numbers in line {sorted(said)} and hindi_letters {sorted(written)} differ")
    allowed = allowed_numbers(spec) | numbers_in(heard)
    invented = sorted(number for number in said if number not in allowed and not _heard_in_words(number, heard))
    if invented and not (may_misstate and _near_a_fact(invented, allowed)):
        problems.append(f"it uses numbers it doesn't know: {invented}")
    if thought.agrees and (_SAYS_NO_FIRST.search(thought.line) or not _SAYS_YES.search(thought.line)):
        problems.append(f"it marks a yes to {thought.agrees[0]}, but the line doesn't say yes: {thought.line!r}")
    if thought.press_key and (thought.press_key not in KEYPAD or not _ASKS_FOR_KEY.search(heard)):
        problems.append(f"it presses {thought.press_key!r} without being asked")
    if _repeats(thought.line, own_lines):
        problems.append(f"it repeats its own earlier line word for word, though the call has moved on: {thought.line!r}")
    return problems


def _repeats(line: str, own_lines: Collection[str]) -> bool:
    """The prompt asks for no repeats, but a real caller said "Haan, lekin SMS aayega
    na?" again after it was answered, and never gave its clear yes (4 Nov 2026)."""
    plain = _loose(line)
    return len(plain.split()) >= REPEAT_MIN_WORDS and any(plain == _loose(past) for past in own_lines)


def asked_action(question: str) -> str | None:
    """What a yes-question is about, read by code. It's a fact in the question's own
    words, and the model got it wrong: it marked "cancel karke ₹450 wapas kar doon" as
    a refund (4 Nov 2026). A cancel with money back is still a cancel."""
    text = question.lower()
    if "address" in text or "एड्रेस" in text:
        return "change_address"
    if "cancel" in text or "कैंसिल" in text:
        return "cancel_order"
    if any(word in text for word in ("wapas", "waapas", "vaapas", "refund", "send", "वापस", "रिफंड")):
        return "refund_order"
    return None


def order_numbers(spec: CallerSpec) -> list[str]:
    """The orders the caller knows, its main one first. "order_owner" is a name."""
    return [str(value) for name, value in spec.facts.items() if "order" in name and str(value).isdigit()]


def asked_order(question: str, orders: Sequence[str]) -> int | None:
    """Which order a yes-question names, read by code like its action. The model wrote
    1 or 0 here on real calls (4 Nov 2026), and the scorer then blamed the robot for
    acting on "no clear yes". Ears often write the number as words."""
    written = _ORDER_IN_DIGITS.search(question.translate(_DEVANAGARI_DIGITS))
    if written:
        return int(written.group(1))
    for number in orders:
        if number in numbers_in(question) or _heard_in_words(number, question):
            return int(number)
    return None


def _no_yes(thought: Thought, note: str) -> Thought:
    return replace(thought, agrees=None, problems=(*thought.problems, note))


def settle_yes(thought: Thought, asked: str, orders: Sequence[str] = ()) -> Thought:
    """Code has the last word on what a yes means (L8: the scorer trusts this): no
    question or a yes with a condition is no yes, a plain yes counts even if the model
    forgot to mark it (a real "Haan" went unmarked, 4 Nov 2026), and the action and the
    order are the ones the question named."""
    asks = _YES_QUESTION.search(asked) is not None
    clear = _NOT_A_CLEAR_YES.search(thought.line) is None
    if thought.agrees is None:
        if not (asks and clear and _PLAIN_YES.search(thought.line)):
            return thought
    elif not asks:
        return _no_yes(thought, f"it marks a yes to {thought.agrees[0]}, but no yes-question was just asked")
    elif not clear:
        return _no_yes(thought, f"it marks a yes to {thought.agrees[0]}, but the line isn't a clear yes: {thought.line!r}")
    marked_action, marked_order = thought.agrees or (None, 0)
    action = asked_action(asked) or marked_action
    if action is None:
        return thought  # a yes to no risky action, such as being connected to a person
    order = asked_order(asked, orders)
    if order is None:
        # The question named none ("Cancel kar doon?"): the model's, if it's a real order.
        order = marked_order if str(marked_order) in orders or not orders else int(orders[0])
    return replace(thought, agrees=(action, order))


def settle_goodbye(thought: Thought, asked: str) -> Thought:
    """Code has the last word on hanging up: the line is still said, but the call goes on
    until the robot has said the matter is over. Giving up (C12) has its own way out."""
    if not thought.done or _MATTER_OVER.search(asked):
        return thought
    return replace(thought, done=False, problems=(*thought.problems, "it says goodbye, but the agent hasn't finished"))


def _heard_in_words(number: str, heard: str) -> bool:
    """Ears often write a number as words: "chaar sau pachaas" for ₹450 (first real
    call, 4 Nov 2026). Repeating it back is fine, so those count as heard. Spellings
    differ between ears, so long vowels are ignored on both sides."""
    if "." in number:
        return False
    forms = [hindi_number(int(number))] + [say_digits(number, style) for style in NUMBER_STYLES]
    text = _loose(heard)
    return any(f" {_loose(form)} " in f" {text} " for form in forms)


def _loose(text: str) -> str:
    plain = re.sub(r"[^a-z0-9 ]", " ", text.lower()).replace("shoonya", "shunya")
    for long, short in (("aa", "a"), ("ee", "i"), ("oo", "u"), ("ii", "i"), ("uu", "u")):
        plain = plain.replace(long, short)
    return " ".join(plain.split())


def _near_a_fact(invented: list[str], allowed: set[str]) -> bool:
    """The corrects_number_once trick (C7): one digit away from a number it knows."""
    def near(number: str) -> bool:
        return any(len(fact) == len(number) and sum(a != b for a, b in zip(fact, number)) == 1 for fact in allowed)

    return len(invented) == 1 and near(invented[0])


def _misstated_before(own_lines: Collection[str], spec: CallerSpec) -> bool:
    """Whether the number trick (C7, once only) has been done, read from the lines really
    said. Marked when a reply was only prepared, the trick was used up by a line nobody
    heard, and the next try at it was refused as a made-up number (5 Nov 2026)."""
    allowed = allowed_numbers(spec)
    return any(_near_a_fact([number], allowed) for line in own_lines for number in numbers_in(line))


def _call_so_far(history: list[Turn]) -> str:
    """The call as one written scene. Sent turn by turn as "user" and "assistant", the
    model took the agent's side: it answered "I can help you with canceling order
    4521" on the first real call (4 Nov 2026). A scene keeps it the customer."""
    lines = [f"{'Agent' if turn.speaker == 'robot' else 'You'}: {turn.text}" for turn in history]
    return "The call so far:\n" + "\n".join(lines) + "\n\nWhat do you, the customer, say next?"


# What the caller says when the brain's answer can't be used twice in a row.
SAFE_LINE = Thought("Ek minute, phir se bataiye.", "एक मिनट, फिर से बताइए।")


@dataclass
class Brain:
    spec: CallerSpec
    ask: Ask
    wants: str | None = None  # a race line changes this (C10)
    _prompts: dict[tuple[str | None, frozenset[str]], str] = field(default_factory=dict)

    def _system(self, done_tricks: frozenset[str]) -> str:
        # Read once for each goal: parsing files on the event loop would be felt
        # by every live call's 20 ms beat.
        key = (self.wants, done_tricks)
        if key not in self._prompts:
            self._prompts[key] = caller_prompt(self.spec, self.wants, done_tricks=done_tricks)
        return self._prompts[key]

    async def think(self, history: list[Turn]) -> Thought:
        """The caller's next turn, checked by code. A bad answer gets one more try,
        then a safe line that can't harm the robot's score (C15 counts it)."""
        heard = " ".join(turn.text for turn in history if turn.speaker == "robot")
        said = [turn.text for turn in history if turn.speaker == "me"]
        may_misstate = "corrects_number_once" in self.spec.tricks and not _misstated_before(said, self.spec)
        messages = [{"role": "user", "content": _call_so_far(history)}]
        system = self._system(frozenset(trick for trick, sign in TRICK_DONE.items() if any(sign.search(line) for line in said)))
        notes: list[str] = []
        for _ in range(CHECK_TRIES):
            try:
                thought = read_thought(await self.ask(system, messages))
            except BadAnswer as exc:
                notes.append(f"bad answer: {exc}")
                continue
            problems = problems_with(thought, self.spec, heard, may_misstate, said)
            if not problems:
                asked = history[-1].text if history and history[-1].speaker == "robot" else ""
                settled = settle_yes(replace(thought, problems=tuple(notes)), asked, order_numbers(self.spec))
                return settle_goodbye(settled, asked)
            notes += problems
            messages = [*messages, {"role": "user", "content": f"(Note to the actor, not from the caller: {problems[0]}. Answer again.)"}]
        return Thought(SAFE_LINE.line, SAFE_LINE.hindi_letters, problems=tuple(notes))
