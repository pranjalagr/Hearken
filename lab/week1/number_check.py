"""Can the Talker's AI point at the order number in what the ears wrote? (Part 2,
decision 27, 7 Nov 2026.)

The AI copies the words where the caller said the order number; code
(agents/common/number_reader.py) turns them into digits and checks them. The AI never
gives digits. Each answer is marked twice, by code:

- the AI's pick, word by word against the labels (lab/week1/number_transcripts.py):
  the right words, another number (an amount, the wrong one before a correction),
  nothing, or something else;
- what the robot would end with: right; wrong number accepted (the dangerous one: only
  the read-back would still save it); asked again or missed (safe, a turn lost). A
  failure also says whose it was: the AI's (other words) or the reader's (it misread
  the right words), so one can't hide behind the other (the user asked for this).

Built to waste no tokens (the user, 7 Nov 2026): every answer is written down the
moment it arrives, a rerun skips what's done, each key is paced under its per-minute
limit, a key at its daily limit drops out and its cases go to the others, and a case
that keeps failing is given up after a few tries instead of hanging.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.common.number_reader import _FILLERS, NONE, read_number, words_of
from lab.week1.number_transcripts import ASK, OTHER, TranscriptCase, load_transcripts

ROOT = Path(__file__).resolve().parents[2]
LENGTH = 4  # the food pack's order numbers
MODEL = "openai/gpt-oss-20b"  # Groq: the Talker's model isn't decided yet
EFFORT = "low"
GAP_S = 3.2  # one key's 8,000 tokens a minute, at ~400 tokens a request
# 8 Nov 2026: Sarvam's model for finding numbers, chosen on the 831 cases (8 wrong and 51
# missed of 777, against Qwen's 10 and 154). Its request limit isn't published: with no
# gap a one-at-a-time run was refused now and then, so a small gap, not Groq's 3.2 s.
SARVAM_MODEL = "sarvam-105b-conversations"
SARVAM_GAP_S = 0.4
TIMEOUT_S = 30.0  # one request; past this it's stuck, not slow
CASE_TRIES = 3  # a case that fails this often (not a limit) is given up and reported
RIGHT, WRONG, ASKED, MISSED = "right", "wrong number accepted", "asked again", "missed"
# List mode: two or more order numbers, or one the caller isn't sure of, go to the
# Thinker, which sees the whole call and chooses or asks (the user, 7 Nov 2026).
SENT = "sent to the Thinker"
PICK_RULES = """You read one turn of a phone call to a food delivery support line in India, as speech-to-text wrote it. The caller may give an order number: 4 digits, said as single digits, pairs or words, in Hindi or English, often with fillers, restarts, corrections and mishearings.

Copy, exactly as it is written, the stretch of words where the caller says the order number they want help with. If they correct themselves, take the last number they settle on. Copy only the number's own words, as one unbroken stretch, and nothing around them. Never fix, translate or reorder the words, and never write digits that aren't written there.
If there is no order number, if the caller isn't sure which one, or if they take it back without a new one, answer with an empty string."""
PICK_FORM = {"type": "object", "properties": {"words": {"type": "string"}}, "required": ["words"]}
# List mode (7 Nov 2026): the AI only finds and tags every number; it never chooses.
# Choosing between numbers ("X nahi, Y wala") is what it got wrong in run1, and what the
# Thinker is for. Said positively, with examples, not a list of don'ts.
# v2 (7 Nov 2026): Qwen returned nothing for 29% of turns, numbers said only in words
# ("saat nau saat shoonya"), and tagged order numbers without the word "order" as other.
# So it now says numbers can be words, and what an order number looks like. The
# examples use numbers and wording that aren't in the test set.
# v3 tried and dropped (8 Nov 2026): a separate true/false, "does the caller still mean
# it", instead of the unsure tag. On the 203 hard cases on Sarvam it made 6 wrong and 12
# missed, against v2's 8 and 5: no clear gain, and a longer answer. On Qwen it doubled the
# misses (9 to 20 of 41). So the simpler v2 stayed.
# v4 (8 Nov 2026): v2 said to copy "one unbroken stretch", so the AI handed the reader
# "tiraasi ji sadsath", which it can't read. The user: the AI leaves those words out.
# v5 (9 Nov 2026): v4 went from 8 to 15 wrong numbers, mostly garbled corrections ("9042 oh
# no, sorry nine thousand 03 do") where the broken second number was tagged other. So one
# example shows both as unsure. Its numbers aren't in the test set.
LIST_RULES = """You read one turn of a phone call to a food delivery support line in India, as speech-to-text wrote it. The caller speaks Hindi, English or both, written in English letters.

Find every number the caller says. A number can be digits (4521) or words, in Hindi or English, said one digit at a time or in pairs: "chaar paanch do ek", "four five two one", "paintaalis ikkees", "chaar hazaar paanch sau ikkees".

List the numbers in the order they are said. For each, copy only its number words exactly as written, leaving out any other words said in between, and give it one tag:
- order: a number that could be the caller's order number, usually 4 digits, with or without the word "order"
- other: an amount, a price, a time, an address, a count, or any other number
- unsure: an order number the caller isn't sure of, or takes back
Give an empty list only when the caller says no number at all.

Examples:
"teen aath do chhah" -> [{"words": "teen aath do chhah", "tag": "order"}]
"Haan ji, chhattees baarah, Priya Sharma" -> [{"words": "chhattees baarah", "tag": "order"}]
"6181 nahi, six one eight two" -> [{"words": "6181", "tag": "order"}, {"words": "six one eight two", "tag": "order"}]
"Mere do order hain, 3355 aur teen teen paanch chhah" -> [{"words": "3355", "tag": "order"}, {"words": "teen teen paanch chhah", "tag": "order"}]
"Order 2718 par chaar sau pachaas rupaye wapas chahiye" -> [{"words": "2718", "tag": "order"}, {"words": "chaar sau pachaas", "tag": "other"}]
"Do aath saat ek, galat hai, ruko dekhta hoon" -> [{"words": "Do aath saat ek", "tag": "unsure"}]
"Order saat hmm do, ek sec, nau chaar hai" -> [{"words": "saat do nau chaar", "tag": "order"}]
"Order 38 aa 67 kab aayega" -> [{"words": "38 67", "tag": "order"}]
"3167 oh no, sorry three thousand 16 aath" -> [{"words": "3167", "tag": "unsure"}, {"words": "three thousand 16 aath", "tag": "unsure"}]
"Do din pehle order kiya tha, cancel kar do" -> []"""
LIST_FORM = {"type": "object", "required": ["numbers"], "properties": {"numbers": {"type": "array", "items": {
    "type": "object", "required": ["words", "tag"],
    "properties": {"words": {"type": "string"}, "tag": {"type": "string", "enum": ["order", "other", "unsure"]}}}}}}
Ask = Callable[[str, str], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Result:
    case_id: str
    kind: str
    run: int  # which try of this case (the AI varies)
    picked: str
    pick: str  # the AI's pick against the labels
    accepted: str | None  # the number the robot would act on
    outcome: str
    fault: str = ""  # for failures: "AI" or "reader"
    key: str = ""
    seconds: float = -1.0  # the wait for this answer, retries included; -1 if not timed


def _plain(text: str) -> list[str]:
    return [word for word in words_of(text) if word not in _FILLERS]


def pick_of(case: TranscriptCase, picked: str) -> str:
    """The AI's pick, word by word: did it catch the order number's words, and only them?"""
    mine, number, same = _plain(picked), _plain(case.number_words), _plain(case.same_number)
    if not mine:
        return "nothing" if case.expect not in (ASK, NONE) else "nothing (right)"
    if number and mine in (number, same):
        return "right words"
    # Inside one of the other numbers (an amount, the wrong number before a correction).
    for text, role in case.pieces:
        other = _plain(text)
        if role == OTHER and any(other[i:i + len(mine)] == mine for i in range(len(other) - len(mine) + 1)):
            return "another number"
    if case.expect == NONE:
        return "picked when there was none"
    return "other words"


def outcome_of(case: TranscriptCase, accepted: str | None, status: str) -> str:
    if case.expect in (ASK, NONE):
        return RIGHT if accepted is None else WRONG
    if accepted == case.expect:
        return RIGHT
    if accepted is not None:
        return WRONG
    return MISSED if status == NONE else ASKED


def mark(case: TranscriptCase, picked: str, run: int, key: str = "") -> Result:
    reading = read_number(picked, case.heard, LENGTH)
    outcome = outcome_of(case, reading.digits, reading.status)
    fault = ""
    if outcome != RIGHT:
        # Given the labelled words, would the reader have got it? If not, it's its own fault.
        labelled = read_number(case.number_words, case.heard, LENGTH)
        fault = "AI" if outcome_of(case, labelled.digits, labelled.status) == RIGHT else "reader"
    return Result(case.id, case.kind, run, picked, pick_of(case, picked), reading.digits, outcome, fault, key)


def decide(case: TranscriptCase, numbers: list[dict[str, str]]) -> tuple[str | None, str]:
    """Code's part in list mode: one order number is the fast path; two different ones,
    one the caller isn't sure of, or one that can't be read beside one that can, go to
    the Thinker. Returns (number, status)."""
    if any(item.get("tag") == "unsure" for item in numbers):
        return None, SENT
    readings = [read_number(item.get("words", ""), case.heard, LENGTH) for item in numbers if item.get("tag") == "order"]
    if not readings:
        return None, NONE
    found = {reading.digits for reading in readings if reading.digits}
    # An unread order number still counts as a number the caller said. Dropping it made the
    # other look like the only one: "7980 oh no, sorry seven thousand 97 shoonya" gave 7980
    # (9 Nov 2026). Across the 831 cases this cost 5 instant right numbers and removed 2 wrong.
    if len(found) > 1 or (found and not all(reading.digits for reading in readings)):
        return None, SENT
    if len(found) == 1:
        return found.pop(), "ok"
    return None, "unreadable"


def mark_list(case: TranscriptCase, numbers: list[dict[str, str]], run: int, key: str = "") -> Result:
    accepted, status = decide(case, numbers)
    outcome = SENT if status == SENT else outcome_of(case, accepted, status)
    fault = ""
    if outcome in (WRONG, ASKED, MISSED):
        labelled = read_number(case.number_words, case.heard, LENGTH)
        fault = "AI" if outcome_of(case, labelled.digits, labelled.status) == RIGHT else "reader"
    orders = [item.get("words", "") for item in numbers if item.get("tag") == "order"]
    found = bool(case.number_words) and any(_plain(words) in (_plain(case.number_words), _plain(case.same_number)) for words in orders)
    if not numbers:
        pick = "nothing (right)" if case.expect in (ASK, NONE) and not case.number_words else "nothing"
    else:
        pick = "found the number" if found else ("listed numbers, none the right one" if case.number_words else "listed numbers")
    shown = "; ".join(f"{item.get('tag')}: {item.get('words')}" for item in numbers)
    return Result(case.id, case.kind, run, shown, pick, accepted, outcome, fault, key)


def _daily_limit(error: BaseException) -> bool:
    text = str(error).lower()
    return "per day" in text or "(tpd)" in text or "(rpd)" in text


async def check(cases: list[TranscriptCase], asks: dict[str, Ask], tries: int, done: set[tuple[str, int]],
                on_result: Callable[[Result], None], log: Callable[[str], None] = print, mode: str = "pick") -> dict[str, Any]:
    """Every case `tries` times, shared out over the keys. Results go to on_result as
    they arrive; work already in `done` is skipped. Returns what happened per key."""
    by_id = {case.id: case for case in cases}
    queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    for run in range(1, tries + 1):
        for case in cases:
            if (case.id, run) not in done:
                queue.put_nowait((case.id, run))
    failures: Counter[tuple[str, int]] = Counter()
    info: dict[str, Any] = {key: {"answers": 0, "stopped": ""} for key in asks}
    given_up: list[str] = []

    async def worker(key: str, ask: Ask) -> None:
        while True:
            try:
                case_id, run = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            case = by_id[case_id]
            try:
                # A whole ask, retries included, can't hang the run.
                answer = await asyncio.wait_for(ask(LIST_RULES if mode == "list" else PICK_RULES, case.heard), TIMEOUT_S * 4)
            except Exception as exc:  # noqa: BLE001 - every failure is sorted and reported
                if _daily_limit(exc):
                    queue.put_nowait((case_id, run))  # another key takes it
                    info[key]["stopped"] = f"daily limit: {str(exc)[:120]}"
                    log(f"[{key}] daily limit reached; its remaining cases go to the other keys")
                    return
                failures[(case_id, run)] += 1
                if failures[(case_id, run)] < CASE_TRIES:
                    queue.put_nowait((case_id, run))
                else:
                    given_up.append(f"{case_id} try {run}: {type(exc).__name__}: {str(exc)[:120]}")
                continue
            info[key]["answers"] += 1
            if mode == "list":
                on_result(mark_list(case, list(answer.get("numbers", [])), run, key))
            else:
                on_result(mark(case, str(answer.get("words", "")), run, key))

    await asyncio.gather(*(worker(key, ask) for key, ask in asks.items()))
    info["left over"] = queue.qsize()
    info["given up"] = given_up
    return info


# ---- the report ---------------------------------------------------------------------------


def _speed_line(results: list[Result]) -> list[str]:
    """How long the answers took, when they were timed (older runs weren't)."""
    timed_ones = sorted(r.seconds for r in results if r.seconds >= 0)
    if not timed_ones:
        return []
    return [f"Speed: usual answer {timed_ones[len(timed_ones) // 2]:.2f} s, 9 in 10 under {timed_ones[int(len(timed_ones) * 0.9)]:.2f} s, "
            f"slowest {timed_ones[-1]:.2f} s (retries included).", ""]


def report(results: list[Result], cases: list[TranscriptCase], when: str, run_info: dict[str, Any], model: str = MODEL) -> str:
    by_id = {case.id: case for case in cases}
    total = Counter(r.outcome for r in results)
    picks = Counter(r.pick for r in results)
    n = max(1, len(results))
    lines = [f"# Order-number check ({when})", "",
             f"{len(cases)} transcripts (`lab/week1/number_transcripts.py`); {len(results)} answers. "
             f"Model `{model}`. The AI finds the words; code reads them.", "",
             *_speed_line(results),
             "## What the robot ends up with", "",
             "| Right | Wrong number accepted | Sent to the Thinker | Asked again | Missed |", "|---|---|---|---|---|",
             "| " + " | ".join(f"{total[o]} ({100 * total[o] / n:.1f}%)" for o in (RIGHT, WRONG, SENT, ASKED, MISSED)) + " |", "",
             f"Failures by whose fault: AI {sum(1 for r in results if r.fault == 'AI')}, reader {sum(1 for r in results if r.fault == 'reader')}.", "",
             "## The AI's pick, word by word", "", "| Pick | Answers |", "|---|---|"]
    lines += [f"| {pick} | {count} ({100 * count / n:.1f}%) |" for pick, count in picks.most_common()]
    tries = defaultdict(list)
    for r in results:
        tries[r.case_id].append(r.picked.strip().lower())
    twice = [answers for answers in tries.values() if len(answers) > 1]
    if twice:
        same = sum(1 for answers in twice if len(set(answers)) == 1)
        lines += ["", f"**The same answer on both tries:** {same} of {len(twice)} cases ({100 * same / len(twice):.0f}%)."]
    lines += ["", "## By kind (worst first)", "", "| Kind | Answers | Right | Wrong accepted | To the Thinker | Asked again | Missed | Right words found |",
              "|---|---|---|---|---|---|---|---|"]
    kinds = defaultdict(list)
    for r in results:
        kinds[r.kind].append(r)
    for kind, mine in sorted(kinds.items(), key=lambda item: -sum(r.outcome != RIGHT for r in item[1]) / len(item[1])):
        c = Counter(r.outcome for r in mine)
        right_words = sum(1 for r in mine if r.pick in ("right words", "found the number", "nothing (right)"))
        lines.append(f"| {kind} | {len(mine)} | {c[RIGHT]} | {c[WRONG]} | {c[SENT]} | {c[ASKED]} | {c[MISSED]} | {right_words} |")
    for title, wanted in (("Wrong numbers accepted", lambda r: r.outcome == WRONG),
                          ("Other failures, the AI's fault", lambda r: r.outcome in (ASKED, MISSED) and r.fault == "AI"),
                          ("Other failures, the reader's fault", lambda r: r.outcome in (ASKED, MISSED) and r.fault == "reader")):
        chosen = [r for r in results if wanted(r)]
        lines += ["", f"## {title} ({len(chosen)})", "", "| Case | Marked transcript | Expected | AI picked | Got |", "|---|---|---|---|---|"]
        for r in chosen[:150]:
            case = by_id[r.case_id]
            lines.append(f"| {r.case_id} | {case.marked()[:110].replace('|', '/')} | {case.expect} | {r.picked[:45] or '-'} | {r.accepted or '-'} ({r.pick}) |")
    sent = [r for r in results if r.outcome == SENT]
    lines += ["", f"## Sent to the Thinker ({len(sent)}), by kind", ""]
    lines += [f"- {kind}: {count}" for kind, count in Counter(r.kind for r in sent).most_common()]
    lines += ["", "## The run", "", "```", json.dumps(run_info, indent=1, ensure_ascii=False)[:3000], "```"]
    return "\n".join(lines) + "\n"


# ---- running it ---------------------------------------------------------------------------


def load_done(path: Path) -> list[Result]:
    if not path.exists():
        return []
    return [Result(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def timed(ask: Ask, seconds: dict[str, float]) -> Ask:
    """Records each answer's wait by its text. It goes under the pacing, so the gap
    between requests doesn't count as the model being slow."""
    async def timed_ask(rules: str, text: str) -> dict[str, Any]:
        start = time.monotonic()
        answer = await ask(rules, text)
        seconds[text] = round(time.monotonic() - start, 2)
        return answer

    return timed_ask


async def live_run(keys: dict[str, str], tries: int, label: str, log: Callable[[str], None] = print,
                   mode: str = "list", model: str = MODEL, effort: str = EFFORT,
                   kinds: set[str] | None = None, provider: str = "groq") -> tuple[Path, list[Result]]:
    """The real check, on every key at once. `keys`: .env name -> key. Answers go to
    runs/week1/number_check_<label>.jsonl as they come; rerun with the same label to
    finish what's left without asking anything twice. `kinds`: only those case kinds,
    so a prompt change can be tried on the hard cases before spending a day's tokens."""
    from common.groq import groq_client
    from common.sarvam import sarvam_client
    from lab.judge.judge import groq_judge, paced, sarvam_judge

    out_dir = ROOT / "runs" / "week1"
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_file = out_dir / f"number_check_{label}.jsonl"
    results = load_done(answers_file)
    done = {(r.case_id, r.run) for r in results}
    cases = [c for c in load_transcripts() if kinds is None or c.kind in kinds]
    tokens: Counter[str] = Counter()
    make_client = sarvam_client if provider == "sarvam" else groq_client
    clients = {name: make_client(key, TIMEOUT_S) for name, key in keys.items()}
    seconds: dict[str, float] = {}
    heard = {case.id: case.heard for case in cases}
    try:
        asks = {}
        for name, client in clients.items():
            def used(usage: dict[str, Any], name: str = name) -> None:
                tokens[name] += usage.get("total_tokens", 0)

            def retried(error: BaseException, wait: float, name: str = name) -> None:
                log(f"[{name}] retry in {wait:.0f} s: {str(error)[:100]}")

            form = LIST_FORM if mode == "list" else PICK_FORM
            if provider == "sarvam":
                asks[name] = paced(timed(sarvam_judge(client, model, on_retry=retried, on_usage=used, form=form), seconds), SARVAM_GAP_S)
            else:
                asks[name] = paced(timed(groq_judge(client, model, effort, on_retry=retried, on_usage=used, form=form), seconds), GAP_S)
        with answers_file.open("a", encoding="utf-8") as sink:
            def keep(result: Result) -> None:
                result = replace(result, seconds=seconds.get(heard[result.case_id], -1.0))
                results.append(result)
                sink.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                sink.flush()
                if len(results) % 50 == 0:
                    log(f"{len(results)} answers ({dict(tokens)} tokens)")

            log(f"{len(done)} answers already saved; {tries * len(cases) - len(done)} to ask over {len(asks)} keys")
            info = await check(cases, asks, tries, done, keep, log, mode)
    finally:
        for client in clients.values():
            await client.aclose()
    info["tokens this session"] = dict(tokens)
    info["mode"] = mode
    info["model"] = f"{model} on {provider} (reasoning {'off' if provider == 'sarvam' else effort})"
    when = datetime.now().strftime("%Y%m%d_%H%M")
    path = out_dir / f"number_check_{label}.md"
    path.write_text(report(results, cases, when, info, model), encoding="utf-8")
    return path, results
