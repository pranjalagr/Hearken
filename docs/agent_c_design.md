# Agent C: design and build order

Agent C is Hearken's own voice agent: a Talker, a Thinker and a Guard around the
same ears, voice, tools and lab as Agent A. This file is the part of the project plan
that building Agent C needs. "Decision N" points to the plan's Part 2 decisions; the
numbers here are starting guesses, kept in a settings file, never in code.

**The rule for all of it: start simple, measure, and add a fix only when a test shows a
problem.**

## Why Agent C exists: where Agent A fails

Agent A (`agents/agent_a/`) is the plain Pipecat chain: one Flash-Lite brain with the
backend's tools. Its clean practice run on 15 Nov 2026
(`runs/lab/2026-11-15_210345_agent-a-practice-7`, not in git) scored 4 of 15, with 8
harmful actions and a 3.4 s median reply. Agent C must win these cases on the backend's
records, not on the judge alone:

| # | Moment (practice task) | What Agent A did | What Agent C does | Part |
|---|---|---|---|---|
| 1 | Caller talks while a confirm is in flight (food_03, 09, 11, 12, 13, 14, 15) | Pipecat cancelled the tool call; the cancel reached the backend, the change was not saved, and A twice told the caller "ho gaya" anyway | The confirm is sent by code and shielded by the call box; the caller's words go to the Talker (answers now) and the Thinker (next step); the outcome is spoken only from the backend's answer via `[result]` | call box + Talker + `[result]` |
| 2 | "ruko" right after a yes / while saving (food_12, food_15) | No way back | The save waits ~0.8 s before its transaction (backend already does this); the haan-ruko model pauses the ticket at once; the words decide | haan-ruko + Guard |
| 3 | "haan haan" while the robot is still asking (food_08) | Any "haan" is a yes | A yes counts only after the question's "played" receipt | Guard |
| 4 | "yes, but..." with a condition (food_09) | Confirmed / looped | The judge labels it "other"; code asks again | Guard |
| 5 | Made-up or misheard numbers (food_02, 09, 13) | Said 4535 for 4533, 512 for 4512 | The Talker writes no digit; code fills `[order]` / `[amount]` from the backend | slots |
| 6 | Going round in circles (food_05, food_06) | 7 and 4 hand-overs in a row | Code stops when the same tool, inputs and answer repeat twice; one hand-over | Thinker limits |
| 7 | Order goes out while cancelling (food_02) | Said the truth this time, by luck | `[result]` is filled from what the backend answered | `[result]` |
| 8 | Speed on a lookup turn | 3.4 s median reply | The Talker replies at once ("ek second"); listen-ahead looks up early; target ≤ 0.8 s | Talker + listen-ahead |

## The shape

```
caller ──► EARS ──► TALKER ──► voice ──► caller
             │        ▲  reads notes
             │        │
             │     NOTEBOOK ◄── code copies every backend answer
             │        ▲
             └──► THINKER ──► backend tools (lookups, prepare_*)
                      │
                   GUARD ──► yes-ticket ──► code presses confirm (shielded)
```

- **Talker** (`gemini-3.5-flash-lite`): does all the talking, presses no backend button.
  Its one tool is `ask_thinker`.
- **Thinker** (`gemini-3.5-flash-lite`, the user's choice on 15 Nov 2026: free tier,
  500 requests a day): a background task in the same program, off Pipecat's pipeline. Loop:
  decide → tool → read the result → decide again, until it presses no tool.
- **Notebook**: a plain in-memory list per call, with a bell (asyncio.Event) rung on each
  new note. No queue with a size limit, no locks (one event loop), every wait has a time
  limit, at most 200 notes per call.
- **Guard**: code plus a one-word AI judge. Only code holding the backend's `GuardDoor`
  (`engine/safety/guard_door.py`) can get a yes-ticket.

## Decisions to build (condensed)

**Jobs (decision 7).** Only the Talker opens a job, by calling `ask_thinker`. While a job
is open, code passes **every** caller turn to the Thinker, even "haan haan" (no word list).
A job closes only when the Thinker's step presses no tool **and** five code checks pass:
1. that step began after the newest caller turn (turns are numbered);
2. no caller turn arrived after it began;
3. no question or yes is waiting (the last "need info"/"need yes" note has an answer after it);
4. no change is running (a counter: up when sent, down on answer or "result unknown");
5. the final reply was fully spoken (the voice played it to the end, not cut off).
Then the job is at "answer given"; the caller's next turn still goes to the Thinker; if the
Thinker stops again and the checks pass, the job closes. Hand-over, a Thinker crash and
hang-up end a job at once. One job at a time.

**What the Talker may say (decisions 8, 10, 18).**
- Only facts from the notebook. No fact yet → "ek second, check kar raha hoon", never a guess.
- **Code writes every number.** Notes show `[order]` and `[amount]`; the Talker copies the
  slot, code fills it (`agents/agent_c/talker/slots.py` already does `[amount]`, with
  `spoken_amount` from `packs/food/money.py`). A digit the Talker types outside a slot, or a
  slot no note gave, is never spoken: a safe line plays instead.
- **Outcomes come from `[result]`**, filled by code from the backend's latest answer for that
  action: done → "cancel ho gaya"; refused → "cancel nahi hua" + the backend's reason;
  stopped by the Guard before sending → "cancel nahi kiya"; still saving / no answer → "ek
  second, check kar raha hoon". Outcome words outside the slot are measured first (the
  judge flags them), not blocked by a word list (decided 15 Nov 2026).
- When the caller speaks during a job and no new note has come, the Talker **acknowledges
  and repeats back, never promises an outcome** ("Ji, address bhi dekh leta hoon").
- Statuses are fixed phrases from `packs/food/spoken.yaml`.
- 1–2 sentences per reply, enforced by code. Temperature 1.0, thinking minimal.
  Style guide: `agents/agent_c/talker/prompts/style.md` (it still shows digits: move it to
  `[order]` slots, then rerun the Talker checks; a prompt change has hurt it before).
- The voice gets "main" → "mai" from `agents/agent_c/talker/voice_spelling.py`.
- It never speaks over the caller: a note that arrives while the caller talks waits.

**Language (decision 10, 15 Nov 2026).** A sticky call language kept by plain code: start in
Hinglish; a finished caller turn changes it only if it has at least 4 words and the ears
report another language. The slot filler and the voice read this one value.

**Who writes the yes-question (decision 9).** Code: the backend fills it at prepare time;
read it through `GuardDoor.question(token, language)`. Always ends "Haan ya nahi?".

**Who presses confirm (decision 17).** Code, the instant the Guard gives a ticket, through
`CallBox.change(...)` so it is shielded: a caller cutting in never cancels it, and the box
waits up to ~10 s for its answer, else "result unknown".

**The Guard (decisions 20–23).**
- A yes needs all of: the judge's label "yes", code checks, a finished sentence, and the
  caller's "haan" starting **after** the question's "played" receipt (the wire's `Mark` /
  `Played` messages in `common/wire/protocol.py`). An early "haan" never counts; if the
  caller then stays quiet, ask "Haan ya nahi?".
- Labels: yes / no / stop / unclear / other. The judge (Flash-Lite) answers one word,
  ~1 s limit; a timeout is "unclear", never a yes.
- A no or stop needs only one voice: the judge **or** a word list ("ruko", "nahi", "mat",
  "wait", "stop") on the partial words. No, stop and other cancel any open ticket at once
  (`GuardDoor.cancel_ticket`).
- Asking: ask 1 (full) → ask 2 ("Maaf kijiye, sunai nahi diya. Haan ya nahi?") → keypad
  ("haan ke liye 1, nahi ke liye 2"; the wire carries `Key`) → human. Only "unclear"
  counts toward this; "other" is answered and doesn't count.
- Save every yes-moment (question, words, label, timing) in the diary: training data later.

**Stop or keep talking (decision 26).** The haan-ruko model (Part 4, `lab/haan_ruko/`,
v4, line 0.30) is Agent C's only stop-or-keep check: no word list, no word count. It looks at
the last 2 s of caller sound every 0.1 s while the robot talks (one check well under 20 ms on
CPU; `lab/haan_ruko/model.py` has the ONNX runner). Above the line: stop the voice
(Pipecat interruption), bump the job version, pause any open ticket. It keeps running until
the caller finishes. The model file lives outside git (a path in the settings); tests use a
fake model.

**Versions (decision 16).** Every job has a version. Code bumps it at once when the
stop-or-keep answer turns to stop, and when the caller finishes any turn during a job. The
Talker only speaks notes from the current version; older ones wait until the Thinker says
"still good" or redoes the work.

**Limits and retries (decision 19, code not the Thinker).**

| What went wrong | What to do |
|---|---|
| Same tool, inputs and answer repeat twice in a row | stop; one hand-over |
| Backend says `refused` | no retry; tell the reason, offer a human |
| Lookup failed or timed out | retry once |
| Change got no answer | retry with the same token, up to 2 more (backend answers `already_done`) |
| `expired` | prepare again, ask again |
| Still stuck | tell the truth and hand over; never "ho gaya" when unsure |

Plus the lab's cap: at most `--max-brain-requests` (40) brain requests per call, Talker,
Thinker and judge together (`agents/common/pipecat_brain_cap.py`).

**Time limits (decision 30).** Talker first word ~1.2 s (filler + retry once); voice first
sound ~1 s; one Thinker step ~8 s (retry once, then a "failed" note → hand over); Guard
judge ~1 s; lookup tool 5 s; change tool 10 s.

**Start and end (decision 13).** The fixed greeting plays at once (same as Agent A). Silent
caller: ~6 s → "Hello, aap sun rahe hain?"; ~15 s more → "Lagta hai line kat gayi", hang up.
Hand-over: `hand_over_to_human`, say so once, end the call (the `end_call` pattern in
`agents/common/pipecat_hangup.py`).

**Waiting lines (decision 11).** Short pre-made lines in the robot's own voice ("Ek second,
dekh raha hoon"), picked by code: the first after ~1 s of silence, then one every ~4 s, never
the same twice in a row.

**End of turn (decision 25).** Namo Turn Detector v1 Hindi
(`videosdk-live/Namo-Turn-Detector-v1-Hindi`) reads the words after Silero says the caller
went quiet; it needs Hindi letters (Sarvam's partial words are in Hindi letters). Number
mode: wait longer right after asking for a number. Until it is built and tested on real ears
text, use Pipecat's default (Smart Turn), as Agent A does.

**Quick line and listen-ahead (decisions 27–29).** The quick line ("4521, ek second") plays
only when the Talker calls `ask_thinker`, and says a number only when exactly one clear order
number was found (`lab/week1/number_check.py`, Sarvam; `agents/common/number_reader.py`).
Listen-ahead: code fires a **lookup only** (never a change) once it has both the number and
the name; a guess whose number or name differs from the final words is thrown away.

## Build order

Each step leaves a working Agent C with its tests passing.

1. **Skeleton**: `agents/agent_c/robot.py` + `cli/agent_c.py`, started by the lab exactly like
   Agent A (same arguments, plus `--max-brain-requests`). Reuse `agents/common/pipecat_line.py`,
   `pipecat_diary.py`, `pipecat_hangup.py`, `pipecat_brain_cap.py`, `call_box.py`, `tools.py`.
   The Talker gets only `ask_thinker`.
2. **Notebook + Thinker**: the notebook, the Thinker loop as a crash-safe call-box task,
   code copying backend answers into notes, versions, the five job-close checks.
3. **Talker speaks only notes**: `[order]`, `[amount]`, `[result]` slots and the number check,
   voice spelling, sticky language, acknowledge-never-promise.
4. **Guard**: code plays the yes-question, played-receipt timing, the one-word judge, code
   checks, ticket via `GuardDoor`, shielded confirm, ask-again ladder.
5. **haan-ruko live**: the model on caller audio while the robot talks; stop bumps the version
   and pauses the ticket.
6. **Thinker limits**: the repeat check, the retry table, one hand-over.
7. **End of turn**: Namo + Silero + number mode (needs a real-ears test, ~₹2.60 of Sarvam).
8. **Speed**: waiting lines, silent-caller lines, quick line, listen-ahead.
9. **Lab wiring**: `agent:c` in `cli/lab_run.py` ROBOTS, safe mode **with** the ticket check;
   15 practice calls; compare with Agent A's run 7.

## Settings and keys

- One settings file, `agents/agent_c/settings.yaml`, like `agents/agent_a/settings.yaml`;
  prompts in text files next to it.
- Keys come only from `common/env.py` (`gemini_key()`, `sarvam_key()`), chosen by name with
  `HEARKEN_GEMINI_KEY` / `HEARKEN_SARVAM_KEY`; never switched automatically, never printed.
