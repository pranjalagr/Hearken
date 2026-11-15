# Agent C: night build report (15 Nov 2026)

Steps 1-6 of the build order in `docs/agent_c_design.md` are built, each in its own
commit on branch `agent-c-night`, each with its tests green. Nothing called a real
outside service: every AI and voice part was tested with fakes, and the backend parts
with the real backend on a call copy.

## What each step built

| Step | Files | Tests |
|---|---|---|
| 1 Skeleton | `agents/agent_c/robot.py`, `settings.py`, `settings.yaml`, `talker/ask_thinker.py`, `talker/prompts/role.md`, `cli/agent_c.py` | `tests/agents/test_agent_c_skeleton.py` (7) |
| 2 Notebook + Thinker | `notebook.py`, `job.py` (turns, versions, the five close checks), `desk.py` (jobs), `live_retry.py`, `thinker/` (brain, loop, copy_notes, prompt), `talker/feed.py`, `talker/speech_watch.py`; `BrainCap.add` in `agents/common/pipecat_brain_cap.py` | `test_agent_c_thinker.py` (18) |
| 3 Talker speaks only notes | `talker/note_view.py` ([order], [amount], [result] slots, ready phrases), `talker/speaker.py` (2 sentences, number check, slot filling, voice spelling), `talker/language.py` (sticky language), `talker/mouth.py` (between Talker and voice), `talker/words.yaml`; style guide moved to slots | `test_agent_c_talker.py` (10) |
| 4 Guard | `guard/guard.py` (yes-question from `GuardDoor`, mark + "played" timing, ticket, shielded confirm, asking ladder, keypad), `guard/judge.py` (one-word judge, word list), `guard/partial_words.py`, `guard/prompts/judge.md`; the CLI opens the door on its own connection | `test_agent_c_guard.py` (15, with a parametrized one) |
| 5 haan-ruko live | `stop_check.py` (ONNX runner without torch, listener every 0.1 s while the robot talks), `stop_switch.py` (the only interruption) | `test_agent_c_stop_check.py` (5) |
| 6 Thinker limits | `hand_over.py` (one hand-over, then the call ends), repeat check and lookup retry in `thinker/loop.py`, confirm retries / expired / give-up in `guard/guard.py` | `test_agent_c_limits.py` (7) |

Agent C tests: 62, all green. Full suite (`python -m pytest -q -p no:warnings`, without
`tests/lab/test_haan_ruko.py`) after each step: 1232, 1250, 1260, 1275, 1280, 1287 passed;
the only failures each time were the two known ones (`test_number_reader` hard24, hard34).

## Choices the design did not cover

1. **Setup.** `download.pytorch.org` is blocked by this machine's network policy, so
   torch is not installed and `tests/lab/test_haan_ruko.py` (needs torch) was skipped
   in every run. `pip` could not install `soundfile==0.14.0` together with
   `pipecat-ai==1.16.0` (it wants `~=0.13.1`), so 0.13.1 was used here;
   `requirements.txt` was not changed. Agent C's own code never imports torch.
2. **Commits per step.** I wrote steps 1-6 faster than the full suite runs, so each
   step's commit was rebuilt as a snapshot in a separate worktree and its full suite run
   there before committing. Each commit is a working Agent C.
3. **No `end_call` for the Talker.** The design gives the Talker only `ask_thinker`, so it
   cannot hang up after a goodbye. The call ends on a hand-over (code), the caller's
   hang-up, the brain cap or the box's limits. Ending after a goodbye is left for later.
4. **Number check = the existing one.** A number the Talker types is allowed if it is in a
   note line it was shown or in the caller's words (`slots.stray_numbers`, decision 8),
   not "any digit outside a slot". Notes never show order numbers or amounts (only slots),
   so the 4535-for-4533 error is still caught, and the caller's own number can be
   repeated back. The Talker's history keeps its replies with slots, not filled numbers.
5. **Slot names.** `[order]`, `[amount]`, and a money field's own name (`[refunded]`). A
   second different value gets `[order_2]`, `[amount_2]`, so two orders never swap.
   Statuses and payment methods appear as the pack's ready phrases.
6. **Result words** live in `agents/agent_c/talker/words.yaml` (same checked format as a
   pack's `spoken.yaml`), one phrase per action and outcome (done / refused / stopped /
   unknown), with the safe line, the asking lines and the hand-over line.
7. **"Still good"** is the Thinker writing exactly `still good`; "need info" is a line
   starting `Need:`. Plain text, no extra tools.
8. **Acknowledge, never promise** is only in the Talker's role prompt; code does not add
   a hint per turn (it would cost a second Talker request). Outcome words outside
   `[result]` are left to be measured by the judge, as decided.
9. **Notes held during a yes-question.** While the Guard's question is open the note feed
   gives the Talker nothing new, so a Thinker line is never spoken over the question.
10. **When the question has played.** The Guard sends the mark when the voice stops
    uncut after speaking the question's last words (seen in Pipecat's spoken-text
    frames); the caller's "played" receipt gives the time. A haan's start time is the
    newest caller piece when the voice detector fires, less its 0.2 s start delay
    (`guard.vad_start_s`).
11. **Early haan.** A yes that started before "played" never counts; after
    `reask_wait_s` (1.5 s, a guess) of quiet the Guard asks the short "Haan ya nahi?"
    (not counted on the ladder). "Other" asks the full question again, not counted.
12. **"Pause the ticket".** The door has no pause, only cancel, so a stop from the model
    (or a listed word) cancels the ticket being confirmed; the outcome is "stopped", and
    the Thinker can prepare again if the caller still wants it. Safe side: never a change
    the caller stopped.
13. **No stop model file -> Pipecat's own stop.** The model lives outside git
    (`runs/week1/haan_ruko/haan_ruko.onnx`). Without it Agent C stops on any caller
    sound, as Agent A does, and writes that in its diary. With it, Pipecat's turn start
    no longer interrupts; only the model does. The check runs on the event loop
    (measured under 20 ms with a stand-in model); a slow check is written as an error.
14. **Repeat check** = the same tool, inputs and answer 3 times in a row (first time plus
    two repeats), `same_call_limit` in settings. Then one hand-over.
15. **Expired** is also caught at ticket time (the door refuses an expired token, matched
    on its own message "expired"): prepare again once, ask again; again -> unknown and
    hand-over.
16. **Thinker failed or crashed** -> a "failed" note, then the one hand-over. A full
    notebook (200 notes) raises inside the Thinker, so it ends the same way.
17. **The Talker's first-word retry** (decision 30) is not wired: Pipecat's own
    `retry_on_timeout` would be a hidden retry, so it was left off.
18. **Agent C refuses to start without `confirm`** in the catalog (safe mode only): in
    plain mode the Thinker's tools could change orders with no Guard.
19. **Brain cap** counts Talker (as Agent A), Thinker steps and judge requests.

## What is left

- Steps 7-9: end of turn (Namo + number mode), speed (waiting lines, silent-caller lines,
  quick line, listen-ahead), lab wiring (`agent:c` in `cli/lab_run.py`, safe mode with the
  ticket check, 15 practice calls vs Agent A's run 7).
- Rerun the week-1 Talker checks (`flash_lite_check`, `money_check`) on the changed
  style guide (now `[order]` and `[result]` slots): not run here, they need keys.
- Nothing has run against real Gemini, Sarvam or the real model file. The Pipecat wiring
  (aggregator events, spoken-text frames, interruptions) is tested part by part, not in
  a live call.

## First things to try on real calls

1. One call, food_01, with the diary on: check the order of "notes to talker", "yes
   moment" (played_ms vs start_ms), "change result", and that the yes-question's mark
   is sent after the question (decision 10 above).
2. food_08 ("haan haan" while asking) and food_12 / food_15 ("ruko" after a yes):
   the Guard's main cases.
3. Check that Sarvam's partial words reach `PartialWords` (they must be
   `InterimTranscriptionFrame`s) and that the ears report a language code.
4. With the model file in place: does the voice stop on "ruko" and keep going on
   "haan" during long replies, and do checks stay under 20 ms on the laptop?
5. Watch the brain cap: Talker + Thinker + judge per call, against the 40 limit.
