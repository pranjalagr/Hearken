# Hearken

A Hinglish voice agent that listens carefully: it tells "haan" (go on) from
"ruko" (stop) and acts only on a real yes. (The code folder is still called
`pakka`, the project's first working name.)

What is here so far:
- **the backend** (Part 1): the pretend food business the robots talk to, with
  safe actions, breakdown switches and race tests;
- **the wire**: the one WebSocket line every caller and robot speaks, plus
  audio helpers and a stopwatch;
- **the call box**: everything one call runs, ended cleanly when the call ends;
- **the test lab's first parts**: the bell board, the director (exact-moment
  race lines and drops), the dummy robot, the caller's number words, and a web
  phone to call any robot from the browser.

The plan and every design decision are kept in a separate document, not in
this repo.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

Node.js is optional: if it is installed, `pytest` also runs the web phone's
JavaScript tests (otherwise they are skipped).

This folder is inside OneDrive. The tests keep their databases in the system
temp folder, so syncing never touches them. Call copies made by `cli.make_call`
go to `runs/` here; if OneDrive ever locks one ("database is locked"), pass
`--workdir` with a folder outside OneDrive.

## Try it

```powershell
# make a fresh call copy from a task (prints its path)
python -m cli.make_call --task packs/food/tasks/food_01_cancel_prepaid.yaml --mode safe

# serve that call's tools over MCP (stdio), for an agent to connect to
python -m cli.serve_call --db <printed path>

# measure the Part 1 numbers to optimize
python -m cli.measure_backend
```

### Call a robot from the browser

```powershell
# window 1: the echo robot plays your voice back 2 seconds later
python -m cli.dummy_robot

# window 2: the web phone page
python -m cli.phone_page
```

Open http://127.0.0.1:8080 in Chrome or Edge, press **Call**, allow the
microphone, and say something: it comes back 2 seconds later, at phone
quality (8 kHz). The keypad sends keys, and **Silent drop (test)** makes the
line go dead without a goodbye, to test a robot's "no audio for 30 s" rule.

### Week-1 checks (real AI and voice services)

These call Google and Sarvam, so they need keys in `pakka/.env` (git-ignored):

```
GEMINI_API_KEY=...
SARVAM_API_KEY=...
```

```powershell
# decision 2: 20 Talker replies for a person to judge as natural or not
python -m cli.flash_lite_check

# decision 10: does the Talker put every amount through the [amount] slot? (judged by code)
python -m cli.money_check

# decision 12: candidate voices, saved at phone quality, with a page to judge them by ear
python -m cli.voice_test --round screen
python -m cli.voice_test --lines E3,E9 --voices sarvam-sunny-enhi
```

Reports, clips and listening pages go to `runs/week1/` (not in git). The voice
test skips clips it already made, so a run stopped by a daily limit carries on
where it left off.

## The wire

One WebSocket per call. Audio goes as binary frames, each exactly one 20 ms
piece: 160 samples of 16-bit little-endian PCM at 8 kHz (320 bytes). Control
messages are one JSON object per text frame:

| From | Message | Meaning |
|---|---|---|
| caller | `start` | the call begins (first message) |
| robot | `mark` | "tell me when the audio before this has played" |
| caller | `played` | it has, at `caller_ms` |
| robot | `clear` | stop playing; throw away what is not yet played |
| caller | `cleared` | how many robot pieces were really heard, and which marks were dropped |
| caller | `key` | a keypad key |
| either | `end` | a clean hang-up |

The caller sends one piece every 20 ms for the whole call, so "20 ms x pieces
sent" is a clock both sides read exactly (`caller_ms`). The robot sends its
audio as fast as its voice service makes it; the caller's speaker keeps the
real speed. Anything off-spec is cut off with close code 1002.

## Folders

```
engine/                 knows nothing about any business
  contract/             the Action and Pack shapes every pack fills in, and spoken words
  storage/              database interface + SQLite, engine tables, per-call copies, call settings
  safety/               tokens (only once), yes-tickets (only after a real yes), the Guard's door
  lab_controls/         pretend clock, breakdown switches, timeline, normal delays, bells
  records/              the action log
  service/              backend (front desk), tool catalog, replies, MCP server
  tasks/                task files (with the caller: section): load, set up a call, check the result
packs/
  food/                 the food-delivery pack: tables, seed, rules, actions, tasks,
                        caller-fact checks, and spoken.yaml (ready phrases said out loud)
  toy/                  a tiny pack used only to prove the engine works without food
  registry.py           the only place that lists every pack
common/                 shared by robots and callers
  wire/                 the WebSocket line: messages, the caller's speaker, the beat, both ends
  audio/                PCM, resampling, phone sound, background noise, sound files
  stopwatch.py          time stamps for every step of a call
  env.py                reads the keys in .env, never printing one
agents/
  common/call_box.py    one box per call: ends cleanly, shields a change already sent
  agent_c/talker/       the Talker's style prompt, and its [amount] slots and number check
lab/
  week1/                the week-1 checks: Talker replies, amounts, voices
  bell_board.py         where backends ring their bells for the director
  caller/director/      exact-moment race lines, tricks and drop tests
  caller/mouth/         the caller's number words (kept apart from any robot code)
  dummy_robot/          the puppet robot (fixed script) and the echo robot
  web_phone/            the browser phone (its logic is tested with Node)
  synthetic_voice.py    a stand-in voice until the real ones are picked
cli/                    command-line entry points
tests/
  engine/               engine tests, run on the toy pack
  food/                 food rules, actions, races and spoken words
  end_to_end/           tasks checked by the task checker, and a real MCP server process
  common/               the wire (parts and real calls), audio, stopwatch, freeze check, .env
  agents/               the call box, the Talker's slots
  lab/                  bells, director, dummy robot, whole lab calls, number words, web phone,
                        and the week-1 checks (offline: fake networks, no keys needed)
```

Tests that run live audio also check that the event loop never freezes for
over 100 ms (Part 2 decision 14): a heartbeat ticks every 10 ms, and a
watcher thread names the code that was running if it is late.

## How one tool call works

Every tool call first waits its **normal delay**: 0.3 s for lookups and
prepares, 0.8 s for a risky action's save, plus any slow-before-save switch.
The backend itself answers in milliseconds, which no real company does. The
wait happens outside the transaction, so a "ruko" can still cancel the ticket
meanwhile. The values are saved in `call_info`, and `measure_backend` turns
them off to measure raw speed.

Then it is one write transaction:
**move the world (timeline) → check → act → write the log → commit**.

In safe mode a risky action takes two steps:

1. `prepare_<action>` checks the rules and saves a **token**. Nothing changes yet.
   With the token it saves the exact **yes-question** the caller will hear, in
   Hinglish and English, filled in from the pack's `spoken.yaml`. Agent code
   reads it through the Guard's door; it is not in the AI's tool reply.
2. `confirm(token, ticket)` checks, in this order:
   1. token already done? → "already done", even if it has expired
   2. token expired?
   3. yes-ticket valid?
   4. rules still pass?

   Only then does it act. The action, the token marked done, the ticket marked
   used and the log row are all saved together.
