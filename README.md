# Hearken

A Hinglish voice agent that listens carefully: it tells "haan" (go on) from
"ruko" (stop) and acts only on a real yes. (The code folder is still called
`pakka`, the project's first working name.)

Hinglish voice agent backend: safe actions and race tests. This folder is
**Part 1**, the pretend business the agents talk to. The plan and every design
decision live in `../hinglish_voice_agent_project.md` (one folder up).

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

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

## Folders

```
engine/                 knows nothing about any business
  contract/             the Action and Pack shapes every pack fills in
  storage/              database interface + SQLite, engine tables, per-call copies, call settings
  safety/               tokens (only once), yes-tickets (only after a real yes), the Guard's door
  lab_controls/         pretend clock, breakdown switches, timeline
  records/              the action log
  service/              backend (front desk), tool catalog, replies, MCP server
  tasks/                task files: load, set up a call, check the result
packs/
  food/                 the food-delivery pack: tables, seed, rules, actions, tasks
  toy/                  a tiny pack used only to prove the engine works without food
  registry.py           the only place that lists every pack
cli/                    command-line entry points (wires engine + packs together)
tests/
  engine/               engine tests, run on the toy pack
  food/                 food rules, actions and races
  end_to_end/           tasks checked by the task checker, and a real MCP server process
```

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
2. `confirm(token, ticket)` checks, in this order:
   1. token already done? → "already done", even if it has expired
   2. token expired?
   3. yes-ticket valid?
   4. rules still pass?

   Only then does it act. The action, the token marked done, the ticket marked
   used and the log row are all saved together.
