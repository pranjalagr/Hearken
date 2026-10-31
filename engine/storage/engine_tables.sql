-- Tables every call's database gets, whatever the pack.
-- These must never mention orders, food or any other business.
-- All times are pretend-clock seconds since the call started.
-- Rows are never deleted: tokens and tickets change state instead, so the
-- black box keeps the whole story.

-- How this call is set up. Exactly one row.
CREATE TABLE call_info (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    call_id         TEXT NOT NULL,
    task_id         TEXT,
    pack            TEXT NOT NULL,
    mode            TEXT NOT NULL CHECK (mode IN ('plain', 'safe')),
    ticket_check    INTEGER NOT NULL CHECK (ticket_check IN (0, 1)),
    -- Real-world start time. Only the live clock uses it, so that every
    -- process working on this call agrees on "now".
    started_at_wall REAL NOT NULL,
    -- Decision 8: plain mode has no confirm step, so there is nothing to check a ticket at.
    CHECK (NOT (mode = 'plain' AND ticket_check = 1))
) STRICT;

-- Tokens: one prepared risky action each. waiting -> done, or it expires.
CREATE TABLE prepared_actions (
    token        TEXT PRIMARY KEY,
    action       TEXT NOT NULL,
    inputs_json  TEXT NOT NULL,
    -- The same action on the same thing gets the same key (decision 14).
    request_key  TEXT NOT NULL,
    summary      TEXT NOT NULL,           -- what the agent reads to the caller
    state        TEXT NOT NULL CHECK (state IN ('waiting', 'done')),
    created_at   REAL NOT NULL,
    expires_at   REAL NOT NULL,
    done_at      REAL,
    reply_json   TEXT                     -- saved reply, sent again on "already done"
) STRICT;
CREATE INDEX prepared_actions_by_request ON prepared_actions (request_key);

-- Yes-tickets, written only by the Guard's private door (not an AI tool).
CREATE TABLE tickets (
    ticket       TEXT PRIMARY KEY,
    token        TEXT NOT NULL REFERENCES prepared_actions (token),
    -- A copy of the exact action the caller said yes to.
    action       TEXT NOT NULL,
    inputs_json  TEXT NOT NULL,
    state        TEXT NOT NULL CHECK (state IN ('open', 'used', 'cancelled')),
    issued_at    REAL NOT NULL,
    expires_at   REAL NOT NULL,
    closed_at    REAL,                    -- when it was used or cancelled
    note         TEXT NOT NULL DEFAULT '' -- e.g. why it was cancelled
) STRICT;

-- Breakdown switches, one per business action (not per tool name), so the
-- same task breaks the same way in plain mode and in safe mode.
CREATE TABLE breakdown_switches (
    action   TEXT PRIMARY KEY,
    kind     TEXT NOT NULL CHECK (kind IN ('fail', 'slow', 'reply_lost')),
    seconds  REAL NOT NULL DEFAULT 0 CHECK (seconds >= 0),
    -- Only for 'slow': wait before the save (can still be stopped) or after it (too late).
    delay_at TEXT CHECK (delay_at IN ('before_save', 'after_save')),
    times    INTEGER NOT NULL DEFAULT 1 CHECK (times >= 1),  -- how many saves it breaks
    used     INTEGER NOT NULL DEFAULT 0,
    CHECK ((kind = 'slow') = (delay_at IS NOT NULL))
) STRICT;

-- Planned changes that happen on their own (the pretend kitchen and rider).
CREATE TABLE timeline (
    id           INTEGER PRIMARY KEY,
    target       TEXT NOT NULL,           -- what changes, in the pack's own words
    change       TEXT NOT NULL,           -- the change, in the pack's own words
    at_second    REAL,                    -- trigger by time ...
    after_event  TEXT,                    -- ... or right after an event
    state        TEXT NOT NULL DEFAULT 'planned' CHECK (state IN ('planned', 'applied', 'skipped')),
    done_at      REAL,
    note         TEXT NOT NULL DEFAULT '', -- why it was skipped
    CHECK ((at_second IS NULL) <> (after_event IS NULL))  -- exactly one trigger
) STRICT;

-- One row per tool call. Written in the same transaction as the action itself.
CREATE TABLE action_log (
    id           INTEGER PRIMARY KEY,
    at           REAL NOT NULL,
    tool         TEXT NOT NULL,           -- what the AI called
    action       TEXT,                    -- the business action behind it
    inputs_json  TEXT NOT NULL,
    outcome      TEXT NOT NULL CHECK (outcome IN ('ok', 'refused', 'already_done', 'expired', 'error')),
    -- 1 only when the action's change really happened. A successful prepare is
    -- 'ok' too but changes nothing, so the scorer counts actions by this, not by outcome.
    acted        INTEGER NOT NULL DEFAULT 0 CHECK (acted IN (0, 1)),
    reason       TEXT NOT NULL DEFAULT '',
    token        TEXT,
    ticket       TEXT,
    switch_note  TEXT NOT NULL DEFAULT '' -- what a breakdown switch did, if anything
) STRICT;
