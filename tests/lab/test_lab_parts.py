"""The test lab's smaller parts (Part 5): its settings file, the robot's
diary, the scripted caller, the business records, and the call factory with
calls starting at once."""

import asyncio

import pytest

from agents.common.diary import Diary
from common.stopwatch import MemorySink
from common.wire.protocol import PIECE_BYTES, SILENT_PIECE
from engine.storage.call_factory import CallFactory
from engine.storage.call_settings import CallSettings
from engine.tasks.records import business_tables
from lab.caller.scripted import ScriptedCaller
from lab.puppets.food import CallerTurn
from lab.settings import SETTINGS_FILE, load_settings
from packs.food.pack import FoodPack

TONE = b"\x00\x40" * (PIECE_BYTES // 2)  # loud enough to be speech


# ---- settings (L14) ----------------------------------------------------------------


def test_the_settings_file_loads_with_every_number():
    settings = load_settings()
    assert settings.calls_at_once == 5 and settings.freeze_limit_ms == 100 and settings.stop_within_s == 1.0


@pytest.mark.parametrize(("change", "problem"), [
    (lambda text: text.replace("calls_at_once: 5", "calls_at_onse: 5"), "missing ['calls_at_once'], unknown ['calls_at_onse']"),
    (lambda text: text.replace("calls_at_once: 5", "calls_at_once: five"), "calls_at_once must be a int"),
    (lambda text: text.replace("calls_at_once: 5", "calls_at_once: 5.5"), "calls_at_once must be a int"),
    (lambda text: text.replace("max_reruns: 2", "max_reruns: -1"), "must not be negative"),
])
def test_a_wrong_setting_fails_when_the_lab_starts(tmp_path, change, problem):
    path = tmp_path / "settings.yaml"
    path.write_text(change(SETTINGS_FILE.read_text(encoding="utf-8")), encoding="utf-8")
    with pytest.raises(ValueError, match=problem.replace("[", r"\[").replace("]", r"\]")):
        load_settings(path)


# ---- the diary (L5) ----------------------------------------------------------------


def test_the_diary_writes_timed_lines_of_known_kinds_only():
    sink = MemorySink()
    diary = Diary("c1", sink, clock_ns=lambda: 7)
    diary.said("Ho gaya, [amount] wapas.", filled="Ho gaya, 62 rupaye 50 paise wapas.")
    diary.tool("confirm", {"token": "T-1"}, "ok", 812.345)
    assert sink.records[0] == {"call": "c1", "kind": "said", "at_ns": 7, "text": "Ho gaya, [amount] wapas.",
                               "filled": "Ho gaya, 62 rupaye 50 paise wapas."}
    assert sink.records[1]["ms"] == 812.3
    with pytest.raises(ValueError, match="unknown diary kind"):
        diary.write("feelings", text="x")


# ---- the scripted caller -------------------------------------------------------------


class Heard:
    def __init__(self):
        self.events = []

    def __call__(self, name, fields):
        self.events.append((name, fields))


def run_caller(turns, robot_ms, total_ms, stop_at_ms=None):
    """The robot talks for robot_ms, then is quiet; the caller is driven tick by tick."""
    heard = Heard()
    caller = ScriptedCaller(turns, heard, voice=lambda text: [TONE] * 10)
    for tick in range(total_ms // 20):
        if stop_at_ms is not None and tick * 20 == stop_at_ms:
            caller.mic.stop()
        caller.mic.next_piece(tick)
        caller.heard(TONE if tick * 20 < robot_ms else None, tick * 20)
    return heard.events


def test_the_scripted_caller_speaks_after_the_robots_turn_and_logs_the_true_start_and_end():
    events = run_caller([CallerTurn("4521, Rahul Verma.", ("cancel_order", 4521))], robot_ms=1000, total_ms=3000)
    (name, line), = events
    # The robot's sound ends at 1000 ms; at 1780 ms the 0.8 s of quiet is
    # complete, and the line goes out from the next piece: exactly 0.8 s later.
    assert name == "caller.line" and (line["start_ms"], line["end_ms"]) == (1800, 2000)
    assert line["agrees"] == ["cancel_order", 4521] and line["language"] == "hinglish" and not line["cut"]


def test_a_line_cut_off_by_the_director_is_never_a_yes():
    events = run_caller([CallerTurn("Haan, kar do.", ("cancel_order", 4521))], 1000, 3000, stop_at_ms=1900)
    (_name, line), = events
    assert line["cut"] and line["agrees"] is None and line["end_ms"] == 1900


def test_a_turn_with_nothing_to_say_is_logged_and_skipped():
    events = run_caller([CallerTurn(None), CallerTurn("later")], robot_ms=1000, total_ms=3000)
    assert [name for name, _ in events] == ["caller.skipped_turn"]


def test_silence_from_the_robot_never_counts_as_its_turn():
    heard = Heard()
    caller = ScriptedCaller([CallerTurn("hello")], heard)
    for tick in range(200):
        caller.heard(SILENT_PIECE, tick * 20)  # a silent piece is not the robot talking
    assert heard.events == [] and not caller.mic.speaking


# ---- records and the factory (L7, L3) ---------------------------------------------------


def test_the_business_records_are_the_packs_own_tables():
    assert business_tables(FoodPack()) == ["customers", "orders", "order_items", "refunds", "handoffs"]


async def test_calls_starting_at_once_share_one_template_safely(tmp_path):
    """On 1 Nov 2026, five calls each rebuilding the template locked each other up for good."""
    factory = CallFactory(FoodPack(), tmp_path)
    settings = [CallSettings(call_id=f"c{n}", pack="food", mode="safe", ticket_check=True) for n in range(5)]
    paths = await asyncio.wait_for(asyncio.gather(*(factory.new_call(s) for s in settings)), 30)
    assert len({p.name for p in paths}) == 5 and all(p.exists() for p in paths)
