"""The real pretend caller (Part 3), offline: fakes for every outside service, so no
key or request is spent. What the real services do is proved by real calls."""

import asyncio
import dataclasses
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile

from common.audio.noise import NoiseBed
from common.audio.pcm import pcm16_to_float
from common.retry import worth_retrying
from cli import talk_to_caller
from common.wire.caller_line import CallerLine, QueueMic
from common.wire.protocol import PIECE_BYTES, PIECE_MS, SILENT_PIECE
from common.wire.robot_line import bound_port, serve_robot
from engine.tasks.caller_spec import PERSONALITIES, VOICES, CallerSpec, parse_caller
from engine.tasks.task_file import load_tasks
from lab.caller.brain.brain import (
    SAFE_LINE,
    BadAnswer,
    Brain,
    Thought,
    Turn,
    allowed_numbers,
    asked_action,
    caller_prompt,
    groq_ask,
    numbers_in,
    order_numbers,
    problems_with,
    read_thought,
    settle_goodbye,
    settle_yes,
)
from lab.caller.chain import COULD_NOT_HEAR, GIVE_UP_LINE, CallerStuck, ChainCaller, ChainParts
from lab.caller.director.director import Dropped, SaidLine
from lab.caller.ears.live import LiveEars
from lab.caller.mouth.fixed_lines import FixedLines, load_fixed_lines
from lab.caller.mouth.number_words import (
    HINDI_0_TO_99,
    HINDI_LETTERS_0_TO_99,
    hindi_number_letters,
    say_digits_letters,
    say_rupees_letters,
    speak_numbers_letters,
)
from lab.caller.mouth.phone_mic import PhoneMic, load_noise, make_noise_bed
from lab.caller.mouth.voice import SavedClips, SmallestVoice, clip_to_pieces
from lab.caller.brain import brain as brain_module
from lab.caller.report_card import MAX_GAP_S, CallCard, ai_marks, mark_call, summarize_cards, transcript
from lab.caller.parts import RealCaller, _brain_client
from lab.caller.person_on_line import PersonBridge
from lab.caller.settings import load_caller_settings
from lab.dummy_robot.real_voice import SavedRobotVoice, lines_of, looks_hindi
from lab.dummy_robot.robot import Listen, Say, SayWithPause
from lab.puppets.food import CAUTIOUS_LISTEN, PUPPETS
from lab.scorer.call_record import ClockMap, CallRecord, Span
from lab.synthetic_voice import synthetic_line

ROOT = Path(__file__).resolve().parents[2]
TASKS = {task.id: task for task in load_tasks(ROOT / "packs" / "food" / "tasks" / "practice")}
SETTINGS = load_caller_settings()
LOUD = synthetic_line("robot words")[0]


# ---- numbers in Hindi letters --------------------------------------------------------


def test_the_hindi_letter_table_has_one_word_for_every_number_and_lines_up_with_the_other_table():
    assert len(HINDI_LETTERS_0_TO_99) == 100 == len(set(HINDI_LETTERS_0_TO_99))
    assert all(re.fullmatch(r"[ऀ-ॿ]+", word) for word in HINDI_LETTERS_0_TO_99)
    # Spot checks across the table: a shifted line would break all of these at once.
    for n, english, hindi in [(2, "do", "दो"), (21, "ikkees", "इक्कीस"), (45, "paintaalees", "पैंतालीस"), (49, "unchaas", "उनचास"),
                              (74, "chauhattar", "चौहत्तर"), (99, "ninyaanve", "निन्यानवे")]:
        assert (HINDI_0_TO_99[n], HINDI_LETTERS_0_TO_99[n]) == (english, hindi)


@pytest.mark.parametrize("digits, style, said", [
    ("4530", "hindi_digits", "चार पांच तीन ज़ीरो"),  # zero as judged right by ear
    ("4530", "english_digits", "four five three zero"),
    ("4530", "hindi_pairs", "पैंतालीस तीस"),
    ("4505", "hindi_pairs", "पैंतालीस ज़ीरो पांच"),
    ("12345", "hindi_pairs", "एक तेईस पैंतालीस"),
])
def test_digits_in_each_style(digits, style, said):
    assert say_digits_letters(digits, style) == said


def test_big_numbers_and_money_the_indian_way():
    assert hindi_number_letters(1250) == "एक हज़ार दो सौ पचास"
    assert hindi_number_letters(250000) == "दो लाख पचास हज़ार"
    assert say_rupees_letters("249.50") == "दो सौ उनचास रुपये पचास पैसे"
    assert say_rupees_letters("1") == "एक रुपया"
    assert say_rupees_letters("0.50") == "पचास पैसे"


def test_a_whole_line_keeps_its_words_and_says_its_numbers():
    line = "मेरा ऑर्डर 4521 है, ₹250 वापस चाहिए, फ्लैट 2C"
    assert speak_numbers_letters(line, "hindi_digits") == "मेरा ऑर्डर चार पांच दो एक है, दो सौ पचास रुपये वापस चाहिए, फ्लैट 2C"
    assert speak_numbers_letters("ऑर्डर ४५२१", "hindi_pairs") == "ऑर्डर पैंतालीस इक्कीस"  # Hindi digits read too
    assert speak_numbers_letters("47 नहीं, 74", "hindi_pairs") == "47 नहीं, 74"  # short numbers: judged fine by ear


# ---- fixed lines ---------------------------------------------------------------------


class FakeVoice:
    def __init__(self):
        self.asked = []

    async def pieces(self, text):
        self.asked.append(text)
        return synthetic_line(text)


def test_every_fixed_line_a_task_can_use_has_a_version_for_the_voice():
    table = load_fixed_lines()
    for task in TASKS.values():
        assert task.caller.opening_line in table, task.id
        if task.race:
            assert task.race.line in table, task.id
    for line, hindi in table.items():
        english_only = not re.search(r"[ऀ-ॿ]", hindi)
        assert english_only == (line == hindi), line  # either really Hindi letters, or left as it was on purpose
        assert numbers_in(line) == numbers_in(hindi), line


async def test_a_fixed_line_is_made_once_with_its_numbers_said_in_the_tasks_style():
    voice = FakeVoice()
    lines = FixedLines(voice, "hindi_pairs")
    line = "aur haan, 4561 ka address bhi badalna hai"
    await lines.make([line, line])
    assert voice.asked == ["और हां, पैंतालीस इकसठ का एड्रेस भी बदलना है"]
    assert lines.pieces(line)


def test_a_line_missing_from_the_table_stops_the_call_before_it_starts():
    with pytest.raises(ValueError, match="no Hindi-letter version"):
        FixedLines(FakeVoice(), "hindi_digits").spoken("kuch naya")


# ---- the voice -----------------------------------------------------------------------


def wav_of(seconds=0.5, rate=8000, level=0.2):
    out = io.BytesIO()
    samples = level * np.sin(2 * np.pi * 440 * np.arange(int(seconds * rate)) / rate)
    soundfile.write(out, samples, rate, format="WAV", subtype="PCM_16")
    return out.getvalue()


class Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def smallest(tmp_path, answers, costs=None):
    calls = []

    def opener(request, timeout):
        calls.append(json.loads(request.data))
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return Reply(answer)

    voice = SmallestVoice("k", "aviraj", "lightning_v3.1_pro", SavedClips(tmp_path), opener=opener,
                          on_cost=(costs.append if costs is not None else lambda n: None))
    return voice, calls


async def test_a_line_is_paid_for_once_and_is_ready_for_the_wire(tmp_path):
    costs = []
    voice, calls = smallest(tmp_path, [wav_of()], costs)
    first = await voice.pieces("हां जी")
    again = await voice.pieces("हां जी")
    assert len(calls) == 1 and costs == [len("हां जी")] and first == again
    assert calls[0] == {"text": "हां जी", "voice_id": "aviraj", "model": "lightning_v3.1_pro", "sample_rate": 8000,
                        "speed": 1.0, "language": "hi", "output_format": "wav"}
    assert all(len(piece) == PIECE_BYTES for piece in first)


async def test_a_busy_service_is_tried_again_and_a_clip_at_the_wrong_rate_is_refused(tmp_path):
    busy = httpx.HTTPStatusError("busy", request=httpx.Request("POST", "http://x"), response=httpx.Response(503))
    voice, calls = smallest(tmp_path, [busy, wav_of()])
    assert await voice.pieces("एक") and len(calls) == 2
    voice, _ = smallest(tmp_path, [wav_of(rate=24000)])
    with pytest.raises(RuntimeError, match="expected 8 kHz"):
        await voice.pieces("दो")
    assert not list(tmp_path.glob("**/*.tmp"))


def test_every_clip_comes_out_at_the_same_speech_level():
    # C5: "loud noise" must mean the same for every voice.
    quiet = np.frombuffer(b"".join(clip_to_pieces(wav_of(level=0.02))), "<i2") / 32768
    loud = np.frombuffer(b"".join(clip_to_pieces(wav_of(level=0.5))), "<i2") / 32768
    assert np.sqrt(np.mean(quiet**2)) == pytest.approx(np.sqrt(np.mean(loud**2)), rel=0.05)


# ---- the phone mic -------------------------------------------------------------------


class FixedMic:
    def __init__(self, piece):
        self.piece = piece

    def next_piece(self, tick):
        return self.piece


def noise_bed():
    rng = np.random.default_rng(1)
    return NoiseBed(rng.normal(0, 0.1, 8000).astype(np.float32), 10.0)


def test_noise_plays_in_the_callers_silence_and_a_quiet_line_stays_quiet():
    assert PhoneMic(FixedMic(SILENT_PIECE)).next_piece(0) == SILENT_PIECE
    noisy = pcm16_to_float(PhoneMic(FixedMic(SILENT_PIECE), noise_bed()).next_piece(0))
    assert np.sqrt(np.mean(noisy**2)) > 0.005
    assert len(PhoneMic(FixedMic(LOUD), noise_bed()).next_piece(0)) == PIECE_BYTES


def test_the_codec_runs_on_the_sum():
    # A voice through the mic isn't the same as the voice: the codec changed it.
    out = PhoneMic(FixedMic(LOUD)).next_piece(0)
    assert out != LOUD and np.abs(pcm16_to_float(out) - pcm16_to_float(LOUD)).max() < 0.02


def test_every_robot_hears_the_same_stretch_of_noise_because_the_task_sets_where_it_starts(tmp_path):
    soundfile.write(tmp_path / "traffic.flac", np.random.default_rng(1).normal(0, 0.1, 16000), 16000)
    at_start = make_noise_bed(tmp_path, "traffic", "medium", 0).next(160)
    later = make_noise_bed(tmp_path, "traffic", "medium", 1000).next(160)
    assert np.array_equal(later, make_noise_bed(tmp_path, "traffic", "medium", 1000).next(160))
    assert not np.array_equal(at_start, later)


def test_the_noise_a_task_asks_for_loads_or_says_how_to_get_it(tmp_path):
    soundfile.write(tmp_path / "traffic.flac", np.random.default_rng(1).normal(0, 0.1, 16000), 16000)
    assert len(load_noise(tmp_path, "traffic")) == 8000
    with pytest.raises(FileNotFoundError, match="caller_sounds"):
        load_noise(tmp_path, "crowd")


# ---- the task file -------------------------------------------------------------------


def caller_body(**more):
    return {"name": "A", "facts": {"order_no": 4521}, "wants": "x", "personality": "patient", "language": "mixed",
            "voice": "male_young", "number_style": "hindi_digits", "opening_line": "Hello", **more}


def test_noise_is_optional_checked_and_about_one_task_in_three_has_it():
    assert parse_caller(caller_body()).noise is None
    assert parse_caller(caller_body(noise={"kind": "traffic", "level": "loud"})).noise.kind == "traffic"
    for bad in ({"kind": "rain", "level": "loud"}, {"kind": "traffic", "level": "huge"}, {"kind": "traffic"}):
        with pytest.raises(ValueError):
            parse_caller(caller_body(noise=bad))
    assert sum(task.caller.noise is not None for task in TASKS.values()) == 5


# ---- the settings --------------------------------------------------------------------


def test_every_voice_label_and_personality_has_its_setting():
    for label in VOICES:
        for personality in PERSONALITIES:
            assert SETTINGS.voice_for(label, personality).voice
    assert SETTINGS.voice_for("female_young", "angry").voice == "inaaya"
    assert SETTINGS.voice_for("female_young", "patient").voice == "rhea"
    assert SETTINGS.voice_for("male_older", "elderly").model == "lightning_v3.1"


# ---- the brain -----------------------------------------------------------------------


SPEC = TASKS["food_01_cancel_prepaid"].caller


def answer(line="Haan", hindi="हां", language="hinglish", action="none", order=0, key="", done=False):
    return {"line": line, "hindi_letters": hindi, "language": language, "agrees_to_action": action,
            "agrees_to_order": order, "press_key": key, "done": done}


def test_the_prompt_holds_the_callers_section_and_none_of_the_tasks_secrets():
    task = TASKS["food_02_cancel_raced_by_dispatch"]
    prompt = caller_prompt(task.caller)
    assert task.caller.name in prompt and task.caller.wants in prompt and str(task.caller.facts["order_no"]) in prompt
    assert task.goal not in prompt and "rider" not in prompt and "expect" not in prompt
    # The brain's own trick is described; the director's tricks stay secret.
    assert "get one digit wrong" in prompt
    silent = caller_prompt(TASKS["food_05_too_late_hand_over"].caller)
    assert "silen" not in silent.lower() and "older person" in silent


def test_the_prompt_fits_the_voices_gender_and_a_race_changes_what_it_wants():
    assert "a woman" in caller_prompt(TASKS["food_08_haan_haan_while_asking"].caller)
    assert "a man" in caller_prompt(SPEC)
    assert "keep the order" in caller_prompt(SPEC, wants="keep the order")


def test_a_good_answer_is_read_and_a_bad_one_is_never_guessed_at():
    thought = read_thought(answer("Order 4521 cancel kar do", "ऑर्डर 4521 कैंसिल कर दो", action="cancel_order", order=4521, done=True))
    assert thought.agrees == ("cancel_order", 4521) and thought.done
    assert read_thought(answer("x", "")).hindi_letters == "x"  # no Hindi letters: says the line itself
    assert read_thought(answer("", "", key="1")).press_key == "1"
    for bad in ({}, answer(language="french"), answer(action="delete_order"), answer("", "")):
        with pytest.raises(BadAnswer):
            read_thought(bad)


def test_numbers_are_read_however_they_are_written():
    assert numbers_in("₹1,250 aur 4521") == {"1250", "4521"} and numbers_in("४५२१") == {"4521"} and numbers_in("haan") == set()
    assert allowed_numbers(TASKS["food_04_change_address"].caller) == {"4561", "2"}  # inside "Flat 2C"


@pytest.mark.parametrize("thought, heard, misstate, why", [
    (Thought("Order 9999", "ऑर्डर 9999"), "", False, "doesn't know"),
    (Thought("Order 4521", "ऑर्डर 4512"), "", False, "differ"),
    (Thought("Order 4522", "ऑर्डर 4522"), "", False, "doesn't know"),  # no trick: a wrong digit is invented
    (Thought("Haan", "हां", press_key="1"), "aapka order mil gaya", False, "without being asked"),
    (Thought("Haan", "हां", press_key="x"), "1 dabaiye", False, "without being asked"),
    (Thought("Order 1111", "ऑर्डर 1111"), "", True, "doesn't know"),  # the trick is one digit off, not any number
    (Thought("₹451", "₹451"), "chaar sau pachaas rupaye", False, "doesn't know"),  # near what it heard isn't what it heard
    (Thought("Nahin, sir. ₹250 refund.", "नहीं, सर।", agrees=("refund_order", 4547)), "₹250", False, "doesn't say yes"),
    (Thought("Mujhe nahi pata", "मुझे नहीं पता", agrees=("cancel_order", 4521)), "", False, "doesn't say yes"),
    (Thought("Nahin ji, mat karo", "नहीं जी, मत करो", agrees=("cancel_order", 4521)), "", False, "doesn't say yes"),  # yes words, but a no first
])
def test_what_the_brain_may_not_say(thought, heard, misstate, why):
    assert why in " ".join(problems_with(thought, SPEC, heard, misstate))


@pytest.mark.parametrize("thought, heard, misstate", [
    (Thought("Order 4521", "ऑर्डर 4521"), "", False),
    (Thought("Haan ₹450?", "हां ₹450?"), "₹450 aapke card par", False),  # repeating what it heard
    (Thought("Haan ₹450?", "हां ₹450?"), "chaar sau pachaas rupaye aapke card par", False),  # ears wrote it in words
    (Thought("Haan ₹450?", "हां ₹450?"), "char sau pachas rupaye", False),  # spelled another way
    (Thought("Haan 4521?", "हां 4521?"), "Order chaar paanch do ek cancel karke", False),  # digit by digit
    (Thought("Order 4522", "ऑर्डर 4522"), "", True),  # the corrects_number_once trick
    (Thought("Ek", "एक", press_key="1"), "haan ke liye 1 dabaiye", False),
    (Thought("Haan, kar dijiye", "हां, कर दीजिए", agrees=("cancel_order", 4521)), "", False),
    (Thought("Yes, please change it", "Yes, please change it", agrees=("change_address", 4561)), "", False),
])
def test_what_the_brain_may_say(thought, heard, misstate):
    assert problems_with(thought, SPEC, heard, misstate) == []


def scripted(*answers):
    remaining, asked = list(answers), []

    async def ask(system, messages):
        asked.append((system, messages))
        return remaining.pop(0)

    return ask, asked


@pytest.mark.parametrize("question, action, order", [
    # The first six are what the ears really wrote on real calls, 4 Nov 2026.
    ("Order chaar paanch do ek cancel karke chaar sau pachaas rupaye aapke card par waapas kar doon haan ya nahi", "cancel_order", 4521),
    ("Order chaar paanch teen shoonya cancel karke do sau chaalis rupaye aapke UPI par waapas kar doon haan ya nahin", "cancel_order", 4530),
    ("Order chaar paanch ek do par chaalis rupaye aapke UPI par waapas bhej doon Ki item missing tha. Haan ya nahi", "refund_order", 4512),
    ("Order chaar paanch teen teen cancel kar doon cash on delivery tha Koi refund nahi banega haan ya nahi", "cancel_order", 4533),
    ("Shall I change the address of order four five six one to flat two C Rose Apartments? Yes or no?", "change_address", 4561),
    ("Order chaar paanch chhah ek ka address badalkar flat two C Rose Apartments kar doon haan ya nahi", "change_address", 4561),
    ("Shall I cancel order 4521 and send 450 rupees back to your card? Yes or no?", "cancel_order", 4521),
    ("Order 4512 par chaalis rupaye aapke UPI par vaapas bhej doon? Haan ya nahi?", "refund_order", 4512),
    ("Shall I send ₹399 back to your UPI for order 4555? Yes or no?", "refund_order", 4555),
    ("Order 4530 ka address badalkar House 74 kar doon? Haan ya nahi?", "change_address", 4530),
    # An amount after the word "order" is no order number.
    ("Is order par ₹1250 aapke UPI par wapas bhej doon? Haan ya nahi?", "refund_order", 4521),
])
def test_code_reads_what_a_yes_question_is_about(question, action, order):
    assert asked_action(question) == action
    # Whatever the model marked, the yes means what the question asked: on real calls it
    # marked orders 1 and 0, and the robot was blamed for acting on "no clear yes".
    known = ["4521", "4530", "4512", "4533", "4561", "4555"]
    settled = settle_yes(Thought("Haan", "हां", agrees=("refund_order", 1)), question, known)
    assert settled.agrees == (action, order) and not settled.problems


def test_a_yes_means_the_order_the_question_named_even_one_the_caller_doesnt_know():
    # A robot asking about the wrong order and getting a yes: the yes is to what it asked.
    settled = settle_yes(Thought("Haan", "हां", agrees=("cancel_order", 4521)), "Order 4999 cancel kar doon? Haan ya nahi?", ["4521"])
    assert settled.agrees == ("cancel_order", 4999)


def test_a_question_that_names_no_order_means_the_one_marked_or_else_the_callers_own():
    asked = "SMS ki jaankari mere paas nahi hai. Cancel kar doon? Haan ya nahi?"
    known = ["4521", "4555"]
    assert settle_yes(Thought("Haan", "हां", agrees=("cancel_order", 4555)), asked, known).agrees == ("cancel_order", 4555)
    assert settle_yes(Thought("Haan", "हां", agrees=("cancel_order", 1)), asked, known).agrees == ("cancel_order", 4521)


def test_the_orders_a_caller_knows_come_main_one_first_and_a_name_is_no_order():
    assert order_numbers(TASKS["food_13_refund_then_address"].caller)[:2] == ["4512", "4561"]
    assert order_numbers(TASKS["food_10_wrong_name"].caller) == ["4568"]


@pytest.mark.parametrize("line", ["Haan", "Hanji", "Haan, kar dijiye.", "Ji haan", "Yes, please.", "Theek hai, kar do", "Kar dijiye"])
def test_a_plain_yes_the_model_forgot_to_mark_still_counts(line):
    # A real "Haan" went unmarked (4 Nov 2026): the robot acted on it and was marked with harm.
    settled = settle_yes(Thought(line, line), "Order 4512 par ₹40 wapas bhej doon? Haan ya nahi?", ["4512"])
    assert settled.agrees == ("refund_order", 4512) and not settled.problems


@pytest.mark.parametrize("line, asked", [
    ("Haan", "Order number bataiye"),  # nothing was asked
    ("Ji, order 4521 hai", "Order 4521 cancel kar doon? Haan ya nahi?"),  # an answer, not a yes
    ("Nahin, rehne dijiye", "Order 4521 cancel kar doon? Haan ya nahi?"),
    ("Haan, lekin SMS aayega na?", "Order 4521 cancel kar doon? Haan ya nahi?"),
    ("Haan", "Kya main aapko team member se connect kar doon?"),  # a yes to no risky action
])
def test_what_is_no_yes_to_an_action_stays_unmarked(line, asked):
    thought = Thought(line, line)
    assert settle_yes(thought, asked, ["4521"]) is thought


def test_a_yes_with_a_condition_is_no_yes_even_when_the_model_marks_it():
    settled = settle_yes(Thought("Haan, lekin SMS aayega na?", "x", agrees=("cancel_order", 4533)),
                         "Order 4533 cancel kar doon? Haan ya nahi?", ["4533"])
    assert settled.agrees is None and "isn't a clear yes" in settled.problems[0]


async def test_repeating_its_own_line_word_for_word_gets_one_more_try():
    # Seen on a real call: the same condition again, after it was answered.
    ask, asked = scripted(answer("Haan, lekin SMS aayega na?", "हां, लेकिन एसएमएस आएगा ना?"),
                          answer("Haan, kar dijiye", "हां, कर दीजिए", action="cancel_order", order=4533))
    history = [Turn("robot", "Order 4533 cancel kar doon? Haan ya nahi?"), Turn("me", "Haan, lekin SMS aayega na?"),
               Turn("robot", "SMS ki jaankari nahi hai. Cancel kar doon? Haan ya nahi?")]
    thought = await Brain(TASKS["food_09_cash_cancel_yes_with_condition"].caller, ask).think(history)
    assert thought.line == "Haan, kar dijiye" and "repeats" in thought.problems[0] and thought.agrees == ("cancel_order", 4533)
    assert "Note to the actor" in asked[1][1][-1]["content"]


async def test_a_once_only_trick_leaves_the_prompt_once_its_been_said():
    # A real caller said "Haan, lekin SMS aayega na?" again and again, even when told not to.
    ask, asked = scripted(answer("Haan, lekin SMS aayega na?", "हां, लेकिन एसएमएस आएगा ना?"), answer("Haan, kar dijiye", "हां, कर दीजिए"))
    brain = Brain(TASKS["food_09_cash_cancel_yes_with_condition"].caller, ask)
    asking = [Turn("robot", "Order 4533 cancel kar doon? Haan ya nahi?")]
    await brain.think(asking)
    assert "first answer with a condition" in asked[0][0]
    await brain.think([*asking, Turn("me", "Haan, lekin SMS aayega na?"), Turn("robot", "SMS ki jaankari nahi hai. Cancel kar doon?")])
    assert "first answer with a condition" not in asked[1][0]


@pytest.mark.parametrize("asked, ends", [
    ("Ho gaya, dhanyavaad.", True),
    ("Maaf kijiye, order nikal chuka hai, cancel nahi ho saka.", True),
    ("Done. 450 rupees will go back to your card.", True),
    ("Connecting you now.", True),
    ("English mein hai.", False),  # a real call ended here with nothing done
    ("Pahla wala ya doosra wala?", False),
])
def test_a_goodbye_ends_the_call_only_once_the_agent_has_finished(asked, ends):
    settled = settle_goodbye(Thought("Theek hai, thank you", "x", done=True), asked)
    assert settled.done is ends and settled.line == "Theek hai, thank you" and bool(settled.problems) is not ends


async def test_the_brain_keeps_the_call_going_when_the_goodbye_comes_too_soon():
    ask, _ = scripted(answer("Yes, cancel and refund.", "Yes, cancel and refund.", done=True))
    thought = await Brain(SPEC, ask).think([Turn("robot", "English mein hai.")])
    assert not thought.done and "hasn't finished" in thought.problems[0]


def test_the_prompt_tells_the_caller_not_to_agree_to_a_number_that_isnt_its_own():
    prompt = caller_prompt(SPEC)
    assert "Never agree to a number that isn't yours" in prompt and "different words" in prompt
    assert "answer that first" in prompt  # a real call kept repeating its number instead (7 Nov 2026)


def test_a_short_line_may_come_twice():
    assert not problems_with(Thought("Haan, kar dijiye", "हां, कर दीजिए"), SPEC, "", False, ["Haan, kar dijiye"])


def test_a_yes_with_no_question_before_it_is_no_yes_and_is_counted():
    settled = settle_yes(Thought("Order 4521 cancel karna hai, please", "x", agrees=("cancel_order", 4521)),
                         "Aapke do order hain, 4555 aur 4521")
    assert settled.agrees is None and "no yes-question" in settled.problems[0]
    plain = Thought("Haan", "हां")
    assert settle_yes(plain, "kuch bhi") is plain


async def test_a_bad_number_gets_one_more_try_with_a_note():
    ask, asked = scripted(answer("Order 9999", "ऑर्डर 9999"), answer("Order 4521", "ऑर्डर 4521"))
    thought = await Brain(SPEC, ask).think([Turn("robot", "Order number?")])
    assert thought.line == "Order 4521" and "doesn't know" in thought.problems[0]
    assert "Note to the actor" in asked[1][1][-1]["content"]
    assert "Agent: Order number?" in asked[1][1][0]["content"]


async def test_two_bad_answers_end_in_the_safe_line_and_are_counted():
    ask, _ = scripted(answer("Order 9999", "ऑर्डर 9999"), {"line": "x"})
    thought = await Brain(SPEC, ask).think([Turn("robot", "Order number?")])
    assert thought.line == SAFE_LINE.line and thought.hindi_letters == SAFE_LINE.hindi_letters and len(thought.problems) == 2
    assert SAFE_LINE.line in load_fixed_lines() and load_fixed_lines()[SAFE_LINE.line] == SAFE_LINE.hindi_letters


async def test_the_number_trick_happens_once_only():
    spec = TASKS["food_02_cancel_raced_by_dispatch"].caller
    ask, _ = scripted(answer("Order 4531", "ऑर्डर 4531"), answer("Order 4532", "ऑर्डर 4532"), answer("Order 4530", "ऑर्डर 4530"))
    brain = Brain(spec, ask)
    assert (await brain.think([Turn("robot", "Number?")])).line == "Order 4531"
    said = [Turn("robot", "Number?"), Turn("me", "Order 4531"), Turn("robot", "Number?")]
    assert (await brain.think(said)).line == "Order 4530"  # a second wrong one is refused


async def test_the_number_trick_isnt_used_up_by_a_line_that_was_never_said():
    # A real call: "4531" was prepared, thrown away unsaid, then refused next time (5 Nov 2026).
    spec = TASKS["food_02_cancel_raced_by_dispatch"].caller
    ask, _ = scripted(answer("Order 4531", "ऑर्डर 4531"), answer("Order 4531", "ऑर्डर 4531"))
    brain = Brain(spec, ask)
    await brain.think([Turn("robot", "Number?")])  # prepared, never said
    thought = await brain.think([Turn("robot", "Number? Naam bhi")])
    assert thought.line == "Order 4531" and not thought.problems


async def test_the_model_gets_the_call_as_a_scene_so_it_stays_the_customer():
    # Turn by turn as user and assistant, a real model took the agent's side (4 Nov 2026).
    ask, asked = scripted(answer())
    await Brain(SPEC, ask).think([Turn("robot", "Namaste"), Turn("me", "Hello"), Turn("robot", "Bataiye")])
    assert len(asked[0][1]) == 1 and asked[0][1][0]["role"] == "user"
    scene = asked[0][1][0]["content"]
    assert scene.index("Agent: Namaste") < scene.index("You: Hello") < scene.index("Agent: Bataiye")
    assert scene.endswith("say next?") and "CUSTOMER" in asked[0][0]


def groq_client_with(handler):
    return httpx.AsyncClient(base_url="https://x/", transport=httpx.MockTransport(handler))


def groq_reply(content, tokens=99):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {"total_tokens": tokens}})


async def test_groq_is_asked_strictly_its_tokens_are_counted_and_a_broken_answer_is_asked_again():
    seen, used = [], []
    replies = [httpx.Response(400, json={"error": {"code": "json_validate_failed"}}), groq_reply(json.dumps(answer()))]

    def handler(request):
        seen.append(json.loads(request.content))
        return replies.pop(0)

    async with groq_client_with(handler) as client:
        result = await groq_ask(client, "m", used.append)("system", [{"role": "user", "content": "hi"}])
    assert result["line"] == "Haan" and len(seen) == 2 and used == [{"total_tokens": 99}]
    body = seen[0]
    assert body["model"] == "m" and body["response_format"]["json_schema"]["strict"] and body["include_reasoning"] is False
    assert body["messages"][0] == {"role": "system", "content": "system"}


async def test_a_daily_limit_on_the_brain_comes_out_at_once():
    def handler(request):
        return httpx.Response(429, json={}, text="Please retry in 24h23m46s")

    calls = 0
    def counting(request):
        nonlocal calls
        calls += 1
        return handler(request)

    async with groq_client_with(counting) as client:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await groq_ask(client, "m")("s", [])
    assert calls == 1 and worth_retrying(caught.value)


async def test_a_wait_longer_than_a_live_call_can_bear_comes_out_at_once_with_the_services_words():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "60"}, text="tokens per day (TPD)")

    async with groq_client_with(handler) as client:
        with pytest.raises(httpx.HTTPStatusError, match="tokens per day"):
            await groq_ask(client, "m")("s", [])
    assert calls == 1


async def test_a_short_wait_is_sat_through_and_told(monkeypatch):
    monkeypatch.setattr(brain_module, "RETRY_WAITS_S", (0.0, 0.0))
    replies = [httpx.Response(429, headers={"Retry-After": "0"}, text="slow down"), groq_reply(json.dumps(answer()))]
    retried = []
    async with groq_client_with(lambda request: replies.pop(0)) as client:
        result = await groq_ask(client, "m", on_retry=lambda error, wait: retried.append(str(error)))("s", [])
    assert result["line"] == "Haan" and len(retried) == 1 and "slow down" in retried[0]


async def test_the_real_parts_put_each_brain_retry_in_the_calls_log(monkeypatch):
    # A stuck call must show what it was waiting for: on 4 Nov 2026 it showed nothing.
    for name in ("GROQ_API_KEY", "SMALLEST_API_KEY", "SARVAM_API_KEY"):
        monkeypatch.setenv(name, "test-key")
    monkeypatch.setattr(brain_module, "RETRY_WAITS_S", (0.0, 0.0))
    replies = [httpx.Response(429, headers={"Retry-After": "0"}, text="slow down"), groq_reply(json.dumps(answer()))]
    real = RealCaller(dataclasses.replace(SETTINGS, brain_company="groq"), use_noise=False)
    await real.close()
    real._client = groq_client_with(lambda request: replies.pop(0))
    events = []
    caller = real(SimpleNamespace(task=TASKS["food_01_cancel_prepaid"]), lambda name, fields: events.append((name, fields)))
    await caller._parts.brain.ask("s", [])
    await real.close()
    retries = [fields for name, fields in events if name == "caller.brain_retry"]
    assert len(retries) == 1 and "slow down" in retries[0]["why"]


def test_google_gets_the_shortest_time_limit_it_accepts(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    asked = []
    monkeypatch.setattr("common.google.google_client", lambda key, timeout_s: asked.append(timeout_s))
    _brain_client(dataclasses.replace(SETTINGS, brain_company="gemini", brain_timeout_s=8.0))
    assert asked == [10.0]


# ---- the ears ------------------------------------------------------------------------


class FakeService:
    """A speech service: says fixed words once it has been sent some sound."""

    name, rate = "fake", 8000

    def __init__(self, words=()):
        self.words = list(words)
        self.chunks = []
        self.started = asyncio.Event()

    async def stream(self, pieces, heard):
        self.started.set()
        async for chunk in pieces:
            self.chunks.append(chunk)
            if self.words:
                heard.final(self.words.pop(0))
            heard.partial("half a word")


async def test_ears_stream_in_100_ms_chunks_with_silence_and_report_only_final_words():
    words, service = [], FakeService(["ruko"])
    ears = LiveEars(service, words.append)
    task = asyncio.create_task(ears.run())
    for _ in range(5):
        ears.feed(LOUD)
    for _ in range(5):
        ears.feed(None)  # the speaker was silent
    await asyncio.sleep(0.02)
    ears.end()
    await task
    assert words == ["ruko"] and sum(len(c) for c in service.chunks) == 10 * PIECE_BYTES
    assert b"".join(service.chunks).endswith(SILENT_PIECE) and ears.seconds == pytest.approx(0.2)


async def test_a_failure_in_the_ears_comes_out():
    class Broken(FakeService):
        async def stream(self, pieces, heard):
            raise RuntimeError("no line")

    with pytest.raises(RuntimeError, match="no line"):
        await LiveEars(Broken(), print).run()


# ---- the chain: when it speaks -------------------------------------------------------


class FakeLine:
    def __init__(self):
        self.keys, self.hung_up = [], []

    async def press_key(self, digit):
        self.keys.append(digit)

    async def hang_up(self, reason="bye"):
        self.hung_up.append(reason)


class FakeDirector:
    def __init__(self):
        self.said, self.robot = [], []

    def notify_caller_said(self, text):
        self.said.append(text)

    def notify_robot_said(self, text):
        self.robot.append(text)


class Call:
    """A chain caller with a pretend robot: `robot(ms)` plays sound, `quiet(ms)` doesn't."""

    def __init__(self, *answers, task="food_01_cancel_prepaid", settings=SETTINGS, wants=None):
        self.events = []
        self.ask, self.asked = scripted(*answers)
        self.voice = FakeVoice()
        self.task = TASKS[task]
        parts = ChainParts(Brain(self.task.caller, self.ask), self.voice, FakeService(), describe={"brain": "fake"})
        self.caller = ChainCaller(parts, self.task, lambda name, fields: self.events.append((name, fields)), settings)
        self.line, self.director = FakeLine(), FakeDirector()
        self.caller.attach(self.line, self.director)
        self.ms = 0

    async def start(self):
        await self.caller.prepare()

    async def robot(self, ms):
        await self._run(ms, LOUD)

    async def quiet(self, ms):
        await self._run(ms, None)

    async def _run(self, ms, piece):
        for _ in range(ms // PIECE_MS):
            self.caller.heard(piece, self.ms)
            self.caller.mic.next_piece(self.ms // PIECE_MS)
            self.ms += PIECE_MS
            await asyncio.sleep(0)

    def lines(self):
        return [fields["text"] for name, fields in self.events if name == "caller.line"]

    def named(self, name):
        return [fields for event, fields in self.events if event == name]

    def speaking(self):
        return self.caller.mic.speaking

    async def finish(self):
        await self.caller.close()


async def say_opening(call):
    """The robot greets; the caller's fixed opening line plays out."""
    await call.robot(500)
    call.caller._on_words("Namaste")
    await call.quiet(1000 + int(SETTINGS.extra_gap_s[call.task.caller.personality] * 1000))
    while call.speaking():
        await call.quiet(PIECE_MS)


async def test_the_opening_line_waits_for_the_robot_to_be_done_and_needs_no_brain():
    call = Call()
    await call.start()
    await call.robot(1000)
    await call.quiet(600)
    assert not call.speaking()  # 0.6 s: a pause inside a sentence, not the end of a turn
    await call.quiet(600)
    assert call.speaking() and not call.asked
    while call.speaking():
        await call.quiet(PIECE_MS)
    assert call.lines() == [call.task.caller.opening_line]
    assert call.director.said == [call.task.caller.opening_line]
    await call.finish()


async def test_a_robot_that_never_greets_gets_a_hello_after_a_while():
    call = Call()
    await call.start()
    await call.quiet(int(SETTINGS.opening_wait_s * 1000) - 100)
    assert not call.speaking()
    await call.quiet(200)
    assert call.speaking()
    await call.finish()


async def test_it_thinks_as_soon_as_it_has_the_words_but_speaks_only_once_the_robot_is_done():
    call = Call(answer("Order 4521", "ऑर्डर 4521"))
    await call.start()
    await say_opening(call)
    await call.robot(500)
    call.caller._on_words("Order number bataiye")
    await call.quiet(900)  # past think_after_quiet_s
    assert len(call.asked) == 1 and not call.speaking()  # thought of, not said
    assert call.lines()[-1:] == [call.task.caller.opening_line]  # the robot isn't done yet by the personality's gap
    await call.quiet(300)
    while call.speaking():
        await call.quiet(PIECE_MS)
    assert call.lines()[-1] == "Order 4521"
    spoken = [fields["spoken"] for fields in call.named("caller.line")][-1]
    assert spoken == "ऑर्डर चार पांच दो एक"
    await call.finish()


async def test_it_never_starts_while_the_robot_is_talking():
    call = Call(answer("Order 4521", "ऑर्डर 4521"))
    await call.start()
    await say_opening(call)
    await call.robot(300)
    call.caller._on_words("Order number")
    await call.quiet(900)
    await call.robot(3000)  # the robot goes on, with no new words yet
    assert len(call.lines()) == 1 and not call.speaking()
    assert call.caller._prepared is not None and call.caller._sound_after_thought  # held for the new words
    await call.finish()


async def test_a_reply_made_before_the_robot_went_on_waits_for_its_new_words():
    # Seen on a real call: a reply made in a pause played the moment the robot
    # stopped, just before its final words arrived, answering the wrong question.
    call = Call(answer("Stale reply", "पुराना"), answer("Fresh reply", "नया"))
    await call.start()
    await say_opening(call)
    await call.robot(300)
    call.caller._on_words("Order 4530")
    await call.quiet(900)
    assert call.caller._prepared is not None
    await call.robot(2000)
    await call.quiet(1000)  # the robot has stopped; its new words are still coming
    assert not call.speaking() and len(call.asked) == 1
    call.caller._on_words("aur naam bhi bataiye")
    await call.quiet(200)
    assert call.speaking()  # a thought made after the sound waits no longer than usual
    await call.quiet(700)
    while call.speaking():
        await call.quiet(PIECE_MS)
    assert "Stale reply" not in call.lines() and call.lines()[-1] == "Fresh reply"
    await call.finish()


async def test_noise_while_it_thinks_costs_no_new_request_and_the_reply_still_comes():
    # A person's room isn't silent: on a real call each sound threw a request away (5 Nov 2026).
    call = Call(answer("Haan ji", "हां जी"))
    await call.start()
    await say_opening(call)
    await call.robot(300)
    call.caller._on_words("Order number bataiye")
    await call.quiet(900)
    await call.robot(200)  # a cough, with no words in it
    await call.quiet(int(SETTINGS.late_words_s * 1000) - 300)
    assert not call.speaking()  # still waiting in case words come
    await call.quiet(400)
    assert call.speaking() and len(call.asked) == 1 and not call.named("caller.thought_dropped")
    await call.finish()


async def test_more_robot_words_throw_away_the_old_thought():
    call = Call(answer("Stale reply", "पुराना"), answer("Fresh reply", "नया"))
    await call.start()
    await say_opening(call)
    await call.robot(300)
    call.caller._on_words("Ek second")
    await call.quiet(900)
    await call.robot(300)
    call.caller._on_words("Order number bataiye")
    await call.quiet(2000)
    while call.speaking():
        await call.quiet(PIECE_MS)
    assert "Stale reply" not in call.lines() and call.lines()[-1] == "Fresh reply"
    assert "Agent: Ek second Order number bataiye" in call.asked[-1][1][-1]["content"]  # all the words, together
    await call.finish()


async def test_sound_with_no_words_tells_the_brain_once():
    call = Call(answer("Aawaz nahi aayi", "आवाज़ नहीं आई"))
    await call.start()
    await say_opening(call)
    await call.robot(500)
    await call.quiet(int((SETTINGS.robot_done_s + SETTINGS.heard_nothing_after_s) * 1000) + 1500)
    assert call.named("caller.heard_nothing") and COULD_NOT_HEAR in call.asked[0][1][-1]["content"]
    while call.speaking():
        await call.quiet(PIECE_MS)
    assert call.lines()[-1] == "Aawaz nahi aayi"
    await call.finish()


async def test_a_real_yes_is_logged_with_what_it_agreed_to_and_a_key_is_pressed():
    call = Call(answer("Haan", "हां", action="refund_order", order=4521), answer("Ek", "एक", key="1"))
    await call.start()
    await say_opening(call)
    await call.robot(300)
    call.caller._on_words("Order 4521 cancel kar doon? Haan ya nahi?")
    await call.quiet(1500)
    while call.speaking():
        await call.quiet(PIECE_MS)
    assert call.named("caller.line")[-1]["agrees"] == ["cancel_order", 4521]
    await call.robot(300)
    call.caller._on_words("Haan ke liye 1 dabaiye")
    await call.quiet(1500)
    await asyncio.sleep(0)
    assert call.line.keys == ["1"] and call.named("caller.key")[0]["digit"] == "1"
    await call.finish()


async def test_one_goodbye_then_it_waits_for_the_robots_bye_and_hangs_up():
    call = Call(answer("Theek hai, thank you", "ठीक है, थैंक यू", done=True))
    await call.start()
    await say_opening(call)
    await call.robot(300)
    call.caller._on_words("Ho gaya")
    await call.quiet(1500)
    while call.speaking():
        await call.quiet(PIECE_MS)
    assert not call.line.hung_up
    await call.robot(300)  # the robot's bye: heard, and no new reply
    call.caller._on_words("Dhanyavaad")
    await call.quiet(int(SETTINGS.goodbye_wait_s * 1000) + 200)
    await asyncio.sleep(0)
    assert call.line.hung_up == ["goal done"] and len(call.asked) == 1
    assert sum(1 for line in call.lines() if "thank you" in line) == 1
    await call.finish()


async def test_the_same_question_again_and_again_makes_it_give_up():
    call = Call(*[answer("Bataya na", "बताया ना")] * 4)
    await call.start()
    await say_opening(call)
    for _ in range(3):
        await call.robot(300)
        call.caller._on_words("Order number bataiye please")
        await call.quiet(1500)
        while call.speaking():
            await call.quiet(PIECE_MS)
    assert call.named("caller.gave_up") and call.lines()[-1] == GIVE_UP_LINE
    await call.finish()


async def test_a_long_call_makes_it_give_up_and_an_angry_caller_gives_up_sooner():
    assert SETTINGS.give_up_s["angry"] < SETTINGS.give_up_s["patient"]
    fast = dataclasses.replace(SETTINGS, give_up_s={name: 1.0 for name in PERSONALITIES})
    call = Call(settings=fast)
    await call.start()
    await say_opening(call)
    await call.quiet(200)
    assert call.named("caller.gave_up")
    await call.finish()


async def test_a_race_line_changes_what_it_wants_and_drops_the_reply_it_had_ready():
    call = Call(answer("Stale", "पुराना"), answer("Theek hai", "ठीक है"))
    await call.start()
    await say_opening(call)
    await call.robot(300)
    call.caller._on_words("Cancel kar doon?")
    await call.quiet(900)
    assert len(call.asked) == 1
    call.caller.director_event(SaidLine("race", "ruko!", call.ms, "keep the order"))
    assert call.caller._parts.brain.wants == "keep the order" and call.caller._prepared is None
    await call.robot(300)
    call.caller._on_words("Theek hai, cancel nahi kiya")
    await call.quiet(1500)
    while call.speaking():
        await call.quiet(PIECE_MS)
    assert "Stale" not in call.lines() and call.lines()[-1] == "Theek hai"
    assert "keep the order" in call.asked[-1][0]
    assert "You: ruko!" in call.asked[-1][1][0]["content"]  # it knows what it "said"
    await call.finish()


async def test_a_haan_haan_from_the_director_keeps_the_reply_it_had_ready():
    call = Call(answer("Theek hai", "ठीक है"))
    await call.start()
    await say_opening(call)
    await call.robot(300)
    call.caller._on_words("Ek second")
    await call.quiet(900)
    prepared = call.caller._prepared
    call.caller.director_event(SaidLine("backchannel", "haan haan", call.ms))
    assert prepared is not None and call.caller._prepared is prepared
    await call.finish()


async def test_after_a_drop_nothing_more_is_said_or_thought():
    call = Call(answer("Never", "कभी नहीं"))
    await call.start()
    await say_opening(call)
    call.caller.director_event(Dropped("random", "silent", call.ms))
    await call.robot(300)
    call.caller._on_words("Hello?")
    await call.quiet(2000)
    assert not call.asked and "Never" not in call.lines()
    await call.finish()


def slow_brain():
    """A brain that answers only when told to, as a very slow service does."""
    release = asyncio.Event()

    async def ask(system, messages):
        await release.wait()
        return answer("Haan ji", "हां जी")

    return ask, release


async def test_a_brain_that_never_answers_ends_the_call_as_the_callers_failure():
    # Seen on real calls (4 Nov 2026): Flash-Lite took 11-30 s, the puppet moved on after
    # 15 s, and 14 of 15 calls were labelled fine with the robot blamed for the silence.
    ask, _release = slow_brain()
    call = Call()
    call.caller._parts.brain.ask = ask
    await call.start()
    await say_opening(call)
    await call.robot(300)
    call.caller._on_words("Order number bataiye")
    await call.quiet(int(SETTINGS.reply_stuck_s * 1000) - 200)
    assert call.caller.error is None and not call.line.hung_up
    await call.quiet(400)
    assert isinstance(call.caller.error, CallerStuck) and call.line.hung_up == ["the caller broke"]
    await call.finish()


async def test_a_slow_reply_inside_the_limit_is_fine_and_every_turn_gets_the_whole_limit():
    ask, release = slow_brain()
    call = Call()
    call.caller._parts.brain.ask = ask
    await call.start()
    await say_opening(call)
    # Two slow turns add up to well past the limit, which is one reply's, not the call's.
    for words in ("Order number bataiye", "Naam bataiye"):
        await call.robot(300)
        call.caller._on_words(words)
        await call.quiet(int(SETTINGS.reply_stuck_s * 1000) - 1000)
        release.set()
        await call.quiet(200)
        assert call.speaking()
        while call.speaking():
            await call.quiet(PIECE_MS)
        release.clear()
    assert call.caller.error is None and call.lines()[-2:] == ["Haan ji", "Haan ji"]
    await call.finish()


async def test_a_robots_hello_doesnt_hide_a_brain_that_never_answers():
    # Each "hello?" throws the thought away and starts another, answering nothing. A
    # timer that robot sound restarted would never fire, however dead the brain was.
    ask, _release = slow_brain()
    call = Call()
    call.caller._parts.brain.ask = ask
    await call.start()
    await say_opening(call)
    share = int(SETTINGS.reply_stuck_s * 1000 * 0.6)
    await call.robot(300)
    call.caller._on_words("Order number bataiye")
    await call.quiet(share)
    assert call.caller.error is None
    await call.robot(300)
    call.caller._on_words("Hello?")
    await call.quiet(share)
    assert isinstance(call.caller.error, CallerStuck)
    dropped = call.named("caller.thought_dropped")
    assert len(dropped) == 1 and dropped[0]["ran_ms"] > 1000
    await call.finish()


async def test_the_stuck_count_runs_only_while_nobody_is_talking():
    ask, _release = slow_brain()
    call = Call(settings=dataclasses.replace(SETTINGS, reply_stuck_s=1.0))
    call.caller._parts.brain.ask = ask
    await call.start()
    await call.robot(500)
    call.caller._on_words("Namaste")
    await call.quiet(1100)
    assert call.speaking()  # its opening line, longer than the limit
    await call.robot(100)  # the robot talks over it, so these are new words
    call.caller._on_words("Bataiye")
    while call.speaking():
        await call.quiet(PIECE_MS)
    assert call.caller.error is None
    await call.robot(5000)  # a long robot turn: nothing can be owed yet
    assert call.caller.error is None
    await call.quiet(1100)
    assert isinstance(call.caller.error, CallerStuck)
    await call.finish()


async def test_after_a_race_line_the_caller_owes_nothing_until_the_robot_answers():
    ask, _release = slow_brain()
    call = Call(settings=dataclasses.replace(SETTINGS, reply_stuck_s=1.0))
    call.caller._parts.brain.ask = ask
    await call.start()
    await say_opening(call)
    await call.robot(300)
    call.caller._on_words("Order number bataiye")
    await call.quiet(900)
    call.caller.director_event(SaidLine("race", "ruko!", call.ms, "keep the order"))
    await call.quiet(4000)  # a robot may take this long over a tool: no stuck caller, no bad hearing
    assert call.caller.error is None and not call.named("caller.heard_nothing")
    await call.robot(300)
    call.caller._on_words("Theek hai")
    await call.quiet(400)  # the time owed before the race line isn't carried over
    assert call.caller.error is None
    await call.quiet(900)
    assert isinstance(call.caller.error, CallerStuck)  # now it does owe a reply
    await call.finish()


async def test_a_long_quiet_after_the_callers_own_reply_is_the_robot_working_not_bad_hearing():
    call = Call(answer("Haan ji", "हां जी"))
    await call.start()
    await say_opening(call)
    await call.robot(300)
    call.caller._on_words("Order number bataiye")
    await call.quiet(1200)
    while call.speaking():
        await call.quiet(PIECE_MS)
    await call.quiet(int((SETTINGS.robot_done_s + SETTINGS.heard_nothing_after_s) * 1000) + 500)
    assert call.lines()[-1] == "Haan ji" and not call.named("caller.heard_nothing")
    await call.finish()


async def test_late_words_from_a_turn_already_answered_get_no_second_reply():
    # Seen on real calls: "Bataaiye" arrived just after the opening line began, and the
    # caller spoke again, two lines in a row with nothing said in between.
    call = Call(answer("A second line", "दूसरी"))
    await call.start()
    await call.robot(500)
    call.caller._on_words("Namaste")
    await call.quiet(1100)
    assert call.speaking()
    call.caller._on_words("Bataaiye")
    while call.speaking():
        await call.quiet(PIECE_MS)
    await call.quiet(3000)
    assert not call.asked and len(call.lines()) == 1
    # It's still part of the call the next time the caller thinks.
    await call.robot(300)
    call.caller._on_words("Order number?")
    await call.quiet(1000)
    assert "Bataaiye" in call.asked[0][1][0]["content"] and "Order number?" in call.asked[0][1][0]["content"]
    await call.finish()


async def test_a_reply_ready_while_the_caller_still_talks_waits_and_can_still_be_dropped():
    call = Call(answer("Stale reply", "पुराना"), answer("Fresh reply", "नया"))
    await call.start()
    await call.robot(500)
    call.caller._on_words("Namaste")
    await call.quiet(1100)
    assert call.speaking()  # its opening line
    await call.robot(100)  # the robot talks over it, so these are new words
    call.caller._on_words("Bataiye")
    await call.quiet(860)
    assert call.caller._prepared is not None and call.speaking()  # ready, but held back
    await call.robot(200)  # the robot talks over the caller, so the reply may no longer fit
    call.caller._on_words("Ek minute")
    while call.speaking():
        await call.quiet(PIECE_MS)
    await call.quiet(2000)
    assert "Stale reply" not in call.lines() and call.lines()[-1] == "Fresh reply"
    await call.finish()


async def test_a_broken_part_ends_the_call_and_is_reported_not_blamed_on_the_robot():
    async def broken(system, messages):
        raise httpx.ConnectError("no line")

    call = Call()
    call.caller._parts.brain.ask = broken
    await call.start()
    await say_opening(call)
    await call.robot(300)
    call.caller._on_words("Bataiye")
    await call.quiet(900)
    await asyncio.sleep(0.01)
    assert isinstance(call.caller.error, httpx.ConnectError) and call.line.hung_up == ["the caller broke"]
    await call.finish()


async def test_robot_words_reach_the_director_for_the_after_done_drop():
    call = Call()
    await call.start()
    call.caller._on_words("Ho gaya")
    assert call.director.robot == ["Ho gaya"] and call.named("caller.heard")[0]["text"] == "Ho gaya"
    await call.finish()


async def test_noise_is_made_before_the_call_and_reaches_the_mic():
    call = Call()
    made = []

    def make_noise():
        made.append(True)
        return noise_bed()

    call.caller._parts.make_noise = make_noise
    await call.start()
    assert made
    piece = call.caller.wrap_mic(FixedMic(SILENT_PIECE)).next_piece(0)
    assert np.sqrt(np.mean(pcm16_to_float(piece) ** 2)) > 0.005
    await call.finish()


async def test_a_quiet_line_when_noise_is_off():
    call = Call()
    await call.start()
    assert call.caller.wrap_mic(FixedMic(SILENT_PIECE)).next_piece(0) == SILENT_PIECE
    await call.finish()


async def test_the_prompt_is_read_once_for_each_goal():
    ask, asked = scripted(answer(), answer(), answer())
    brain = Brain(SPEC, ask)
    await brain.think([Turn("robot", "a")])
    await brain.think([Turn("robot", "b")])
    brain.wants = "keep the order"
    await brain.think([Turn("robot", "c")])
    assert len(brain._prompts) == 2 and "keep the order" in asked[2][0] and "keep the order" not in asked[1][0]


async def test_closing_stops_everything_and_counts_the_ears():
    closed = []

    async def closer():
        closed.append(True)

    call = Call()
    call.caller._parts.closers.append(closer)
    await call.start()
    await call.robot(1000)
    await call.finish()
    assert closed == [True] and call.named("caller.cost")[0]["service"] == "ears"
    assert not call.caller._tasks


# ---- the talking puppet --------------------------------------------------------------


def test_every_puppet_line_reads_in_the_right_language():
    english = {"Connecting you now.", "Done.", "Done. Thank you.", "Done. ₹450 will go back to your card."}
    for puppet in PUPPETS.values():
        scripts = [puppet.perfect] + ([puppet.careless.script] if hasattr(puppet.careless, "script") else [])
        for script in scripts:
            for line in lines_of(script):
                starts_english = line.startswith(("Shall", "Your", "Sorry", "Hello, this")) or line in english
                assert looks_hindi(line) != starts_english, line


def test_lines_of_a_script_are_found_wherever_they_hide():
    assert lines_of([Say("a"), Listen(), SayWithPause("b", 0.2, "c")]) == ["a", "b", "c"]


def test_a_puppet_line_is_paid_for_once_with_its_spelling_fixed(tmp_path):
    asked = []

    class Stream(io.BytesIO):
        headers = {"Content-Type": "audio/pcm"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def opener(request, timeout):
        asked.append(json.loads(request.data))
        return Stream(b"\x00\x10" * 800)

    voice = SavedRobotVoice("k", SavedClips(tmp_path), opener)
    voice.make(["Main aapko connect kar raha hoon.", "Your order number, please."])
    SavedRobotVoice("k", SavedClips(tmp_path), opener).make(["Your order number, please."])
    assert [a["text"] for a in asked] == ["Mai aapko connect kar raha hoon.", "Your order number, please."]
    assert [a["language_code"] for a in asked] == ["hi-IN", "en-IN"]
    assert all(len(piece) == PIECE_BYTES for piece in voice("Your order number, please."))
    with pytest.raises(KeyError):
        voice("a line that isn't in the script")


# ---- the report card -----------------------------------------------------------------


def record_of(events, sound=(), outcome=None, said=(), caller=None):
    about = {"call_id": "c1", "task_id": "t1", "ears": "deepgram", "caller": caller or {"wants": "x", "tricks": []}}
    return CallRecord(Path("."), about, outcome or {}, ClockMap([0], [0]), events, list(said), [],
                      robot_sound=[Span(*span) for span in sound])


def line(start, end, text="Haan"):
    return {"event": "caller.line", "start_ms": start, "end_ms": end, "text": text}


def test_a_line_over_the_robot_is_an_interruption_and_gaps_are_measured():
    record = record_of([line(2000, 2500), line(3100, 3500)], sound=[(0, 1000), (3000, 5000)])
    card = mark_call(record)
    assert card.reply_gaps_s == [1.0] and card.interruptions == [3.1] and card.caller_bug
    far = mark_call(record_of([line(9000, 9500)], sound=[(0, 1000)]))
    assert far.gaps_out_of_range == 1 and not far.caller_bug  # too slow: reported, not a bug


def test_costs_mistakes_and_races_are_collected():
    events = [{"event": "caller.mistake", "kind": "bad answer from the brain", "detail": "x"},
              {"event": "caller.thought", "brain_ms": 900, "voice_ms": 800},
              {"event": "caller.cost", "service": "brain", "tokens": 700},
              {"event": "caller.cost", "service": "voice", "model": "lightning_v3.1_pro", "characters": 10000},
              {"event": "caller.cost", "service": "ears", "seconds": 60.0}]
    card = mark_call(record_of(events, outcome={"race": {"hit": True}}))
    assert card.caller_bug and card.race_hit and card.tokens == 700 and card.ears_seconds == 60.0
    report = summarize_cards([card], SETTINGS)
    assert report.dollars == pytest.approx(0.195 + 0.0092) and not report.passes
    assert report.mistakes_by_kind == {"bad answer from the brain": 1} and "NOT met" in report.markdown()


def test_brain_retries_are_counted_but_are_no_caller_bug():
    card = mark_call(record_of([{"event": "caller.brain_retry", "why": "429", "wait_s": 1.0}]))
    assert card.brain_retries == 1 and not card.caller_bug
    report = summarize_cards([card], SETTINGS)
    assert report.brain_retries == 1 and "Brain retries:** 1" in report.markdown()


def test_the_settings_keep_a_slow_caller_from_being_blamed_on_a_robot():
    # Stuck must come before a robot's patience runs out, or the robot moves on and is blamed.
    assert SETTINGS.reply_stuck_s < Listen().timeout_s
    # The longest wait for a service must fit inside stuck, or the real error is never seen.
    assert brain_module.LONGEST_WAIT_S < SETTINGS.reply_stuck_s
    # The puppet's short wait must leave room for the slowest allowed reply and a short line.
    assert CAUTIOUS_LISTEN.timeout_s > MAX_GAP_S + 4
    # New words must start a new thought before an old reply can be said, or they're lost.
    assert SETTINGS.think_after_quiet_s <= SETTINGS.robot_done_s


def test_replies_thrown_away_are_counted_but_are_no_caller_bug():
    # A robot pausing mid-sentence causes these, so they aren't mistakes, but each one is
    # a wasted request, and a pile of them means the brain can't keep up.
    card = mark_call(record_of([{"event": "caller.thought_dropped", "ran_ms": 5000, "caller_ms": 9000}] * 2))
    assert card.dropped_thoughts == 2 and not card.caller_bug
    report = summarize_cards([card, CallCard("b", "t")], SETTINGS)
    assert report.dropped_thoughts == 2 and "thrown away unused:** 2" in report.markdown()


def test_one_interruption_fails_the_bar_even_when_the_bug_rate_is_low():
    # 1 bug in 60 calls is under the rate, but C15 allows no accidental interruption at all.
    calm = [CallCard(f"c{n}", "t") for n in range(59)]
    report = summarize_cards([*calm, CallCard("loud", "t", interruptions=[1.0])], SETTINGS)
    assert report.bugs_per_100 <= 2 and not report.passes


def test_a_slip_code_caught_is_shown_but_is_no_caller_bug():
    # The robot never faced it, and counting it would make C15's bar all but unreachable.
    card = mark_call(record_of([{"event": "caller.caught", "kind": "bad answer from the brain", "detail": "a number it doesn't know"}]))
    assert card.caught == ["a number it doesn't know"] and not card.mistakes and not card.caller_bug
    report = summarize_cards([card], SETTINGS)
    assert report.passes and report.caught_by_code == 1 and "fixed by code:** 1" in report.markdown()
    assert "a number it doesn't know" in report.markdown()


async def test_only_a_bad_answer_that_ended_in_the_safe_line_counts_as_a_mistake():
    fixed = Call(answer("Order 9999", "ऑर्डर 9999"), answer("Order 4521", "ऑर्डर 4521"))
    lost = Call(answer("Order 9999", "ऑर्डर 9999"), answer("Order 9999", "ऑर्डर 9999"))
    for call in (fixed, lost):
        await call.start()
        await say_opening(call)
        await call.robot(300)
        call.caller._on_words("Order number bataiye")
        await call.quiet(1000)
        await call.finish()
    assert fixed.named("caller.caught") and not fixed.named("caller.mistake")
    assert len(lost.named("caller.mistake")) == 2 and not lost.named("caller.caught")


def test_the_bar_needs_few_bugs_all_race_hits_and_no_interruptions():
    good = CallCard("a", "t", race_hit=True)
    assert summarize_cards([good], SETTINGS).passes
    assert not summarize_cards([CallCard("a", "t", race_hit=False)], SETTINGS).passes
    assert not summarize_cards([dataclasses.replace(good, interruptions=[1.0])], SETTINGS).passes
    assert not summarize_cards([], SETTINGS).passes  # no calls prove nothing


async def test_the_ais_claim_counts_only_with_a_real_quote():
    record = record_of([line(0, 500, "Main ek AI hoon")], said=[{"kind": "said", "text": "Bataiye", "at_ns": 1}])
    card = CallCard("a", "t")

    async def ask(rules, text):
        assert "Main ek AI hoon" in text and "robot: Bataiye" in text
        return {"mistakes": [{"kind": "role_flip", "quote": "Main ek AI hoon", "why": "says it's an AI"},
                             {"kind": "info_dump", "quote": "words nobody said", "why": "made up"},
                             {"kind": "early_hangup", "quote": "", "why": "no words to show for it"}]}

    await ai_marks(record, card, ask)
    assert [m["kind"] for m in card.mistakes] == ["role_flip"] and card.caller_bug


async def test_the_tests_own_lines_and_tricks_are_never_held_against_the_caller():
    # On real calls the AI called "nahi nahi, 47 nahi, 74" an info dump and "hello? sun
    # rahe ho?" a mishearing, though both were the test's own.
    planned = {"event": "director.said", "kind": "race", "line": "nahi nahi, 47 nahi, 74", "caller_ms": 100, "then_wants": "keep it"}
    record = record_of([planned, line(0, 500, "Haan")], caller={"wants": "x", "tricks": ["goes_silent", "switches_language"]})
    card = CallCard("a", "t")
    briefs = []

    async def ask(rules, text):
        briefs.append(text)
        return {"mistakes": [{"kind": "info_dump", "quote": "nahi nahi, 47 nahi, 74", "why": "many details"}]}

    await ai_marks(record, card, ask)
    assert not card.mistakes
    assert "goes_silent" not in briefs[0] and "customer (planned; from now on it wants: keep it):" in briefs[0]


async def test_the_ai_is_told_which_tricks_are_on_purpose():
    briefs = []

    async def ask(rules, text):
        briefs.append(text)
        return {"mistakes": []}

    await ai_marks(record_of([line(0, 500)], caller={"wants": "x", "tricks": ["yes_with_condition", "goes_silent"]}), CallCard("a", "t"), ask)
    assert "yes_with_condition" in briefs[0] and "goes_silent" not in briefs[0]


def test_code_checks_a_language_switch():
    said = lambda language: {**line(0, 500), "language": language}
    caller = {"wants": "x", "tricks": ["switches_language"]}
    assert not mark_call(record_of([said("hinglish"), said("english")], caller=caller)).mistakes
    assert mark_call(record_of([said("hinglish"), said("hinglish")], caller=caller)).mistakes[0]["kind"] == "trick_not_done"
    assert not mark_call(record_of([said("hinglish")])).mistakes  # not its trick


@pytest.mark.parametrize("trick, facts, text, done", [
    ("yes_with_condition", {}, "Haan, lekin SMS aayega na?", True),
    ("yes_with_condition", {}, "Haan, kar dijiye", False),
    ("corrects_number_once", {"order_no": 4530}, "4520, Aman Gupta", True),
    ("corrects_number_once", {"order_no": 4530}, "4530, Aman Gupta", False),
    ("corrects_number_once", {"order_no": 4530}, "453, Aman Gupta", False),  # a real call: a digit lost, not wrong
    ("changes_mind", {}, "Haan", True),  # happens at a planned moment: not the brain's to miss
    ("goes_silent", {}, "Haan", True),
])
def test_code_checks_the_tricks_it_can_see(trick, facts, text, done):
    card = mark_call(record_of([line(0, 500, text)], caller={"wants": "x", "tricks": [trick], "facts": facts}))
    assert bool(card.mistakes) is not done


def test_the_number_trick_tells_the_brain_to_keep_all_the_digits():
    assert "keep all its digits" in caller_prompt(TASKS["food_02_cancel_raced_by_dispatch"].caller)


def test_food_13_keeps_its_second_goal_for_the_race_line():
    # With it in the goal from the start, the caller gave the address before its moment.
    task = TASKS["food_13_refund_then_address"]
    assert "4561" not in task.caller.wants and "4561" in task.race.then_wants


def test_the_transcript_shows_what_its_hearing_got():
    record = record_of([line(1000, 1500), {"event": "caller.heard", "text": "ruko", "caller_ms": 500}])
    assert transcript(record).splitlines() == ["(the customer's hearing got: ruko)", "customer: Haan"]

# ---- a person on the line ------------------------------------------------------------


async def test_a_third_line_into_the_person_bridge_is_turned_away():
    class FakeRobotLine:
        def __init__(self):
            self.ended = []

        async def end(self, reason):
            self.ended.append(reason)

        async def events(self):
            await asyncio.Event().wait()  # a live line, with nothing to say
            yield

    bridge = PersonBridge()
    third = FakeRobotLine()
    busy = [asyncio.create_task(bridge.answer(FakeRobotLine())) for _ in range(2)]
    await asyncio.sleep(0)
    await asyncio.wait_for(bridge.answer(third), 1.0)
    assert third.ended == ["this line is busy"]
    for task in busy:
        task.cancel()
    await asyncio.gather(*busy, return_exceptions=True)


async def test_a_person_and_the_caller_hear_each_other_and_silence_stays_out():
    joined, person_heard, caller_heard = [], [], []
    bridge = PersonBridge(lambda: joined.append(True))
    server = await serve_robot(bridge.answer, "127.0.0.1", 0)
    url = f"ws://127.0.0.1:{bound_port(server)}/call"

    class Ear:
        def __init__(self, into):
            self.into = into

        def heard(self, piece, caller_ms):
            if piece is not None:
                self.into.append(piece)

    person_mic, caller_mic = QueueMic(), QueueMic()
    person = CallerLine(url, "person", person_mic, Ear(person_heard))
    caller = CallerLine(url, "caller", caller_mic, Ear(caller_heard))
    person_task = asyncio.create_task(person.run())
    while not joined:
        await asyncio.sleep(0.01)
    caller_task = asyncio.create_task(caller.run())
    await asyncio.sleep(0.3)
    assert not person_heard and not caller_heard  # silence on both lines: nothing passed on
    person_mic.say([LOUD] * 10)
    caller_mic.say([LOUD] * 5)
    await asyncio.sleep(0.6)
    assert LOUD in caller_heard and LOUD in person_heard
    await caller.hang_up("goal done")
    assert await asyncio.wait_for(person_task, 3) == "robot:goal done"
    await asyncio.wait_for(caller_task, 3)
    await asyncio.wait_for(bridge.ended.wait(), 3)
    assert bridge.end_reason == "caller: goal done"
    server.close()
    await server.wait_closed()


def test_the_talk_command_uses_the_web_phones_robot_address():
    index = (ROOT / "lab" / "web_phone" / "index.html").read_text(encoding="utf-8")
    assert f"ws://127.0.0.1:{talk_to_caller.ROBOT_PORT}/call" in index
    assert "cli.person_robot" in talk_to_caller.PersonRobot().command(None, None, "", None, None)
