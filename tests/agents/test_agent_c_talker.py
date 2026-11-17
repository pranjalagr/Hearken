"""Agent C's Talker speaks only notes (build step 3): code writes every number and every
outcome, cuts replies to two sentences, keeps one call language, and fixes spellings."""

from pipecat.frames.frames import (InterruptionFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame,
                                   TTSSpeakFrame)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import SleepFrame, run_test

from agents.agent_c.notebook import ANSWER, FACT, NEED_INFO, RESULT, Note
from agents.agent_c.robot import WORDS, CallParts
from agents.agent_c.settings import load_settings
from agents.agent_c.talker.language import ENGLISH, HINGLISH, StickyLanguage, from_ears
from agents.agent_c.talker.mouth import Mouth
from agents.agent_c.talker.note_view import NoteView
from agents.agent_c.talker.speaker import Speaker, first_sentences
from agents.common.diary import Diary
from engine.contract.spoken import SpokenWords
from packs.food.pack import FoodPack

FOUND = {"tool": "find_order", "inputs": {}, "status": "ok", "data": {
    "order_no": 4561, "status": "placed", "items": ["1 x Club Sandwich"], "amount": "₹249.50", "payment": "card",
    "address": "House 7, Lane 3, Model Town, Ludhiana", "refunded": "₹0"}}


class ListSink:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


def words() -> SpokenWords:
    return SpokenWords({**FoodPack().spoken_words().groups, **SpokenWords.from_yaml(WORDS).groups})


def talker(language=HINGLISH):
    w = words()
    view = NoteView(w)
    lang = StickyLanguage(4, start=language)
    return view, Speaker(w, view, lang, 2), lang


def note(kind, text="", data=None, version=1, number=1):
    return Note(number, kind, text, version, data or {})


# ---- the note view ------------------------------------------------------------------


def test_a_found_order_reads_with_slots_and_ready_phrases_never_raw_numbers():
    view, _, _ = talker()
    line = view.line(note(FACT, data=FOUND), HINGLISH)
    assert "4561" not in line and "249" not in line
    assert "order no [order]" in line and "amount [amount]" in line and "refunded [refunded]" in line
    assert "restaurant ne abhi banana shuru nahi kiya" in line  # the status's ready phrase
    assert view.slots["order"] == "4561"


def test_a_second_order_gets_its_own_slot_and_the_thinkers_words_are_slotted_too():
    view, _, _ = talker()
    view.line(note(FACT, data=FOUND), HINGLISH)
    other = {**FOUND, "data": {**FOUND["data"], "order_no": 4512, "amount": "₹320"}}
    line = view.line(note(FACT, data=other), HINGLISH)
    assert "[order_2]" in line and "[amount_2]" in line
    assert view.line(note(ANSWER, "Order 4561 is placed, ₹249.50 paid."), HINGLISH) == "Order [order] is placed, [amount] paid."
    assert view.line(note(NEED_INFO, "which order, 4561 or 4512?"), HINGLISH) == "ask the caller: which order, [order] or [order_2]?"


def test_a_result_note_gives_the_result_slot_and_keeps_the_reason():
    view, speaker, _ = talker()
    line = view.line(note(RESULT, data={"action": "cancel_order", "outcome": "refused", "reason": "Already out for delivery."}),
                     HINGLISH)
    assert line == "cancel order [result]; reason: Already out for delivery."
    speaker.given(line)
    assert speaker.prepare("[result] Order rider ke saath nikal chuka hai.").spoken.startswith("Order cancel nahi hua.")


# ---- the speaker ----------------------------------------------------------------------


def test_slots_are_filled_in_the_calls_language_with_the_packs_money_words():
    view, speaker, lang = talker()
    speaker.given("NOTES: " + view.line(note(FACT, data=FOUND), HINGLISH))
    speech = speaker.prepare("Order [order] ka amount [amount] hai.")
    assert speech.spoken == "Order 4561 ka amount 249 rupaye 50 paise hai." and not speech.safe
    lang.value = ENGLISH
    assert speaker.prepare("Order [order] costs [amount].").spoken == "Order 4561 costs 249 rupees 50 paise."


def test_a_number_no_note_gave_or_a_made_up_slot_is_never_spoken():
    view, speaker, _ = talker()
    speaker.given("NOTES: " + view.line(note(FACT, data=FOUND), HINGLISH))
    speaker.heard("mera order 4533 hai")
    made_up = speaker.prepare("Order 4535 mil gaya.")  # Agent A's run 7: 4535 for 4533
    assert made_up.safe and made_up.spoken == "Ek second, mai check kar raha hoon." and "4535" in made_up.why
    assert speaker.prepare("Aapka [refund] ho gaya.").safe
    assert not speaker.prepare("Aapne 4533 bola tha?").safe  # the caller's own number may be repeated


def test_replies_are_cut_to_two_sentences_and_main_becomes_mai():
    assert first_sentences("Ek. Do? Teen! Char.", 2) == "Ek. Do?"
    _, speaker, _ = talker()
    speech = speaker.prepare("Main dekh raha hoon. Ek second. Aur kuch?")
    assert speech.spoken == "Mai dekh raha hoon. Ek second." and speech.reply == "Main dekh raha hoon. Ek second."


# ---- the sticky language ----------------------------------------------------------------


def test_the_language_changes_only_on_a_long_enough_turn_in_another_language():
    lang = StickyLanguage(4)
    lang.ears_said = "en-IN"
    assert lang.turn_finished("okay thank you") == HINGLISH
    assert lang.turn_finished("I want to cancel my order") == ENGLISH
    lang.ears_said = "Language.HI_IN"
    assert lang.turn_finished("haan") == ENGLISH
    assert (from_ears("hi-IN"), from_ears(None), from_ears("ta-IN")) == (HINGLISH, None, None)


# ---- the mouth, in a pipeline -------------------------------------------------------------


async def test_the_mouth_sends_one_checked_line_to_the_voice_and_keeps_slots_in_the_talkers_history():
    view, speaker, _ = talker()
    speaker.given("NOTES: " + view.line(note(FACT, data=FOUND), HINGLISH))
    context, sink = LLMContext(), ListSink()
    frames = [LLMFullResponseStartFrame(), LLMTextFrame("Order [order] mil "), LLMTextFrame("gaya."), LLMFullResponseEndFrame()]
    down, _ = await run_test(Mouth(speaker, context, Diary("c1", sink)), frames_to_send=frames,
                             expected_down_frames=[TTSSpeakFrame])
    assert down[0].text == "Order 4561 mil gaya." and down[0].append_to_context is False
    assert context.messages[-1] == {"role": "assistant", "content": "Order [order] mil gaya."}


async def test_a_reply_cut_off_by_the_caller_is_never_spoken_later():
    _, speaker, _ = talker()
    # The pauses let each frame land in order: a system frame otherwise jumps the queue.
    frames = [LLMFullResponseStartFrame(), LLMTextFrame("Ho gaya"), SleepFrame(0.05), InterruptionFrame(), SleepFrame(0.05),
              LLMFullResponseEndFrame()]
    down, _ = await run_test(Mouth(speaker, LLMContext(), Diary("c1", ListSink())), frames_to_send=frames,
                             expected_down_frames=[InterruptionFrame])
    assert not any(isinstance(frame, TTSSpeakFrame) for frame in down)


def test_the_call_parts_feed_the_number_check_with_every_note_and_turn():
    call = CallParts(load_settings(), Diary("c1", ListSink()), FoodPack().spoken_words())
    line = call.notes_line([note(FACT, data=FOUND)])
    assert line.startswith("NOTES: find_order: ok;")
    call.caller_finished("haan 4561 wala")
    assert not call.speaker.prepare("Order [order] ka status sun lijiye.").safe


def test_the_talker_reads_the_backends_reason_for_a_refusal():
    # 16 Nov 2026, food_02: the Talker saw only "prepare_cancel_order: refused" and made up
    # "khana dispatch ho chuka hai" for what was really "No order 4531 found".
    from agents.agent_c.notebook import Note
    from agents.agent_c.talker.note_view import NoteView
    from agents.agent_c.thinker.copy_notes import note_for_reply
    from agents.agent_c.thinker.brain import ToolCall
    from engine.contract.spoken import SpokenWords
    from engine.service.replies import ToolReply

    kind, text, data = note_for_reply(ToolCall("prepare_cancel_order", {"order_no": "4531"}),
                                      ToolReply("refused", "No order 4531 found. Ask the caller to repeat the number."))
    line = NoteView(SpokenWords()).line(Note(1, kind, text, 1, data), "hinglish")
    assert "No order 4531 found" in line


async def test_speech_watch_keeps_the_words_that_arrive_before_the_voice_starts():
    # 16 Nov 2026, food_03: from silence, the question's text came before its sound; the
    # watch cleared its words when the sound started and reported "" for a full question.
    from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, TTSTextFrame
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection

    from agents.agent_c.talker.speech_watch import SpeechWatch

    ended: list[tuple[bool, str]] = []
    watch = SpeechWatch(lambda: None, lambda cut, words: ended.append((cut, words)))

    async def push(frame):
        await watch.on_push_frame(FramePushed(source=None, destination=None, frame=frame,
                                              direction=FrameDirection.DOWNSTREAM, timestamp=0))

    await push(TTSTextFrame("Order 4512 par refund kar doon?", aggregated_by="sentence"))
    await push(TTSTextFrame("Haan ya nahi?", aggregated_by="sentence"))
    await push(BotStartedSpeakingFrame())
    await push(BotStoppedSpeakingFrame())
    await push(BotStartedSpeakingFrame())
    await push(BotStoppedSpeakingFrame())
    assert ended == [(False, "Order 4512 par refund kar doon? Haan ya nahi?"), (False, "")]  # reported once


async def test_speech_watch_counts_the_voices_words_only_as_the_output_plays_them():
    # 16 Nov 2026, food_01: the question's text was seen while the line before it played,
    # so its mark went out ~2 s before the question ended.
    from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, TTSTextFrame
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection

    from agents.agent_c.talker.speech_watch import SpeechWatch

    voice, output = object(), object()
    ended: list[tuple[bool, str]] = []
    watch = SpeechWatch(lambda: None, lambda cut, words: ended.append((cut, words)))
    watch.output = output

    async def push(frame, source):
        await watch.on_push_frame(FramePushed(source=source, destination=None, frame=frame,
                                              direction=FrameDirection.DOWNSTREAM, timestamp=0))

    ek_second, question = TTSTextFrame("Ek second.", aggregated_by="sentence"), TTSTextFrame("Haan ya nahi?", aggregated_by="sentence")
    await push(BotStartedSpeakingFrame(), output)
    await push(ek_second, voice)
    await push(question, voice)  # made by the voice, not played yet
    await push(ek_second, output)
    await push(BotStoppedSpeakingFrame(), output)
    await push(BotStartedSpeakingFrame(), output)
    await push(question, output)
    await push(BotStoppedSpeakingFrame(), output)
    assert ended == [(False, "Ek second."), (False, "Haan ya nahi?")]


def test_a_notes_line_the_talker_copied_is_never_spoken():
    # 16 Nov 2026, food_03: "NOTES: refund missing item...; amount ₹380" was read to the caller.
    view, speaker, _ = talker()
    speaker.given("NOTES: refund missing item; order no 4512; amount ₹380")
    speech = speaker.prepare("NOTES: refund missing item; order no 4512; amount ₹380.")
    assert speech.safe and "NOTES" not in speech.spoken and speech.why == "a NOTES line, not a reply"
