"""The Talker's parts that outlived the two graphs (17 Nov 2026): the sticky call language
and the speech watch the Guard times its yes-question by. The Talker graph itself is
tested in test_talker_graph.py."""

from agents.agent_c.talker.language import ENGLISH, HINGLISH, StickyLanguage, from_ears


def test_the_language_changes_only_on_a_long_enough_turn_in_another_language():
    lang = StickyLanguage(4)
    lang.ears_said = "en-IN"
    assert lang.turn_finished("okay thank you") == HINGLISH
    assert lang.turn_finished("I want to cancel my order") == ENGLISH
    lang.ears_said = "Language.HI_IN"
    assert lang.turn_finished("haan") == ENGLISH
    assert (from_ears("hi-IN"), from_ears(None), from_ears("ta-IN")) == (HINGLISH, None, None)


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
