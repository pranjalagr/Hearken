"""Agent C: the team robot (docs/agent_c_design.md). Ears -> Talker -> voice, with a
Thinker and a Guard beside the pipeline.

It shares the lab's ears, voice, tools, greeting, call box, diary, hang-up and brain
cap with Agent A, so a comparison measures the design, not the parts. What differs:
- the Talker holds no backend tool: its one tool is ask_thinker (decision 7);
- the caller's receipts ("played", "cleared") and keys are read, not dropped:
  the Guard times a yes from them (decision 22).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.sarvam.stt import SarvamRealtimeSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.transcriptions.language import Language

from agents.agent_c.desk import JobDesk
from agents.agent_c.job import Job, Turns
from agents.agent_c.notebook import Note, Notebook
from agents.agent_c.settings import AgentCSettings
from agents.agent_c.talker.ask_thinker import ASK_THINKER_SCHEMA, AskThinker
from agents.agent_c.talker.feed import NoteFeed
from agents.agent_c.talker.language import EarsLanguage, StickyLanguage
from agents.agent_c.talker.mouth import Mouth
from agents.agent_c.talker.note_view import NoteView
from agents.agent_c.talker.speaker import Speaker
from agents.agent_c.talker.speech_watch import SpeechWatch
from agents.agent_c.thinker.brain import GeminiThinker, Step, ThinkerBrain
from agents.agent_c.thinker.loop import Thinker, ThinkerLimits
from agents.common.call_box import CallBox, CallBoxSettings, CallResult
from agents.common.diary import NO_DIARY, Diary
from agents.common.pipecat_brain_cap import BRAIN_CAP, BrainCap
from agents.common.pipecat_diary import DiaryObserver
from agents.common.pipecat_line import LineTransport
from agents.common.tools import Tools
from common.google import google_client
from common.wire.protocol import PIECE_MS, SAMPLE_RATE, Cleared, End, Key, Played
from common.wire.robot_line import CallerAudio, LineClosed, RobotLine
from engine.contract.spoken import SpokenWords
from engine.service.tool_catalog import CONFIRM, ToolSpec

BrainMaker = Callable[[Sequence[ToolSpec]], ThinkerBrain]
WORDS = Path(__file__).resolve().parent / "talker" / "words.yaml"


@dataclass(frozen=True)
class Keys:
    gemini: str
    sarvam: str


def talker_tools() -> ToolsSchema:
    """The Talker's whole tool list. No backend tool, no confirm, no Guard door: it can
    only ask (decision 7). A test holds this to exactly one tool."""
    return ToolsSchema(standard_tools=[ASK_THINKER_SCHEMA])


def thinker_tools(catalog: Sequence[ToolSpec]) -> list[ToolSpec]:
    """The backend's tools for this call's mode, less confirm: code presses confirm, the
    instant the Guard gives a ticket (decision 17). Agent C runs in safe mode, where
    every risky action is a prepare, so the Thinker can change nothing by itself."""
    names = {spec.name for spec in catalog}
    if CONFIRM not in names:
        raise ValueError("Agent C needs safe mode: without confirm, the Thinker's tools could change orders unchecked")
    return [spec for spec in catalog if spec.name != CONFIRM]


class CountedBrain:
    """Counts each Thinker request against the call's brain cap, as the lab counts
    the Talker's (Talker, Thinker and judge together, decision 19)."""

    def __init__(self, brain: ThinkerBrain, cap: BrainCap) -> None:
        self._brain = brain
        self._cap = cap

    async def step(self, news, results) -> Step:
        await self._cap.add(1)
        return await self._brain.step(news, results)


class AgentC:
    def __init__(self, settings: AgentCSettings, keys: Keys, tools: Tools, *,
                 box_settings: CallBoxSettings = CallBoxSettings(), diary: Diary = NO_DIARY,
                 max_brain_requests: int = 40, brain_maker: BrainMaker | None = None,
                 words: SpokenWords | None = None) -> None:
        self._settings = settings
        self._keys = keys
        self._tools = tools
        self._box_settings = box_settings
        self._diary = diary
        self._max_brain_requests = max_brain_requests  # the lab passes its own (lab/settings.yaml)
        self._brain_maker = brain_maker or self._gemini_thinker
        self._words = words or SpokenWords()  # the call's pack's spoken words (cli/agent_c.py)
        self._limits = ThinkerLimits.from_settings(settings.section("thinker_limits"))
        self._heard_pieces = 0
        self.result: CallResult | None = None

    async def __call__(self, line: RobotLine) -> None:
        call = CallParts(self._settings, self._diary, self._words)
        box = CallBox(self._box_settings, on_crash=call.desk.on_crash)
        try:
            self.result = await box.run(lambda b: self._main(line, b, call))
        finally:
            # The ears are paid by the minute of caller audio sent to them.
            self._diary.cost(self._settings.ears["model"], round(self._heard_pieces * PIECE_MS / 60000, 3), "minutes")

    async def _main(self, line: RobotLine, box: CallBox, call: CallParts) -> None:
        transport = LineTransport(line)
        cap = BrainCap(self._max_brain_requests, lambda n: self._cap_reached(line, box, n))
        allowed = thinker_tools(await self._tools.catalog())
        brain = CountedBrain(self._brain_maker(allowed), cap)
        names = {spec.name for spec in allowed}

        def start_thinker(job: Job, desk: JobDesk) -> None:
            thinker = Thinker(job, call.notebook, brain, self._tools, names, desk, self._limits, self._diary)
            box.start("thinker", thinker.run(), crash_safe=True)

        call.desk.start_thinker = start_thinker
        task = PipelineTask(self.pipeline(transport, call),
                            params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE,
                                                  enable_metrics=True, enable_usage_metrics=True),
                            observers=[DiaryObserver(self._diary), cap, EarsLanguage(call.language),
                                       SpeechWatch(call.feed_replied, call.feed_speech_ended)])

        async def tell_talker(text: str) -> None:
            self._diary.decision("notes to talker", text=text)
            await task.queue_frames([LLMMessagesAppendFrame([{"role": "user", "content": text}], run_llm=True)])

        call.start_feed(tell_talker, box)
        box.start("ears", self._listen(line, transport, box))
        # Every robot opens the same way, with a fixed line (decision 13).
        await task.queue_frames([TTSSpeakFrame(self._settings.greeting)])
        await PipelineRunner(handle_sigint=False).run(task)

    async def _cap_reached(self, line: RobotLine, box: CallBox, requests: int) -> None:
        # As Agent A: the wire first, so the lab records why the call ended, then the box.
        self._diary.decision("hang up", reason=BRAIN_CAP, requests=requests)
        await line.end(BRAIN_CAP)
        box.end(BRAIN_CAP)

    def _gemini_thinker(self, tools: Sequence[ToolSpec]) -> ThinkerBrain:
        client = google_client(self._keys.gemini, self._limits.step_s)
        return GeminiThinker(client, self._settings.thinker["model"], self._settings.thinker_prompt, tools,
                             on_usage=self._thinker_usage)

    def _thinker_usage(self, prompt_tokens: int, answer_tokens: int) -> None:
        model = self._settings.thinker["model"]
        self._diary.cost(model, prompt_tokens, "input tokens", by="thinker")
        self._diary.cost(model, answer_tokens, "output tokens", by="thinker")

    def pipeline(self, transport: LineTransport, call: CallParts) -> Pipeline:
        s = self._settings
        ears = SarvamRealtimeSTTService(
            api_key=self._keys.sarvam, sample_rate=SAMPLE_RATE,
            settings=SarvamRealtimeSTTService.Settings(model=s.ears["model"], language=Language(s.ears["language"]),
                                                       mode=s.ears["mode"], stream_type=s.ears["stream_type"],
                                                       silence_duration_ms=s.ears["silence_ms"]))
        talker = GoogleLLMService(
            api_key=self._keys.gemini,
            settings=GoogleLLMService.Settings(
                model=s.talker["model"], system_instruction=s.talker_prompt, temperature=s.talker["temperature"],
                thinking=GoogleLLMService.ThinkingConfig(thinking_level="minimal")))
        AskThinker(call.desk, self._diary).connect(talker)
        voice = SarvamTTSService(api_key=self._keys.sarvam, sample_rate=SAMPLE_RATE,
                                 settings=SarvamTTSService.Settings(model=s.voice["model"], voice=s.voice["speaker"],
                                                                    language=Language(s.voice["language"])))
        context = LLMContext(tools=talker_tools())
        turns = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))

        @turns.user().event_handler("on_user_turn_started")
        async def began(aggregator, strategy) -> None:
            call.desk.caller_began()

        @turns.user().event_handler("on_user_turn_stopped")
        async def finished(aggregator, strategy, message) -> None:
            call.caller_finished(message.content or "")

        mouth = Mouth(call.speaker, context, self._diary)
        return Pipeline([transport.input(), ears, turns.user(), talker, mouth, voice, transport.output(), turns.assistant()])

    async def _listen(self, line: RobotLine, transport: LineTransport, box: CallBox) -> None:
        async for event in line.events():
            if isinstance(event, CallerAudio):
                box.heard_audio()
                self._heard_pieces += 1
                await transport.input().feed(event.piece)
            elif isinstance(event, (Played, Cleared, Key)):
                self.on_receipt(event)
            elif isinstance(event, End):
                box.end(f"caller:{event.reason}")
                return
            elif isinstance(event, LineClosed):
                box.end("line closed")
                return

    def on_receipt(self, event: Played | Cleared | Key) -> None:
        """The caller's receipts and keys: the Guard reads these from build step 4."""


class CallParts:
    """One call's own pieces: its notebook, turns, job desk and note feed."""

    def __init__(self, settings: AgentCSettings, diary: Diary, pack_words: SpokenWords) -> None:
        self.notebook = Notebook(int(settings.section("notebook")["max_notes"]))
        self.turns = Turns()
        self.desk = JobDesk(self.notebook, self.turns, self._no_thinker, diary)
        self._feed_wait_s = float(settings.section("feed")["wait_s"])
        self.feed: NoteFeed | None = None
        # The pack's words (statuses, money) and Agent C's own (safe line, results), as one.
        self.words = SpokenWords({**pack_words.groups, **SpokenWords.from_yaml(WORDS).groups})
        self.language = StickyLanguage(int(settings.section("language")["min_words"]))
        self.view = NoteView(self.words)
        self.speaker = Speaker(self.words, self.view, self.language, int(settings.talker["max_sentences"]))

    @staticmethod
    def _no_thinker(job: Job, desk: JobDesk) -> None:
        raise RuntimeError("the Thinker is started only once the call has begun")

    def caller_finished(self, words: str) -> None:
        self.language.turn_finished(words)
        self.speaker.heard(words)
        self.desk.caller_finished(words)

    def notes_line(self, notes: list[Note]) -> str:
        """The Talker's view of new notes, with slots; the speaker learns it too, so the
        number check knows every number the Talker was shown."""
        line = "NOTES: " + "; ".join(self.view.line(note, self.language.value) for note in notes)
        self.speaker.given(line)
        return line

    def start_feed(self, push, box: CallBox) -> None:
        self.feed = NoteFeed(self.notebook, self.desk, self.turns, push, self._feed_wait_s, self.notes_line)
        box.start("feed", self.feed.run())

    def feed_replied(self) -> None:
        if self.feed is not None:
            self.feed.talker_replied()

    def feed_speech_ended(self, cut: bool) -> None:
        if self.feed is not None:
            self.feed.speech_ended(cut)

