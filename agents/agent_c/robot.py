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
from typing import Any
from dataclasses import dataclass
from pathlib import Path

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesAppendFrame, TTSSpeakFrame
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
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from agents.agent_c.desk import JobDesk
from agents.agent_c.guard.guard import DONE, Door, Guard, GuardSettings
from agents.agent_c.guard.judge import TimedJudge, YesJudge, gemini_ask
from agents.agent_c.guard.partial_words import PartialWords
from agents.agent_c.hand_over import HANDED_OVER, HandOver
from agents.agent_c.job import Job, Turns
from agents.agent_c.notebook import CALLER, RESULT, Note, Notebook
from agents.agent_c.settings import AgentCSettings
from agents.agent_c.stop_check import OnnxStopModel, StopListener, StopModel
from agents.agent_c.stop_switch import StopSwitch
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
from agents.common.pipecat_hangup import END_CALL_SCHEMA, HangUp
from agents.common.pipecat_line import LineTransport
from agents.common.tools import Tools
from common.google import GOOGLE_MIN_TIMEOUT_S, ClientRing, google_client
from common.wire.protocol import PIECE_MS, SAMPLE_RATE, Cleared, End, Key, Played
from common.wire.robot_line import CallerAudio, LineClosed, RobotLine
from engine.contract.spoken import SpokenWords
from engine.service.replies import REFUSED, ToolReply
from engine.service.tool_catalog import CONFIRM, ToolSpec

BrainMaker = Callable[[Sequence[ToolSpec]], ThinkerBrain]
WORDS = Path(__file__).resolve().parent / "talker" / "words.yaml"
JUDGE_PROMPT = Path(__file__).resolve().parent / "guard" / "prompts" / "judge.md"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Keys:
    gemini: str  # the Talker's (Pipecat's Gemini service holds one key)
    sarvam: str
    # Every working Gemini key when the lab rotates them (lab/key_order.yaml): the Thinker's
    # and judge's requests take turns over all but the Talker's, so the Talker keeps its own
    # 15 a minute. Empty: everything on `gemini`.
    gemini_pool: tuple[str, ...] = ()

    def back_office(self) -> list[str]:
        """The keys the Thinker and judge take turns over."""
        others = [key for key in self.gemini_pool if key != self.gemini]
        return others or [self.gemini]


def talker_tools() -> ToolsSchema:
    """The Talker's whole tool list. No backend tool, no confirm, no Guard door: it can
    only ask (decision 7) and hang up after a goodbye (decision 13). Without end_call a
    solved call ran on until the caller gave up: Agent A spent 50 brain requests in one
    call that way (15 Nov 2026). A test holds this to exactly these two tools."""
    return ToolsSchema(standard_tools=[ASK_THINKER_SCHEMA, END_CALL_SCHEMA])


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
                 words: SpokenWords | None = None, door: Door | None = None, judge: YesJudge | None = None,
                 stop_model: StopModel | None = None) -> None:
        self._settings = settings
        self._keys = keys
        self._tools = tools
        self._box_settings = box_settings
        self._diary = diary
        self._max_brain_requests = max_brain_requests  # the lab passes its own (lab/settings.yaml)
        self._brain_maker = brain_maker or self._gemini_thinker
        self._words = words or SpokenWords()  # the call's pack's spoken words (cli/agent_c.py)
        # The Guard's private door, from cli/agent_c.py: plain code only, never a tool (decision 7).
        self._door = door
        self._judge = judge
        self._guard_settings = GuardSettings.from_settings(settings.section("guard"))
        self._stop_model = stop_model if stop_model is not None else load_stop_model(settings, diary)
        self._limits = ThinkerLimits.from_settings(settings.section("thinker_limits"))
        self._heard_pieces = 0
        self.result: CallResult | None = None

    async def __call__(self, line: RobotLine) -> None:
        call = CallParts(self._settings, self._diary, self._words, self._stop_model)
        box = CallBox(self._box_settings, on_crash=call.crashed)
        try:
            self.result = await box.run(lambda b: self._main(line, b, call))
        finally:
            # The ears are paid by the minute of caller audio sent to them.
            self._diary.cost(self._settings.ears["model"], round(self._heard_pieces * PIECE_MS / 60000, 3), "minutes")

    async def _main(self, line: RobotLine, box: CallBox, call: CallParts) -> None:
        transport = LineTransport(line)
        call.speech.output = transport.output()  # the voice's words count as they play (the Guard's mark)
        cap = BrainCap(self._max_brain_requests, lambda n: self._cap_reached(line, box, n))
        allowed = thinker_tools(await self._tools.catalog())
        brain = CountedBrain(self._brain_maker(allowed), cap)
        names = {spec.name for spec in allowed}

        def start_thinker(job: Job, desk: JobDesk) -> None:
            thinker = Thinker(job, call.notebook, brain, self._tools, names, desk, self._limits, self._diary,
                              hand_over_tool=self._settings.raw["hand_over_tool"])
            box.start("thinker", thinker.run(), crash_safe=True)

        call.desk.start_thinker = start_thinker
        task = PipelineTask(self.pipeline(transport, call),
                            params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE,
                                                  enable_metrics=True, enable_usage_metrics=True),
                            observers=[DiaryObserver(self._diary), cap, EarsLanguage(call.language),
                                       call.speech, PartialWords(call.partial_words)])

        async def tell_talker(text: str) -> None:
            self._diary.decision("notes to talker", text=text)
            await task.queue_frames([LLMMessagesAppendFrame([{"role": "user", "content": text}], run_llm=True)])

        async def say(text: str) -> None:
            """A line code wrote (the yes-question): straight to the voice, and into the
            Talker's history, so it knows what the caller heard."""
            call.speaker.heard(text)  # numbers the caller heard from the backend may be repeated
            await task.queue_frames([TTSSpeakFrame(text)])

        async def say_and_end(text: str) -> None:
            """The hand-over line, then the end: the pipeline ends once it has played."""
            call.speaker.heard(text)
            await task.queue_frames([TTSSpeakFrame(text), EndFrame()])

        call.hand_over = HandOver(self._tools, self._settings.raw["hand_over_tool"], call.desk, say_and_end,
                                  lambda: call.words.phrase("line", "hand_over", call.language.value), self._diary)
        call.box = box
        call.desk.on_hand_over = call.thinker_hand_over
        if self._door is not None:
            call.guard = Guard(self._door, self._tools, box, call.notebook, call.desk, self._yes_judge(cap), call.words,
                               call.language, say, line.send_mark, self._guard_settings, self._diary,
                               hand_over=call.hand_over.now)
            call.desk.on_prepared = call.guard.prepared
        call.start_feed(tell_talker, box)
        box.start("ears", self._listen(line, transport, box, call))
        # Every robot opens the same way, with a fixed line (decision 13).
        await task.queue_frames([TTSSpeakFrame(self._settings.greeting)])
        await PipelineRunner(handle_sigint=False).run(task)
        if call.hand_over.done:
            # Handed over and the line has played: say so on the wire (decision 13).
            await line.end(HANDED_OVER)
        elif call.hang_up.reason is not None:
            # The Talker said goodbye and hung up; the goodbye has played.
            await line.end(call.hang_up.reason)

    async def _cap_reached(self, line: RobotLine, box: CallBox, requests: int) -> None:
        # As Agent A: the wire first, so the lab records why the call ended, then the box.
        self._diary.decision("hang up", reason=BRAIN_CAP, requests=requests)
        await line.end(BRAIN_CAP)
        box.end(BRAIN_CAP)

    def _yes_judge(self, cap: BrainCap) -> YesJudge:
        if self._judge is not None:
            return self._judge
        section = self._settings.section("guard")
        # Gemini's own minimum deadline; the judge's ~1 s limit is held by TimedJudge.
        client = self._ring(max(float(section["judge_s"]), GOOGLE_MIN_TIMEOUT_S))
        ask = gemini_ask(client, section["judge_model"], JUDGE_PROMPT.read_text(encoding="utf-8"))
        return TimedJudge(ask, float(section["judge_s"]), lambda: cap.add(1), self._diary.error)

    def _ring(self, timeout_s: float) -> ClientRing:
        """One client per back-office key, taking turns request by request."""
        return ClientRing([google_client(key, timeout_s) for key in self._keys.back_office()])

    def _gemini_thinker(self, tools: Sequence[ToolSpec]) -> ThinkerBrain:
        # Gemini's own minimum deadline; the ~8 s step limit is held by the Thinker loop (try_live).
        client = self._ring(max(self._limits.step_s, GOOGLE_MIN_TIMEOUT_S))
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
        call.hang_up.connect(talker)
        voice = SarvamTTSService(api_key=self._keys.sarvam, sample_rate=SAMPLE_RATE,
                                 settings=SarvamTTSService.Settings(model=s.voice["model"], voice=s.voice["speaker"],
                                                                    language=Language(s.voice["language"])))
        context = LLMContext(tools=talker_tools())
        user_params = LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
        if call.stop_listener is not None:
            # Decision 26: the caller's sound never stops the voice by itself; only the
            # stop-or-keep model does, through the stop switch.
            user_params = LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer(), user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy(enable_interruptions=False),
                       TranscriptionUserTurnStartStrategy(enable_interruptions=False)]))
        turns = LLMContextAggregatorPair(context, user_params=user_params)

        @turns.user().event_handler("on_user_turn_started")
        async def began(aggregator, strategy) -> None:
            call.caller_began()

        @turns.user().event_handler("on_user_turn_stopped")
        async def finished(aggregator, strategy, message) -> None:
            call.caller_finished(message.content or "")

        mouth = Mouth(call.speaker, context, self._diary)
        return Pipeline([transport.input(), call.switch, ears, turns.user(), talker, mouth, voice, transport.output(),
                         turns.assistant()])

    async def _listen(self, line: RobotLine, transport: LineTransport, box: CallBox, call: CallParts) -> None:
        async for event in line.events():
            if isinstance(event, CallerAudio):
                box.heard_audio()
                self._heard_pieces += 1
                call.caller_ms = event.caller_ms
                await transport.input().feed(event.piece)
                await call.heard_piece(event.piece)
            elif isinstance(event, (Played, Cleared, Key)):
                call.receipt(event)
            elif isinstance(event, End):
                box.end(f"caller:{event.reason}")
                return
            elif isinstance(event, LineClosed):
                box.end("line closed")
                return


class CallParts:
    """One call's own pieces: its notebook, turns, job desk and note feed."""

    def __init__(self, settings: AgentCSettings, diary: Diary, pack_words: SpokenWords,
                 stop_model: StopModel | None = None) -> None:
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
        self.guard: Guard | None = None  # set once the call has begun, when the robot holds a door
        self.hand_over: HandOver | None = None  # set once the call has begun
        # The Talker's end_call: refused while the back office still works on the job.
        self.hang_up = HangUp(diary, not_now=lambda: self.desk.why_not_end())
        self.box: CallBox | None = None
        self._diary = diary
        self.switch = StopSwitch(name="stop-switch")
        self.speech = SpeechWatch(self.feed_replied, self.speech_ended)
        self.stop_listener: StopListener | None = None
        if stop_model is not None:
            stop = settings.section("stop_check")
            self.stop_listener = StopListener(stop_model, float(stop["line"]), float(stop["window_s"]), float(stop["every_s"]),
                                              self.stop_talking, lambda ms: diary.error(f"stop check took {ms:.0f} ms"))
        self.caller_ms = 0  # the newest caller piece heard, on the caller's clock (the wire's caller_ms)

    def crashed(self, task: str, error: BaseException) -> None:
        """A crash-safe task (the Thinker) crashed: its job ends with a "failed" note,
        and the caller is handed over, once."""
        self.desk.on_crash(task, error)
        if self.hand_over is not None and self.box is not None:
            self.box.start("hand over", self.hand_over.now("the back office stopped working"))

    @staticmethod
    def _no_thinker(job: Job, desk: JobDesk) -> None:
        raise RuntimeError("the Thinker is started only once the call has begun")

    def caller_began(self) -> None:
        self.desk.caller_began()
        if self.guard is not None:
            self.guard.turn_began(self.caller_ms)

    async def heard_piece(self, piece: bytes) -> None:
        if self.stop_listener is not None:
            await self.stop_listener.heard(piece, self.speech.speaking)

    async def stop_talking(self, chance: float) -> None:
        """The stop-or-keep model said stop: stop the voice, make what was being said
        about old news (a new version), and stop any ticket being confirmed."""
        self._diary.decision("stop talking", chance=round(chance, 3))
        await self.switch.stop_voice()
        self.desk.robot_stopped()
        if self.guard is not None:
            self.guard.stop_ticket("the stop-or-keep model said stop")

    def caller_finished(self, words: str) -> None:
        if self.stop_listener is not None:
            self.stop_listener.turn_finished()
        self.language.turn_finished(words)
        self.speaker.heard(words)
        self.desk.caller_finished(words)
        if self.guard is not None:
            self.guard.turn_finished(words)

    def partial_words(self, words: str) -> None:
        if self.guard is not None:
            self.guard.partial(words)

    def receipt(self, event: Played | Cleared | Key) -> None:
        """The caller's receipts and keys, for the Guard's timing (decision 22)."""
        if self.guard is None:
            return
        if isinstance(event, Played):
            self.guard.played(event.name, event.caller_ms)
        elif isinstance(event, Cleared):
            self.guard.cleared(event.dropped_marks)
        else:
            self.guard.key(event.digit, event.caller_ms)

    async def thinker_hand_over(self, reason: str, order_no: Any = None) -> ToolReply:
        """The Thinker's hand-over, refused while the Guard asks for a yes or confirms one.

        16 Nov 2026, food_01: the caller said "Haan sahi hai" to the yes-question, the
        Thinker read it, decided it must confirm, found no confirm tool and handed the
        call over, though its prompt says code confirms. A prompt rule can be skipped, so
        code refuses; the Thinker hears why and can still hand over once the Guard is done.
        """
        if self.guard is not None and (self.guard.asking or self.guard.confirming is not None):
            self._diary.decision("thinker hand over refused", reason=reason)
            return ToolReply(REFUSED, "Not now: the caller is answering the yes-question. Other code confirms on a "
                                      "clear yes and sends a Result: line. Press no tool; wait for that line.")
        # The same mistake a moment later (food_01, 16 Nov 2026): a Thinker step begun during
        # the question finished after the confirm, and handed over a caller whose cancel had
        # just been done. A change done, with no caller words since, needs no human.
        newest = self.notebook.last((RESULT, CALLER))
        if newest is not None and newest.kind == RESULT and newest.data.get("outcome") == DONE:
            self._diary.decision("thinker hand over refused", reason=reason, why="change just done")
            return ToolReply(REFUSED, "Not needed: the change is done (see the Result: line). Press no tool and "
                                      "write one short line saying it is done.")
        return await self.hand_over.now(reason, order_no)

    def notes_line(self, notes: list[Note]) -> str:
        """The Talker's view of new notes, with slots; the speaker learns it too, so the
        number check knows every number the Talker was shown."""
        line = "NOTES: " + "; ".join(self.view.line(note, self.language.value) for note in notes)
        self.speaker.given(line)
        return line

    def start_feed(self, push, box: CallBox) -> None:
        self.feed = NoteFeed(self.notebook, self.desk, self.turns, push, self._feed_wait_s, self.notes_line,
                             hold=lambda: self.guard is not None and self.guard.asking)
        box.start("feed", self.feed.run())

    def feed_replied(self) -> None:
        if self.feed is not None:
            self.feed.talker_replied()

    def speech_ended(self, cut: bool, spoken: str) -> None:
        if self.feed is not None:
            self.feed.speech_ended(cut)
        if self.guard is not None:
            self.guard.speech_ended(cut, spoken)


def load_stop_model(settings: AgentCSettings, diary: Diary) -> StopModel | None:
    """The trained model, if its file is here. It lives outside git, so a fresh copy of
    the project has none: then the robot stops on any caller sound, as Agent A does,
    and says so in its diary."""
    path = PROJECT_ROOT / settings.section("stop_check")["model_path"]
    if not path.exists():
        diary.error(f"no stop-or-keep model at {path}: stopping on any caller sound instead")
        return None
    return OnnxStopModel(path)
