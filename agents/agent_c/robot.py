"""Agent C: the team robot, as two graphs (decided 17 Nov 2026). Ears -> the Talker graph ->
voice, with the Thinker graph and the Guard beside the pipeline.

It shares the lab's ears, voice, tools, greeting, call box, diary, hang-up and brain cap
with Agent A, so a comparison measures the design, not the parts. What differs:
- No brain sits inside the pipeline. A finished caller turn goes to the Talker graph,
  which decides what to say and says it through the voice (fixed lines, or its AI's
  sentences as they stream).
- When the Talker opens a job, the Thinker graph runs beside the pipeline, in the call
  box, and the two meet only in the notebook.
- The caller's receipts ("played", "cleared") and keys are read: the Guard times a yes from
  them (decision 22), and a typed order number goes to the Talker (decision 27).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VAD_STOP_SECS
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.services.sarvam.stt import SarvamRealtimeSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from agents.agent_c.call_job import CallJob
from agents.agent_c.guard.guard import Door, Guard, GuardSettings
from agents.agent_c.guard.judge import TimedJudge, YesJudge, first_answer, gemini_ask, groq_ask
from agents.agent_c.guard.partial_words import PartialWords
from agents.agent_c.notebook import FAILED, Note, Notebook
from agents.agent_c.settings import AgentCSettings
from agents.agent_c.stop_check import OnnxStopModel, StopListener, StopModel
from agents.agent_c.stop_switch import StopSwitch
from agents.agent_c.talker.graph import TALKER_GRAPH, TalkerParts, TalkerRunner
from agents.agent_c.talker.language import EarsLanguage, StickyLanguage
from agents.agent_c.talker.lines import ClipOut, Clips, clip_lines, clip_name, clips_by_words
from agents.agent_c.talker.quiet import QuietSettings
from agents.agent_c.talker.quiet_watch import CallView, QuietWatch
from agents.agent_c.talker.voice_spelling import spell_for_voice
from agents.agent_c.talker.number_finder import FindNumbers, sarvam_find
from agents.agent_c.talker.speech_watch import SpeechWatch
from agents.agent_c.talker.talker_ai import FailoverTalkerAI, GeminiTalkerAI, GroqTalkerAI, TalkerAI
from agents.agent_c.thinker.graph import THINKER_GRAPH, ThinkerParts, run_job
from agents.agent_c.thinker.address_check import AddressCheck, JevAddressCheck
from agents.agent_c.guard.yes_check import JevYesCheck, YesCheck
from agents.agent_c.thinker.step_brain import GeminiStepBrain, GroqStepBrain, StepBrain, StepRouter
from agents.common.call_box import CallBox, CallBoxSettings, CallResult
from agents.common.diary import NO_DIARY, Diary
from agents.common.pipecat_brain_cap import BRAIN_CAP, BrainCap
from agents.common.pipecat_diary import DiaryObserver
from agents.common.pipecat_line import LineTransport
from agents.common.tools import Tools
from agents.common.ai_log import AiLog
from common.google import GOOGLE_MIN_TIMEOUT_S, ClientRing, google_client
from common.groq import groq_client
from common.sarvam import sarvam_client
from common.wire.protocol import PIECE_MS, SAMPLE_RATE, Cleared, End, Key, Played
from common.wire.robot_line import CallerAudio, LineClosed, RobotLine
from engine.contract.spoken import SpokenWords
from engine.service.tool_catalog import CONFIRM, ToolSpec

WORDS = Path(__file__).resolve().parent / "talker" / "words.yaml"
JUDGE_PROMPT = Path(__file__).resolve().parent / "guard" / "prompts" / "judge.md"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROQ = "groq"  # a provider name in settings.yaml (talker.provider, thinker.provider, guard.judge_provider)
JEV = "jev"  # the address check's provider name in settings.yaml (address_check.provider)
HANDED_OVER = "handed_over"


@dataclass(frozen=True)
class Keys:
    # Every key field is left out of repr (18 Nov 2026): a printed Keys, in an error or a log,
    # must never show a key.
    gemini: str = field(repr=False)  # the Talker's own key
    sarvam: str = field(repr=False)
    # Every working Gemini key when the lab rotates them (lab/key_order.yaml): the Thinker's
    # and judge's requests take turns over all but the Talker's, so the Talker keeps its own
    # 15 a minute. Empty: everything on `gemini`.
    gemini_pool: tuple[str, ...] = field(default=(), repr=False)
    # Every Groq key, the lab's current one first (lab/key_order.yaml): the Thinker's "decide"
    # step starts on the first and moves on when one is refused (17 Nov 2026). Empty: "decide"
    # stays on Gemini like every other step.
    groq_pool: tuple[str, ...] = field(default=(), repr=False)
    # {key: its .env name} (common/env.py key_names), so the AI request log can say which key
    # a request used without writing the key (18 Nov 2026). Never printed: repr=False.
    names: Mapping[str, str] = field(default_factory=dict, repr=False)
    # The address check's key (TypeSafe Jev, 19 Nov 2026). Empty: no check.
    jev: str = field(default="", repr=False)

    def back_office(self) -> list[str]:
        """The keys the Thinker and judge take turns over."""
        others = [key for key in self.gemini_pool if key != self.gemini]
        return others or [self.gemini]

    def name(self, key: str) -> str:
        """The key's .env name, or "" when it has none (a test key)."""
        return self.names.get(key, "")

    def all(self) -> list[str]:
        """Every key held, to scrub out of anything logged."""
        return [self.gemini, self.sarvam, *self.gemini_pool, *self.groq_pool, self.jev]


def thinker_tools(catalog: Sequence[ToolSpec]) -> list[ToolSpec]:
    """The backend's tools for this call's mode, less confirm: code presses confirm, the
    instant the Guard gives a ticket (decision 17). Agent C runs in safe mode, where
    every risky action is a prepare, so the Thinker can change nothing by itself."""
    names = {spec.name for spec in catalog}
    if CONFIRM not in names:
        raise ValueError("Agent C needs safe mode: without confirm, the Thinker's tools could change orders unchecked")
    return [spec for spec in catalog if spec.name != CONFIRM]



_TLS: list[Any] = []


def _shared_tls() -> Any:
    """One secure-connection setup for the program, made on first use and kept."""
    if not _TLS:
        import ssl
        _TLS.append(ssl.create_default_context())
    return _TLS[0]

class CountedTalkerAI:
    """Counts each Talker AI request against the call's brain cap (decision 19)."""

    def __init__(self, ai: TalkerAI, cap: BrainCap) -> None:
        self._ai, self._cap = ai, cap

    async def reply(self, step, facts, exits, speak):
        await self._cap.add(1)
        return await self._ai.reply(step, facts, exits, speak)


class CountedStepBrain:
    """Counts each Thinker AI request against the call's brain cap (decision 19)."""

    def __init__(self, brain: StepBrain, cap: BrainCap) -> None:
        self._brain, self._cap = brain, cap

    async def ask(self, step, facts):
        await self._cap.add(1)
        return await self._brain.ask(step, facts)


class AgentC:
    def __init__(self, settings: AgentCSettings, keys: Keys, tools: Tools, *,
                 box_settings: CallBoxSettings = CallBoxSettings(), diary: Diary = NO_DIARY,
                 max_brain_requests: int = 40, words: SpokenWords | None = None, door: Door | None = None,
                 judge: YesJudge | None = None, stop_model: StopModel | None = None,
                 talker_ai: TalkerAI | None = None, thinker_ai: StepBrain | None = None,
                 find_numbers: FindNumbers | None = None) -> None:
        self._settings = settings
        self._keys = keys
        self._tools = tools
        self._box_settings = box_settings
        self._diary = diary
        # One diary row per AI request (service, key name, time, outcome, the service's reason;
        # 18 Nov 2026: 17 Talker timeouts in one run, and nothing said which key or why).
        self._ai_log = AiLog(diary.decision, secrets=keys.all())
        self._max_brain_requests = max_brain_requests  # the lab passes its own (lab/settings.yaml)
        self._words = words or SpokenWords()  # the call's pack's spoken words (cli/agent_c.py)
        # The Guard's private door, from cli/agent_c.py: plain code only, never a tool (decision 7).
        self._door = door
        self._judge = judge
        # The AIs and the finder; tests give fakes, the real robot makes them per call.
        self._talker_ai = talker_ai
        self._thinker_ai = thinker_ai
        self._find_numbers = find_numbers
        self._guard_settings = GuardSettings.from_settings(settings.section("guard"))
        self._stop_model = stop_model if stop_model is not None else load_stop_model(settings, diary)
        self._clips = load_clips(settings, self._words, diary)  # read once, never during a call
        self._heard_pieces = 0
        self.result: CallResult | None = None

    async def __call__(self, line: RobotLine) -> None:
        call = CallParts(self._settings, self._diary, self._words, self._stop_model, self._clips)
        box = CallBox(self._box_settings, on_crash=call.crashed)
        try:
            self.result = await box.run(lambda b: self._main(line, b, call))
        finally:
            if call.finder_client is not None:
                await call.finder_client.aclose()
            # The ears are paid by the minute of caller audio sent to them.
            self._diary.cost(self._settings.ears["model"], round(self._heard_pieces * PIECE_MS / 60000, 3), "minutes")

    async def _main(self, line: RobotLine, box: CallBox, call: CallParts) -> None:
        """One call: build the pipeline, the Talker graph and the Guard, then run until the
        Talker says goodbye or hands over (or the line or the box ends it)."""
        transport = LineTransport(line)
        call.speech.output = transport.output()  # the voice's words count as they play (the Guard's mark)
        cap = BrainCap(self._max_brain_requests, lambda n: self._cap_reached(line, box, n))
        menu = thinker_tools(await self._tools.catalog())
        task = PipelineTask(self.pipeline(transport, call),
                            params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE,
                                                  enable_metrics=True, enable_usage_metrics=True),
                            observers=[DiaryObserver(self._diary), cap, EarsLanguage(call.language),
                                       call.speech, PartialWords(call.partial_words)])
        self.wire(call, box, line, task, cap, menu)
        box.start("ears", self._listen(line, transport, box, call))
        box.start("talker", call.runner.run())  # not crash-safe: without the Talker the call can't go on
        box.start("quiet", call.quiet.run())  # waiting lines and the silent caller (decisions 11, 13)
        await PipelineRunner(handle_sigint=False).run(task)


    def wire(self, call: CallParts, box: CallBox, line: RobotLine, task: Any, cap: BrainCap,
             menu: Sequence[ToolSpec]) -> None:
        """Join one call's parts: the AIs (counted against the brain cap), the Thinker graph
        started per job, the Talker graph's runner, and the Guard. Kept apart from the
        pipeline, so tests can run a whole call through it without a network."""
        call.box, call.line, call.task, call.tools = box, line, task, self._tools
        talker_ai = CountedTalkerAI(self._talker_ai or self._gemini_talker(), cap)
        thinker_ai = CountedStepBrain(self._thinker_ai or self._gemini_thinker(), cap)
        check_settings = self._settings.section("address_check")
        address_check = self._address_check(check_settings)
        find = self._find_numbers or call.real_finder(self._keys.sarvam, self._settings.section("number_finder"),
                                                        self._ai_log, self._keys.name(self._keys.sarvam))

        def start_thinker() -> None:
            """The Talker opened a job: run its Thinker graph beside the pipeline. A crash
            is caught by the box: a "failed" note, then the hand-over (decision 14)."""
            limits = self._settings.section("thinker_limits")
            parts = ThinkerParts(call.notebook, self._tools, thinker_ai, call.hand_over, menu,
                                 version=lambda: call.job.version, step_s=float(limits["step_s"]),
                                 retries=int(limits["step_retries"]), same_call_limit=int(limits["same_call_limit"]),
                                 ai_retries=int(limits["step_ai_retries"]),
                                 redecide_limit=int(limits["redecide_limit"]),
                                 on_try_failed=lambda step, reason: self._diary.decision("thinker try failed",
                                                                                         step=step, reason=reason),
                                 language=lambda: call.language.value,
                                 on_step=lambda step, answer: self._diary.decision("thinker step", step=step,
                                                                                   answer=answer),
                                 check_address=address_check,
                                 checked_inputs=tuple(check_settings.get("inputs") or ()),
                                 address_line=float(check_settings.get("line", 0.5)))
            seen = call.notebook.newest  # this job's notes start after the job note
            call.thinker = parts

            async def thinker() -> None:
                try:
                    caller_done = await run_job(THINKER_GRAPH, parts, f"{id(call)}-job-{seen}", seen)
                finally:
                    call.thinker = None
                    call.quiet.drop_plan()  # a plan not said by now is old
                call.runner.job_closed(caller_done)

            box.start("thinker", thinker(), crash_safe=True)

        call.quiet = call.make_quiet_watch()
        ladder = self._settings.section("order_ladder")
        talker_parts = TalkerParts(
            call.notebook, call, call.words, talker_ai, find, call.guard_port, start_thinker, call.end_call,
            self._settings.greeting, language=lambda: call.language.value, version=lambda: call.job.version,
            new_turn=call.job.new_turn, last_question=lambda: call.last_question, number_length=int(self._settings.section("number_finder")["number_length"]),
            keypad_s=float(ladder["keypad_wait_s"]), max_sentences=int(self._settings.talker["max_sentences"]),
            reply_s=float(self._settings.section("talker_graph")["reply_s"]), hold_plan=call.quiet.hold_plan,
            on_error=self._diary.error,
            on_step=lambda step, details: self._diary.decision("talker step", step=step, **details))
        call.runner = TalkerRunner(TALKER_GRAPH, talker_parts, f"call-{id(call)}")
        if self._door is not None:
            call.guard = Guard(self._door, self._tools, box, call.notebook, call.job, self._yes_judge(cap), call.words,
                               call.language, call.say_question, line.send_mark, self._guard_settings, self._diary,
                               hand_over=call.hand_over, yes_check=self._yes_check(self._settings.section("yes_check")))

    async def _cap_reached(self, line: RobotLine, box: CallBox, requests: int) -> None:
        # As Agent A: the wire first, so the lab records why the call ended, then the box.
        self._diary.decision("hang up", reason=BRAIN_CAP, requests=requests)
        await line.end(BRAIN_CAP)
        box.end(BRAIN_CAP)

    def _yes_judge(self, cap: BrainCap) -> YesJudge:
        if self._judge is not None:
            return self._judge
        section = self._settings.section("guard")
        racers = int(section["judge_racers"])
        prompt = JUDGE_PROMPT.read_text(encoding="utf-8")
        # The same question on several keys at once, first answer wins: each racer takes
        # the ring's next client, so each goes out on its own key.
        if section.get("judge_provider") == GROQ and self._keys.groq_pool:
            ask = groq_ask(self._groq_ring(float(section["judge_s"])), section["judge_model"], prompt, self._ai_log,
                           section.get("judge_options"))
        else:
            # Gemini's own minimum deadline; the judge's own limit is held by TimedJudge.
            client = self._ring(max(float(section["judge_s"]), GOOGLE_MIN_TIMEOUT_S))
            ask = gemini_ask(client, section["judge_model"], prompt, self._ai_log)
        return TimedJudge(first_answer(ask, racers), float(section["judge_s"]), lambda: cap.add(racers),
                          self._diary.error)

    def _address_check(self, section: dict[str, Any]) -> AddressCheck | None:
        """The address check for this call (thinker/address_check.py), or None: no Jev key, or
        a provider this robot doesn't know. Tests give their own through the Thinker's parts."""
        if section.get("provider") != JEV or not self._keys.jev:
            return None
        limit_s = float(section["check_s"])
        client = httpx.AsyncClient(headers={"Authorization": f"Bearer {self._keys.jev}", "Content-Type": "application/json"},
                                   timeout=limit_s, verify=_shared_tls())
        return JevAddressCheck(client, str(section["model"]), limit_s, self._keys.name(self._keys.jev), self._ai_log,
                               on_error=self._diary.error)

    def _yes_check(self, section: dict[str, Any]) -> YesCheck | None:
        """Jev's "clear yes?" for the Guard (guard/yes_check.py), or None: no Jev key, or a
        provider this robot doesn't know. Then the judge decides alone, as before 20 Nov 2026."""
        if section.get("provider") != JEV or not self._keys.jev:
            return None
        limit_s = float(section["check_s"])
        client = httpx.AsyncClient(headers={"Authorization": f"Bearer {self._keys.jev}", "Content-Type": "application/json"},
                                   timeout=limit_s, verify=_shared_tls())
        return JevYesCheck(client, str(section["model"]), limit_s, float(section["line"]),
                           self._keys.name(self._keys.jev), self._ai_log, on_error=self._diary.error)

    def _groq_ring(self, timeout_s: float) -> ClientRing:
        """One Groq client per key, the lab's current key first, taking turns request by request."""
        keys = list(self._keys.groq_pool)
        return ClientRing([groq_client(key, timeout_s, verify=_shared_tls()) for key in keys],
                          [self._keys.name(key) for key in keys])

    def _ring(self, timeout_s: float) -> ClientRing:
        """One client per back-office key, taking turns request by request."""
        keys = self._keys.back_office()
        return ClientRing([google_client(key, timeout_s) for key in keys], [self._keys.name(key) for key in keys])

    def _gemini_talker(self) -> TalkerAI:
        """The Talker's AI on its own key, then the other keys if that one is refused for
        quota (Gemini's minimum deadline; the graph holds its own limit). On Groq when
        settings.yaml talker.provider says so (18 Nov 2026), over the Groq keys the same way."""
        reply_s = float(self._settings.section("talker_graph")["reply_s"])
        talker = self._settings.talker
        if talker.get("provider") == GROQ and self._keys.groq_pool:
            ais = [GroqTalkerAI(groq_client(key, reply_s, verify=_shared_tls()), talker["model"], self._keys.name(key),
                                self._ai_log, talker.get("options")) for key in self._keys.groq_pool]
            return FailoverTalkerAI(ais, on_switch=lambda why: self._diary.decision("talker key changed", why=why))
        keys = [self._keys.gemini, *[key for key in self._keys.back_office() if key != self._keys.gemini]]
        ais = [GeminiTalkerAI(google_client(key, max(reply_s, GOOGLE_MIN_TIMEOUT_S)), self._settings.talker["model"],
                              self._keys.name(key), self._ai_log)
               for key in keys]
        return FailoverTalkerAI(ais, on_switch=lambda why: self._diary.decision("talker key changed", why=why))

    def _gemini_thinker(self) -> StepBrain:
        """The Thinker's AI: every step on Gemini, taking turns over the back-office keys,
        except "decide", which runs on Groq when the robot holds Groq keys (17 Nov 2026,
        the owner's choice; settings.yaml thinker.decide_model says why)."""
        step_s = float(self._settings.section("thinker_limits")["step_s"])
        thinker = self._settings.thinker
        model = thinker.get("decide_model")
        if not self._keys.groq_pool:
            return GeminiStepBrain(self._ring(max(step_s, GOOGLE_MIN_TIMEOUT_S)), thinker["model"], self._ai_log)
        # One secure-connection setup for every key: making one reads the computer's
        # certificates and freezes the loop for tens of ms (lab/ears_test/services.py).
        tls = _shared_tls()
        names = [self._keys.name(key) for key in self._keys.groq_pool]

        def groq(model_name: str) -> GroqStepBrain:
            clients = [groq_client(key, step_s, verify=tls) for key in self._keys.groq_pool]
            return GroqStepBrain(clients, str(model_name), names=names, log=self._ai_log,
                                 on_switch=lambda why: self._diary.decision("thinker key changed", why=why))

        # The small steps on Groq too when thinker.provider says so (18 Nov 2026): Gemini took
        # 6-8 s a request that evening, over the 10 s step limit with a retry.
        small = (groq(thinker["model"]) if thinker.get("provider") == GROQ
                 else GeminiStepBrain(self._ring(max(step_s, GOOGLE_MIN_TIMEOUT_S)), thinker["model"], self._ai_log))
        return StepRouter(small, {"decide": groq(model)}) if model else small

    def pipeline(self, transport: LineTransport, call: CallParts) -> Pipeline:
        """Ears -> turn detection -> voice. No brain in between: a finished turn goes to the
        Talker graph (call.caller_finished), which speaks through call.say."""
        s = self._settings
        ears = SarvamRealtimeSTTService(
            api_key=self._keys.sarvam, sample_rate=SAMPLE_RATE,
            settings=SarvamRealtimeSTTService.Settings(model=s.ears["model"], language=Language(s.ears["language"]),
                                                       mode=s.ears["mode"], stream_type=s.ears["stream_type"],
                                                       silence_duration_ms=s.ears["silence_ms"]))
        voice = SarvamTTSService(api_key=self._keys.sarvam, sample_rate=SAMPLE_RATE,
                                 settings=SarvamTTSService.Settings(model=s.voice["model"], voice=s.voice["speaker"],
                                                                    language=Language(s.voice["language"])))
        # The turn ends after this much quiet in total. The voice detector already waits
        # VAD_STOP_SECS of it before saying "stopped", and the timer starts only then.
        # 18 Nov 2026 (the owner's choice, decision 25 on hold): until now Pipecat's default
        # Smart Turn decided, and on 187 real lab caller lines it cut the caller off at 65
        # of 90 pauses inside a sentence ("House 74, Sector 15," | "Gurgaon").
        quiet_s = float(s.section("turn_end")["quiet_s"])
        stop = [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=max(0.0, quiet_s - VAD_STOP_SECS))]
        user_params = LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer(),
                                              user_turn_strategies=UserTurnStrategies(stop=stop))
        if call.stop_listener is not None:
            # Decision 26: the caller's sound never stops the voice by itself; only the
            # stop-or-keep model does, through the stop switch.
            user_params = LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer(), user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy(enable_interruptions=False),
                       TranscriptionUserTurnStartStrategy(enable_interruptions=False)], stop=stop))
        # Pipecat's turn collector, kept only to know when a caller turn begins and ends.
        turns = LLMContextAggregatorPair(LLMContext(), user_params=user_params)

        @turns.user().event_handler("on_user_turn_started")
        async def began(aggregator, strategy) -> None:
            call.caller_began()

        @turns.user().event_handler("on_user_turn_stopped")
        async def finished(aggregator, strategy, message) -> None:
            call.caller_finished(message.content or "")

        # The clip player sits after the voice: the ears must never hear the robot's own clip.
        return Pipeline([transport.input(), call.switch, ears, turns.user(), voice, call.clip_out, transport.output()])

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


class GuardPort:
    """What the Talker graph sees of the Guard. The Guard reads caller turns and keys itself,
    straight from the call (CallParts), with their timing; the Talker only hands it the
    prepared change."""

    def __init__(self, call: CallParts) -> None:
        self._call = call

    async def prepared(self, note: Note) -> None:
        if self._call.guard is not None:
            await self._call.guard.prepared(note)

    async def answer(self, words: str, start_ms: int | None) -> None:
        pass  # the Guard already has it (CallParts.caller_finished)

    def key(self, digit: str, caller_ms: int) -> None:
        pass  # the Guard already has it (CallParts.receipt)


class CallParts:
    """One call's own pieces: its notebook, job version, words, language, Guard, voice and
    the Talker graph's runner. Also the Talker graph's voice (say)."""

    def __init__(self, settings: AgentCSettings, diary: Diary, pack_words: SpokenWords,
                 stop_model: StopModel | None = None, clips: dict[str, bytes] | None = None) -> None:
        self._settings = settings
        self._diary = diary
        self.notebook = Notebook(int(settings.section("notebook")["max_notes"]))
        self.job = CallJob()
        # The pack's words (statuses, money) and Agent C's own (lines, results), as one.
        self.words = SpokenWords({**pack_words.groups, **SpokenWords.from_yaml(WORDS).groups})
        self.language = StickyLanguage(int(settings.section("language")["min_words"]))
        self.guard: Guard | None = None  # set once the call has begun, when the robot holds a door
        self.guard_port = GuardPort(self)
        self.runner: TalkerRunner | None = None
        self.box: CallBox | None = None
        self.line: RobotLine | None = None
        self.task: PipelineTask | None = None
        self.finder_client: httpx.AsyncClient | None = None
        self.tools: Tools | None = None  # the backend's tools, for the hand-over
        self._spoken_waiters: list[tuple[str, asyncio.Future]] = []  # lines waiting to be heard to the end
        self._handed_over = False
        self._last_line = ""
        # The robot's last question, for the number finder (20 Nov 2026, food_08): a bare
        # "Paintalis saintalis" was tagged "other" five times because the finder never knew the
        # robot had just asked for the order number. A question, not the last line: fillers like
        # "Ji, ek second." come between the question and the answer.
        self.last_question = ""
        self._last_said_at = -1e9  # when the last line was sent to the voice (the loop's clock)
        self._live_sent_at = -1e9  # when the last line went to the live voice (not a clip)
        self._clip_after_live = -1e9  # when a clip last waited for the live line before it
        self._speech_ended_at = -1e9  # when the voice last stopped (the loop's clock)
        self.clips = clips or {}  # each fixed line's pre-made clip, by its words
        self.clip_out = ClipOut(name="clip-out")
        self.caller_talking = False
        self.thinker: ThinkerParts | None = None  # the running job's Thinker graph parts
        self.quiet: QuietWatch | None = None  # set once the call has begun
        self.switch = StopSwitch(name="stop-switch")
        self.speech = SpeechWatch(lambda: None, self.speech_ended)
        self.stop_listener: StopListener | None = None
        if stop_model is not None:
            stop = settings.section("stop_check")
            self.stop_listener = StopListener(stop_model, float(stop["line"]), float(stop["window_s"]), float(stop["every_s"]),
                                              self.stop_talking, lambda ms: diary.error(f"stop check took {ms:.0f} ms"))
        self.caller_ms = 0  # the newest caller piece heard, on the caller's clock (the wire's caller_ms)

    # ---- the voice -----------------------------------------------------------------------

    async def say(self, text: str, again_if_cut: bool = False) -> None:
        """The Talker graph's voice: one line to the caller, written in the diary as said
        (the scorer reads it). Spelled for the voice first ("main" -> "mai", 2 Nov 2026),
        and played from its pre-made clip when it has one. A line that must be heard is
        said once more if cut."""
        spoken = spell_for_voice(text, self.language.value)
        # Decision 8: the robot never speaks over the caller; a line waits until they finish
        # (at most question_wait_s). Missed in the step 5 rewrite: run 2 (17 Nov 2026) cut the
        # caller off 23.6 times per 100 turns, and a cut "yes" in food_11 didn't count.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(self._settings.section("speed")["question_wait_s"])
        while self.caller_talking and loop.time() < deadline:
            await asyncio.sleep(0.05)
        await self.say_plain(spoken, said=text)
        if again_if_cut and self.box is not None:
            self.box.start("say again if cut", self._again_if_cut(spoken))

    async def say_plain(self, text: str, said: str | None = None, clip: bool = True) -> None:
        """Straight to the voice as written: the clip if there is one, else the live voice.
        The Guard's yes-question comes this way unspelled, as before: its mark compares the
        question's last words with what the voice said."""
        self._diary.said(said or text, filled=text)
        self._last_line = text
        if "?" in text:
            self.last_question = said or text
        loop = asyncio.get_running_loop()
        sound = self.clips.get(text) if clip else None
        if sound is not None:
            await self._after_live_line()
            self._last_said_at = loop.time()
            await self.clip_out.play(sound, text)
        elif self.task is not None:
            self._last_said_at = self._live_sent_at = loop.time()
            await self.task.queue_frames([TTSSpeakFrame(text)])

    async def _after_live_line(self) -> None:
        """A clip goes out at once, but a live line takes ~1 s to be made: a clip sent right
        after one would play first. 18 Nov 2026, food_01: the caller heard "Aur kuch madad kar
        sakta hoon?" (a clip) before "Aapka order cancel ho gaya hai, aur ₹450 ka refund..."
        (live, as it holds the amount). So a clip waits until the live line sent before it has
        started and finished (at most question_wait_s): lines are heard in the order sent."""
        if self._live_sent_at <= self._clip_after_live:
            return  # no live line since the last clip waited: nothing to fall behind
        speed = self._settings.section("speed")
        grace_s, most_s = float(speed["reply_grace_s"]), float(speed["question_wait_s"])
        loop = asyncio.get_running_loop()
        deadline = loop.time() + most_s
        # 1. Until the live line starts playing (or it had its ~1 s to start and didn't).
        while loop.time() < deadline and not self.speech.speaking and loop.time() - self._live_sent_at < grace_s:
            await asyncio.sleep(0.05)
        # 2. Until it has played to its end.
        while loop.time() < deadline and self.speech.speaking:
            await asyncio.sleep(0.05)
        self._clip_after_live = loop.time()

    async def say_question(self, text: str) -> None:
        """The Guard's yes-question, as its own turn: it waits until the robot is quiet
        (nothing playing, and the last line sent long enough ago to have started), at most
        question_wait_s, then speaks. 17 Nov 2026, food_07: the question queued right behind
        the Talker's reply or a waiting line and played as one long turn, so the caller
        couldn't hear it as a question of its own (and the lab's "ruko while asking" never
        found it)."""
        speed = self._settings.section("speed")
        grace_s, most_s = float(speed["reply_grace_s"]), float(speed["question_wait_s"])
        gap_s = float(speed["question_gap_s"])
        loop = asyncio.get_running_loop()
        deadline = loop.time() + most_s

        def busy() -> bool:
            now = loop.time()
            # Still playing, a line just sent that hasn't started, or too soon after the last
            # line ended: a short pause first, so the question is heard as a new question.
            return (self.speech.speaking or now - self._last_said_at < grace_s
                    or now - self._speech_ended_at < gap_s)

        while loop.time() < deadline and busy():
            await asyncio.sleep(0.05)
        # Always the live voice, never a clip: the Guard's mark needs to see the question's
        # words play. 17 Nov 2026, food_08: "Haan ya nahi?" played from its clip, no mark
        # was sent, every later "haan" counted as early, and the call ran out of requests.
        await self.say_plain(text, clip=False)

    def make_quiet_watch(self) -> QuietWatch:
        """The waiting lines and the silent caller, reading this call (decisions 11, 13)."""
        speed = self._settings.section("speed")
        grace_s = float(speed["reply_grace_s"])
        loop = asyncio.get_running_loop()
        view = CallView(
            # A line just sent may take ~1 s to start: it counts as speaking meanwhile.
            robot_speaking=lambda: self.speech.speaking or loop.time() - self._last_said_at < grace_s,
            caller_speaking=lambda: self.caller_talking,
            # The Thinker is running a step and the job is past collecting the number: its
            # quick "a person?" check on each turn while collecting is not work the caller
            # waits on (17 Nov 2026, food_02: a waiting line played while collecting).
            working=lambda: (self.thinker is not None and not self.thinker.waiting_for
                             and self.runner is not None and self.runner.mode == "wait_job"),
            saving=lambda: self.job.changes_running > 0,
            guard_asking=lambda: self.guard is not None and self.guard.asking)
        settings = QuietSettings.from_settings(self._settings.section("waiting_lines"),
                                               self._settings.section("silent_caller"))

        async def say_line(group: str, value: str) -> None:
            await self.say(self.words.phrase(group, value, self.language.value))

        return QuietWatch(view, settings, sorted(self.words.values("waiting")), say_line, self.say, self.end_call,
                          float(speed["tick_s"]))

    async def _again_if_cut(self, text: str) -> None:
        wait_s = float(self._settings.section("hand_over_end")["wait_s"])
        try:
            heard = await asyncio.wait_for(self.await_spoken(text), wait_s)
        except TimeoutError:
            return
        if not heard:
            self._diary.decision("said again", text=text, why="the caller cut it off")
            await self.say_plain(text)

    async def end_call(self, reason: str) -> None:
        """Hang up once the last line (goodbye or the hand-over line) has played, ended by
        code, not by a queued end frame (food_05, 16 Nov 2026: a queued end was dropped when
        the caller spoke over it). A cut-off line is said again, up to `tries` in all, and
        the call ends at the time limit regardless."""
        section = self._settings.section("hand_over_end")
        wait_s, tries = float(section["wait_s"]), int(section["tries"])
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_s
        text = self._last_line
        for attempt in range(tries if text else 0):
            if attempt:
                await self.say_plain(text)  # cut off: once more
            try:
                if await asyncio.wait_for(self.await_spoken(text), max(0.0, deadline - loop.time())):
                    break  # heard to the end
            except TimeoutError:
                break
        self._diary.decision("hang up", reason=reason)
        if self.line is not None:
            await self.line.end(reason)
        if self.box is not None:
            self.box.end(reason)

    def real_finder(self, sarvam_key: str, section: dict[str, Any], log: AiLog | None = None,
                    key_name: str = "") -> FindNumbers:
        """The number finder on Sarvam, one client per call (closed with it)."""
        self.finder_client = sarvam_client(sarvam_key, float(section["try_s"]))
        return sarvam_find(self.finder_client, str(section["model"]), log=log, key=key_name)

    # ---- hand-over and crashes -------------------------------------------------------------

    async def hand_over(self, reason: str, order_no: Any = None) -> None:
        """Press the hand-over tool once, then the Talker says its line and ends the call
        (decision 13). Used by the Thinker, the Guard giving up, and a crash."""
        if self._handed_over:
            return
        self._handed_over = True
        self._diary.decision("hand over", reason=reason)
        if self.runner is not None:
            self.runner.hand_over(reason)
        inputs = {"reason": reason, **({"order_no": order_no} if order_no else {})}
        if self.tools is not None:
            await self.tools.call(str(self._settings.raw["hand_over_tool"]), inputs)

    def crashed(self, task: str, error: BaseException) -> None:
        """A crash-safe task (the Thinker) crashed: a "failed" note (the Talker apologises),
        then the hand-over, once (decision 14)."""
        self._diary.error(f"{task} crashed: {error!r}")
        self.notebook.add(FAILED, f"{task} stopped working", self.job.version, {"error": repr(error)[:200]})
        if self.box is not None:
            self.box.start("hand over", self.hand_over("the back office stopped working"))

    # ---- what the call reports -------------------------------------------------------------

    def caller_began(self) -> None:
        self.caller_talking = True
        if self.guard is not None:
            self.guard.turn_began(self.caller_ms)

    async def heard_piece(self, piece: bytes) -> None:
        if self.stop_listener is not None:
            await self.stop_listener.heard(piece, self.speech.speaking)

    async def stop_talking(self, chance: float) -> None:
        """The stop-or-keep model said stop: stop the voice, make what was being said old
        news (a new version), and stop any ticket being confirmed."""
        self._diary.decision("stop talking", chance=round(chance, 3))
        await self.switch.stop_voice()
        self.job.new_turn()
        if self.guard is not None:
            self.guard.stop_ticket("the stop-or-keep model said stop")

    def caller_finished(self, words: str) -> None:
        """A finished caller turn: to the Guard (it answers a yes-question with its own
        timing) and to the Talker graph."""
        self.caller_talking = False
        # Where each turn ended, in the diary: the ears' "heard" rows are their own pieces, so
        # without this nobody could count how often one sentence was handed on as two turns
        # (18 Nov 2026, the end-of-turn change).
        self._diary.decision("caller turn", words=words)
        if self.stop_listener is not None:
            self.stop_listener.turn_finished()
        self.language.turn_finished(words)
        if self.guard is not None:
            self.guard.turn_finished(words)
        if self.runner is not None and words.strip():
            self.runner.turn(words)

    def partial_words(self, words: str) -> None:
        if self.guard is not None:
            self.guard.partial(words)

    def receipt(self, event: Played | Cleared | Key) -> None:
        """The caller's receipts and keys. A key during the yes moment is the Guard's
        ("1 for yes"); any other key goes to the Talker (a typed order number)."""
        if isinstance(event, Key):
            if self.guard is not None and self.guard.asking:
                self.guard.key(event.digit, event.caller_ms)
            elif self.runner is not None:
                self.runner.key(event.digit, event.caller_ms)
            return
        if self.guard is None:
            return
        if isinstance(event, Played):
            self.guard.played(event.name, event.caller_ms)
        else:
            self.guard.cleared(event.dropped_marks)

    def await_spoken(self, text: str) -> asyncio.Future:
        """A future that is True once the voice has said `text` to its end, False if it was cut."""
        waiter = asyncio.get_running_loop().create_future()
        self._spoken_waiters.append((text, waiter))
        return waiter

    def speech_ended(self, cut: bool, spoken: str) -> None:
        self._speech_ended_at = asyncio.get_running_loop().time()
        for text, waiter in list(self._spoken_waiters):
            tail = "".join(text.split()[-3:]).lower()
            if waiter.done():
                self._spoken_waiters.remove((text, waiter))
            elif cut or tail in "".join(spoken.split()).lower():
                waiter.set_result(not cut)
                self._spoken_waiters.remove((text, waiter))
        if self.guard is not None:
            self.guard.speech_ended(cut, spoken)


def load_clips(settings: AgentCSettings, pack_words: SpokenWords, diary: Diary) -> dict[str, bytes]:
    """The pre-made clips, read once when the robot starts (decision 11): each fixed line's
    sound, by the words the voice would say. Missing clips (a fresh copy of the project has
    none; make them with python -m cli.agent_c_lines make) go to the live voice."""
    words = SpokenWords({**pack_words.groups, **SpokenWords.from_yaml(WORDS).groups})
    folder = PROJECT_ROOT / settings.section("waiting_lines")["clips_folder"] / settings.voice["speaker"]
    names = [clip_name(*line) for line in clip_lines(words, settings.greeting)]
    return clips_by_words(Clips.load(folder, names, diary), words, settings.greeting)


def load_stop_model(settings: AgentCSettings, diary: Diary) -> StopModel | None:
    """The trained model, if its file is here. It lives outside git, so a fresh copy of
    the project has none: then the robot stops on any caller sound, as Agent A does,
    and says so in its diary."""
    path = PROJECT_ROOT / settings.section("stop_check")["model_path"]
    if not path.exists():
        diary.error(f"no stop-or-keep model at {path}: stopping on any caller sound instead")
        return None
    return OnnxStopModel(path)
