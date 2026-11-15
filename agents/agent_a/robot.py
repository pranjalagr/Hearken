"""Agent A: the plain chain (Part 2). Ears -> brain -> voice, with Pipecat's defaults.

It is the "before" picture, what most teams build first, so it is deliberately
left as Pipecat ships it:
- it stops talking on any caller sound (Pipecat's default interruption);
- Smart Turn v3.2, Pipecat's built-in end-of-turn model, decides when the
  caller has finished (our end-of-turn test found it weak on Hinglish: that
  is the honest baseline);
- the brain waits for each tool, unless the background-tool switch is on.
It shares the lab's ears, voice, tools, greeting and call box with the other
robots, so the comparison measures the design, not the parts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.sarvam.stt import SarvamRealtimeSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transcriptions.language import Language

from agents.common.call_box import CallBox, CallBoxSettings, CallResult
from agents.common.diary import NO_DIARY, Diary
from agents.common.pipecat_brain_cap import BRAIN_CAP, BrainCap
from agents.common.pipecat_diary import DiaryObserver
from agents.common.pipecat_hangup import END_CALL_SCHEMA, HangUp
from agents.common.pipecat_line import LineTransport
from agents.common.pipecat_tools import connect_tools, tools_schema
from agents.common.tools import Tools
from common.wire.protocol import PIECE_MS, SAMPLE_RATE, End
from common.wire.robot_line import CallerAudio, LineClosed, RobotLine

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AgentASettings:
    greeting: str
    brain_model: str
    ears: dict
    voice: dict
    background_tools: bool
    prompt: str

    @property
    def models(self) -> dict[str, str]:
        """Kept with every run (run.json), so an old run keeps its meaning."""
        return {"brain": self.brain_model, "ears": self.ears["model"], "voice": f"{self.voice['model']}/{self.voice['speaker']}"}


def load_settings(path: Path = HERE / "settings.yaml", prompt: Path = HERE / "prompt.md",
                  background_tools: bool | None = None) -> AgentASettings:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    background = data["background_tools"] if background_tools is None else background_tools
    return AgentASettings(data["greeting"], data["brain"]["model"], data["ears"], data["voice"], bool(background),
                          prompt.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Keys:
    gemini: str
    sarvam: str


class AgentA:
    def __init__(self, settings: AgentASettings, keys: Keys, tools: Tools, *,
                 box_settings: CallBoxSettings = CallBoxSettings(), diary: Diary = NO_DIARY,
                 max_brain_requests: int = 40) -> None:
        self._settings = settings
        self._max_brain_requests = max_brain_requests  # the lab passes its own (lab/settings.yaml)
        self._keys = keys
        self._tools = tools
        self._box_settings = box_settings
        self._diary = diary
        self._heard_pieces = 0
        self._hang_up = HangUp(diary)
        self.result: CallResult | None = None

    async def __call__(self, line: RobotLine) -> None:
        box = CallBox(self._box_settings)
        try:
            self.result = await box.run(lambda b: self._main(line, b))
        finally:
            # The ears are paid by the minute of caller audio sent to them.
            self._diary.cost(self._settings.ears["model"], round(self._heard_pieces * PIECE_MS / 60000, 3), "minutes")

    async def _main(self, line: RobotLine, box: CallBox) -> None:
        transport = LineTransport(line)
        task = PipelineTask(self._pipeline(transport, await self._tools.catalog()),
                            params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE,
                                                  enable_metrics=True, enable_usage_metrics=True),
                            observers=[DiaryObserver(self._diary),
                                       BrainCap(self._max_brain_requests, lambda n: self._cap_reached(line, box, n))])
        box.start("ears", self._listen(line, transport, box))
        # Every robot opens the same way, with a fixed line.
        await task.queue_frames([TTSSpeakFrame(self._settings.greeting)])
        await PipelineRunner(handle_sigint=False).run(task)
        if self._hang_up.reason is not None:
            # The brain ended the call and the goodbye has played: say so on the wire (decision 13).
            await line.end(self._hang_up.reason)  # the lab adds "robot:" itself

    async def _cap_reached(self, line: RobotLine, box: CallBox, requests: int) -> None:
        # Hang up on the wire first, so the lab records why the call ended, then
        # end the box, which stops the pipeline mid-turn: nothing more is sent.
        self._diary.decision("hang up", reason=BRAIN_CAP, requests=requests)
        await line.end(BRAIN_CAP)
        box.end(BRAIN_CAP)

    def _pipeline(self, transport: LineTransport, tool_specs) -> Pipeline:
        s = self._settings
        language = Language(s.ears["language"])
        ears = SarvamRealtimeSTTService(
            api_key=self._keys.sarvam, sample_rate=SAMPLE_RATE,
            settings=SarvamRealtimeSTTService.Settings(model=s.ears["model"], language=language, mode=s.ears["mode"],
                                                       stream_type=s.ears["stream_type"],
                                                       silence_duration_ms=s.ears["silence_ms"]))
        brain = GoogleLLMService(api_key=self._keys.gemini,
                                 settings=GoogleLLMService.Settings(model=s.brain_model, system_instruction=s.prompt))
        connect_tools(brain, self._tools, self._diary, background=s.background_tools)
        self._hang_up.connect(brain)
        voice = SarvamTTSService(api_key=self._keys.sarvam, sample_rate=SAMPLE_RATE,
                                 settings=SarvamTTSService.Settings(model=s.voice["model"], voice=s.voice["speaker"],
                                                                    language=Language(s.voice["language"])))
        context = LLMContext(tools=tools_schema(tool_specs, (END_CALL_SCHEMA,)))
        turns = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))
        return Pipeline([transport.input(), ears, turns.user(), brain, voice, transport.output(), turns.assistant()])

    async def _listen(self, line: RobotLine, transport: LineTransport, box: CallBox) -> None:
        async for event in line.events():
            if isinstance(event, CallerAudio):
                box.heard_audio()
                self._heard_pieces += 1
                await transport.input().feed(event.piece)
            elif isinstance(event, End):
                box.end(f"caller:{event.reason}")
                return
            elif isinstance(event, LineClosed):
                box.end("line closed")
                return
            # Receipts, "cleared" and keys: Agent A, like most chains, uses none of them.
