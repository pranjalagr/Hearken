"""Agent C: the team robot (docs/agent_c_design.md). Ears -> Talker -> voice, with a
Thinker and a Guard beside the pipeline.

It shares the lab's ears, voice, tools, greeting, call box, diary, hang-up and brain
cap with Agent A, so a comparison measures the design, not the parts. What differs:
- the Talker holds no backend tool: its one tool is ask_thinker (decision 7);
- the caller's receipts ("played", "cleared") and keys are read, not dropped:
  the Guard times a yes from them (decision 22).
"""

from __future__ import annotations

from dataclasses import dataclass

from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
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

from agents.agent_c.settings import AgentCSettings
from agents.agent_c.talker.ask_thinker import ASK_THINKER_SCHEMA, AskThinker, Desk, NoThinkerYet
from agents.common.call_box import CallBox, CallBoxSettings, CallResult
from agents.common.diary import NO_DIARY, Diary
from agents.common.pipecat_brain_cap import BRAIN_CAP, BrainCap
from agents.common.pipecat_diary import DiaryObserver
from agents.common.pipecat_line import LineTransport
from agents.common.tools import Tools
from common.wire.protocol import PIECE_MS, SAMPLE_RATE, Cleared, End, Key, Played
from common.wire.robot_line import CallerAudio, LineClosed, RobotLine


@dataclass(frozen=True)
class Keys:
    gemini: str
    sarvam: str


def talker_tools() -> ToolsSchema:
    """The Talker's whole tool list. No backend tool, no confirm, no Guard door: it can
    only ask (decision 7). A test holds this to exactly one tool."""
    return ToolsSchema(standard_tools=[ASK_THINKER_SCHEMA])


class AgentC:
    def __init__(self, settings: AgentCSettings, keys: Keys, tools: Tools, *,
                 box_settings: CallBoxSettings = CallBoxSettings(), diary: Diary = NO_DIARY,
                 max_brain_requests: int = 40, desk: Desk | None = None) -> None:
        self._settings = settings
        self._keys = keys
        self._tools = tools
        self._box_settings = box_settings
        self._diary = diary
        self._max_brain_requests = max_brain_requests  # the lab passes its own (lab/settings.yaml)
        self._desk = desk or NoThinkerYet()
        self._heard_pieces = 0
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
        task = PipelineTask(self.pipeline(transport),
                            params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE,
                                                  enable_metrics=True, enable_usage_metrics=True),
                            observers=[DiaryObserver(self._diary),
                                       BrainCap(self._max_brain_requests, lambda n: self._cap_reached(line, box, n))])
        box.start("ears", self._listen(line, transport, box))
        # Every robot opens the same way, with a fixed line (decision 13).
        await task.queue_frames([TTSSpeakFrame(self._settings.greeting)])
        await PipelineRunner(handle_sigint=False).run(task)

    async def _cap_reached(self, line: RobotLine, box: CallBox, requests: int) -> None:
        # As Agent A: the wire first, so the lab records why the call ended, then the box.
        self._diary.decision("hang up", reason=BRAIN_CAP, requests=requests)
        await line.end(BRAIN_CAP)
        box.end(BRAIN_CAP)

    def pipeline(self, transport: LineTransport) -> Pipeline:
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
        AskThinker(self._desk, self._diary).connect(talker)
        voice = SarvamTTSService(api_key=self._keys.sarvam, sample_rate=SAMPLE_RATE,
                                 settings=SarvamTTSService.Settings(model=s.voice["model"], voice=s.voice["speaker"],
                                                                    language=Language(s.voice["language"])))
        context = LLMContext(tools=talker_tools())
        turns = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()))
        return Pipeline([transport.input(), ears, turns.user(), talker, voice, transport.output(), turns.assistant()])

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
