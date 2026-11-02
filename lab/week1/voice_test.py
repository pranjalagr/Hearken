"""Week-1 voice test (Part 2, decisions 10 and 12): which voice should the robot use?

Each line in voice_lines.yaml is spoken by each candidate voice and saved at
8 kHz, exactly what a caller hears on the line. The program measures what
code can measure (time to the first sound, from a streaming request); a
person judges the rest by ear on a listening page: natural, English-letter
Hinglish said right, numbers read right.

Two rounds, so the listening stays humane:
- screen: 12 of the hardest lines through every candidate; pick the best
  voice of each company;
- full: all lines through the finalists.

Facts checked against the real services on 1 Nov 2026, before writing this:
- Sarvam's streaming endpoint sends raw 16-bit little-endian PCM at the
  8 kHz we ask for.
- Google's streaming TTS ignores the 8 kHz request and sends 24 kHz, and says
  so in every chunk; so each chunk's declared rate is read, and the audio is
  resampled to 8 kHz here, as the robot would do before the wire.
- Both send little-endian samples (checked by decoding real audio both ways).

This is a batch tool, not a live call, so it is plain synchronous code: the
"never block the event loop" rule (decision 14) is about live calls.
"""

from __future__ import annotations

import base64
import html
import json
import statistics
import time
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from common.audio.pcm import float_to_pcm16, pcm16_to_float
from common.audio.resample import PHONE_RATE, resample
from common.retry import asked_wait_s, asks_too_long, worth_retrying

ROOT = Path(__file__).resolve().parents[2]
LINES = Path(__file__).with_name("voice_lines.yaml")
OUT_DIR = ROOT / "runs" / "week1" / "voice_test"

LANGS = ("hi", "en")
CATEGORIES = (
    "everyday", "english-in-hindi", "english", "order-numbers", "money", "yes-questions",
    "status-phrases", "names", "addresses", "food", "look-alike-words", "very-short",
    "pauses-and-long", "script",
)
# The screening round: one or two of the hardest lines from the riskiest groups.
SCREEN_IDS = ("A1", "B4", "C1", "D1", "D5", "E3", "F1", "F5", "J1", "J2", "K5", "N1")


@dataclass(frozen=True)
class VoiceLine:
    id: str
    category: str
    lang: str
    text: str
    listen: str


@dataclass(frozen=True)
class Setup:
    name: str  # also the folder name of its clips
    company: str  # "sarvam" or "google"
    model: str
    voice: str
    about: str


# Male voices only: the Talker speaks as a man ("bol raha hoon"). Names come
# from each service's own live voice list (1 Nov 2026), not from guesses.
SETUPS = (
    Setup("sarvam-shubh-hi", "sarvam", "bulbul:v4-flash", "shubh_hi_customer", "Sarvam: Hindi customer care, male"),
    Setup("sarvam-sunny-enhi", "sarvam", "bulbul:v4-flash", "sunny_enhi_customer", "Sarvam: Hinglish customer care, male"),
    Setup("google-csagent-1", "google", "gemini-3.8-flash-tts", "en-in-csagent-1",
          "Google: 33-year-old customer service agent from India, speaks Hinglish"),
    Setup("google-csagent-7", "google", "gemini-3.8-flash-tts", "en-in-csagent-7",
          "Google: 29-year-old customer service agent from India, speaks Hinglish"),
    Setup("google-lite-csagent-1", "google", "gemini-3.8-flash-lite-tts", "en-in-csagent-1",
          "Google (cheaper Lite model): the same voice as google-csagent-1"),
)
SETUPS_BY_NAME = {setup.name: setup for setup in SETUPS}

# Pacing: Google's free tier allows 3 speech requests a minute (its own 429,
# 1 Nov 2026), so one every 21 s stays just under; the old 6 s ran into that
# limit. Refused requests may also count against the 10-a-day cap, so it's
# better not to be refused at all. Sarvam is gentler.
PAUSE_S = {"google": 21.0, "sarvam": 0.5}
RETRY_WAITS_S = (15.0, 30.0, 60.0)
# The longest wait for the next bytes of a reply. A normal first sound comes
# in under 7 s, so past this the request is stuck, not slow. Without it, one
# stuck Google request froze the 1 Nov screening run for 19 minutes: the
# Google library waits forever by default.
REQUEST_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class Spoken:
    pcm: bytes  # 16-bit little-endian mono at 8 kHz: what the caller hears
    first_audio_s: float  # request sent -> first audio arrived
    total_s: float  # request sent -> last audio arrived


def load_lines(path: Path = LINES) -> list[VoiceLine]:
    """Strict: a typo must fail here, not as a silent missing clip."""
    lines = []
    for item in yaml.safe_load(path.read_text(encoding="utf-8")):
        if set(item) != {"id", "category", "lang", "text", "listen"}:
            raise ValueError(f"line {item.get('id')}: needs exactly id, category, lang, text, listen")
        if item["category"] not in CATEGORIES or item["lang"] not in LANGS:
            raise ValueError(f"line {item['id']}: unknown category or lang")
        if not str(item["text"]).strip() or not str(item["listen"]).strip():
            raise ValueError(f"line {item['id']}: empty text or listen note")
        lines.append(VoiceLine(**{key: str(value) for key, value in item.items()}))
    ids = [line.id for line in lines]
    if len(ids) != len(set(ids)):
        raise ValueError("line ids must be unique")
    return lines


# ---- Sarvam ---------------------------------------------------------------------

SARVAM_STREAM_URL = "https://api.sarvam.ai/text-to-speech/stream"
SARVAM_LANG = {"hi": "hi-IN", "en": "en-IN"}


def sarvam_request(setup: Setup, line: VoiceLine, key: str) -> urllib.request.Request:
    body = {
        "text": line.text,
        "language_code": SARVAM_LANG[line.lang],
        "model": setup.model,
        "speaker": setup.voice,
        "speech_sample_rate": PHONE_RATE,
        "output_audio_codec": "linear16",
        # Sarvam's own guide: keep preprocessing on and don't hand-expand
        # numbers; that is also decision 10's "nothing special" start.
        "enable_preprocessing": True,
    }
    return urllib.request.Request(
        SARVAM_STREAM_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"api-subscription-key": key, "Content-Type": "application/json"},
        method="POST",
    )


def sarvam_speak(
    setup: Setup,
    line: VoiceLine,
    key: str,
    opener: Callable[..., Any] = urllib.request.urlopen,
    clock: Callable[[], float] = time.perf_counter,
) -> Spoken:
    request = sarvam_request(setup, line, key)
    start = clock()
    with opener(request, timeout=REQUEST_TIMEOUT_S) as response:
        kind = response.headers.get("Content-Type", "")
        if not kind.startswith("audio/pcm"):
            raise RuntimeError(f"expected raw PCM audio, got {kind!r}")
        first = response.read(320)  # one 20 ms piece: the first sound has arrived
        first_s = clock() - start
        rest = response.read()
        total_s = clock() - start
    pcm = first + rest
    if not pcm or len(pcm) % 2:
        raise RuntimeError(f"broken audio: {len(pcm)} bytes")
    return Spoken(pcm, first_s, total_s)


# ---- Google ---------------------------------------------------------------------

# Google's documented default for streamed speech, used only if a chunk
# doesn't state its own rate.
GOOGLE_STREAM_RATE = 24000


def google_speak(setup: Setup, line: VoiceLine, client: Any, clock: Callable[[], float] = time.perf_counter) -> Spoken:
    start = clock()
    first_s: float | None = None
    rate: int | None = None
    chunks: list[bytes] = []
    stream = client.interactions.create(
        model=setup.model,
        input=[{"type": "user_input", "content": [{"type": "text", "text": line.text}]}],
        response_format={"type": "audio", "mime_type": "audio/l16", "sample_rate": PHONE_RATE},
        generation_config={"speech_config": [{"voice": setup.voice}]},
        stream=True,
    )
    for event in stream:
        if event.event_type != "step.delta" or event.delta.type != "audio":
            continue
        if first_s is None:
            first_s = clock() - start
        chunk_rate = getattr(event.delta, "sample_rate", None) or GOOGLE_STREAM_RATE
        if getattr(event.delta, "channels", 1) not in (None, 1):
            raise RuntimeError("expected mono audio")
        if rate is None:
            rate = chunk_rate
        elif chunk_rate != rate:
            raise RuntimeError(f"the sample rate changed mid-stream: {rate} -> {chunk_rate}")
        chunks.append(base64.b64decode(event.delta.data))
    total_s = clock() - start
    if not chunks:
        raise RuntimeError("no audio came back")
    raw = b"".join(chunks)
    if len(raw) % 2:
        raise RuntimeError(f"broken audio: {len(raw)} bytes")
    pcm = float_to_pcm16(resample(pcm16_to_float(raw), rate, PHONE_RATE))
    return Spoken(pcm, first_s, total_s)


# ---- running the test -----------------------------------------------------------


def with_retries(
    call: Callable[[], Spoken],
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[BaseException, float], None] = lambda error, wait: None,
) -> Spoken:
    """on_retry hears every retry before its wait, so none goes unseen.

    When the service says how long to wait, that is honoured: a wait longer
    than ours is used instead ("retry in 54s" after our 15 s would only fail
    again), and one too long to sit through (a daily limit) fails at once."""
    for wait in (*RETRY_WAITS_S, None):
        try:
            return call()
        except Exception as error:
            if wait is None or not worth_retrying(error) or asks_too_long(error):
                raise
            wait = max(wait, asked_wait_s(error) or 0.0)
            on_retry(error, wait)
            sleep(wait)
    raise AssertionError("unreachable")


def describe(error: BaseException) -> str:
    # A timeout's own message can be empty, so the type name always leads.
    return f"{type(error).__name__}: {str(error)[:300]}"


def clip_path(out_dir: Path, setup: Setup, line: VoiceLine) -> Path:
    return out_dir / setup.name / f"{line.id}.wav"


def run_round(
    lines: Iterable[VoiceLine],
    setups: Iterable[Setup],
    speak: Callable[[Setup, VoiceLine], Spoken],
    out_dir: Path = OUT_DIR,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Make every missing clip and record its timing. Clips already made are
    skipped, so a run stopped by a daily limit just continues next time."""
    from common.audio.sound_files import write_sound

    timings = load_timings(out_dir)
    # Models out of quota for this run (they asked for hours): their other
    # clips are left unmade, not failed, so the next run makes them.
    out_of_quota: dict[str, int] = {}
    # Counted across voices: two voices can share one model and so one
    # per-minute limit, so the pause must also apply when the voice changes.
    made = 0
    for setup in setups:
        for line in lines:
            path = clip_path(out_dir, setup, line)
            key = f"{setup.name}/{line.id}"
            if path.exists() and key in timings:
                continue
            if setup.model in out_of_quota:
                out_of_quota[setup.model] += 1
                continue
            if made:
                sleep(PAUSE_S[setup.company])
            made += 1
            retries = []

            def note_retry(error: BaseException, wait: float, key: str = key) -> None:
                retries.append(1)
                log(f"  {key}: retrying in {wait:.0f} s after {describe(error)}")

            try:
                spoken = with_retries(lambda: speak(setup, line), sleep=sleep, on_retry=note_retry)
            except Exception as error:
                timings[key] = {"error": describe(error), "retries": len(retries)}
                log(f"  {key}: FAILED {timings[key]['error']}")
                save_timings(out_dir, timings)
                if asks_too_long(error):
                    out_of_quota[setup.model] = 0
                    log(f"  {setup.model} asks to wait {asked_wait_s(error) / 3600:.1f} h: "
                        "its other clips are left for the next run")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            write_sound(path, pcm16_to_float(spoken.pcm), PHONE_RATE)
            timings[key] = {
                "first_audio_s": round(spoken.first_audio_s, 3),
                "total_s": round(spoken.total_s, 3),
                "seconds": round(len(spoken.pcm) / 2 / PHONE_RATE, 2),
                # The time above is the last try only; retries say how often
                # the service failed before it, which matters for a live call.
                "retries": len(retries),
            }
            save_timings(out_dir, timings)  # after every clip: a crash keeps the progress
            log(f"  {key}: first sound {spoken.first_audio_s:.2f} s")
    for model, left in out_of_quota.items():
        if left:
            log(f"  {model}: {left} clips left unmade (out of quota); run the same command again later")
    return timings


def load_timings(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "timings.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_timings(out_dir: Path, timings: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=1, sort_keys=True), encoding="utf-8")


def timing_summary(setup: Setup, timings: dict[str, Any], lines: Iterable[VoiceLine]) -> dict[str, Any]:
    """Only the given lines count: timings.json holds every run, and a
    5-line rerun once reported the voice's 16 clips from all runs."""
    rows = [timings[key] for line in lines if (key := f"{setup.name}/{line.id}") in timings]
    firsts = [row["first_audio_s"] for row in rows if "first_audio_s" in row]
    return {
        "clips": len(firsts),
        "errors": sum(1 for row in rows if "error" in row),
        # Rows from before retries were counted have none recorded.
        "retries": sum(row.get("retries", 0) for row in rows),
        "median_first_s": round(statistics.median(firsts), 2) if firsts else None,
        "slowest_first_s": round(max(firsts), 2) if firsts else None,
    }


# ---- the listening page ---------------------------------------------------------


def listening_page(lines: list[VoiceLine], setups: list[Setup], timings: dict[str, Any], when: datetime) -> str:
    """A local page to judge every clip by ear. Marks are kept in the
    browser while judging; "Copy results" gives a short text to paste back."""
    e = html.escape
    setup_rows = []
    for setup in setups:
        summary = timing_summary(setup, timings, lines)
        first = f"{summary['median_first_s']} s median, {summary['slowest_first_s']} s slowest" if summary["clips"] else "no clips yet"
        setup_rows.append(
            f"<tr><td><b>{e(setup.name)}</b></td><td>{e(setup.about)}<br><code>{e(setup.model)} / {e(setup.voice)}</code></td>"
            f"<td>{e(first)}</td><td>{summary['clips']} made, {summary['errors']} failed, "
            f"retries: {summary['retries']}</td></tr>"
        )
    cards = []
    for line in lines:
        players = []
        for setup in setups:
            key = f"{setup.name}/{line.id}"
            row = timings.get(key)
            name = f"{setup.name}__{line.id}"
            if row and "first_audio_s" in row:
                clip = f"{e(setup.name)}/{e(line.id)}.wav"
                players.append(
                    f'<div class="player"><span class="who">{e(setup.name)}</span>'
                    f'<audio controls preload="none" src="{clip}"></audio>'
                    f'<label><input type="radio" name="{e(name)}" value="good"> ✅</label>'
                    f'<label><input type="radio" name="{e(name)}" value="wrong"> ❌</label></div>'
                )
            else:
                problem = row["error"] if row else "not made yet"
                players.append(f'<div class="player"><span class="who">{e(setup.name)}</span><i>{e(problem)}</i></div>')
        cards.append(
            f'<section class="card"><h3>{e(line.id)} <small>{e(line.category)} · {e(line.lang)}</small></h3>'
            f'<p class="text">{e(line.text)}</p><p class="listen">Listen for: {e(line.listen)}</p>{"".join(players)}</section>'
        )
    names = json.dumps([setup.name for setup in setups])
    return _PAGE.format(
        when=e(f"{when:%d %b %Y, %H:%M}"),
        setups="".join(setup_rows),
        cards="".join(cards),
        names=names,
    )


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice test</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 980px; margin: 20px auto; padding: 0 16px; }}
table {{ border-collapse: collapse; width: 100%; }} td, th {{ border: 1px solid #8884; padding: 6px; vertical-align: top; }}
.card {{ border: 1px solid #8885; border-radius: 10px; padding: 10px 14px; margin: 14px 0; }}
.card h3 {{ margin: 0 0 4px; }} .card small {{ font-weight: normal; opacity: .7; }}
.text {{ font-size: 1.1em; margin: 4px 0; }} .listen {{ margin: 4px 0 8px; opacity: .85; }}
.player {{ display: flex; align-items: center; gap: 10px; margin: 4px 0; flex-wrap: wrap; }}
.who {{ min-width: 170px; font-family: monospace; }} audio {{ height: 32px; }}
#summary {{ position: sticky; top: 0; background: Canvas; border: 2px solid #8888; border-radius: 10px; padding: 8px 12px; z-index: 1; }}
textarea {{ width: 100%; height: 90px; font-family: monospace; }}
</style></head><body>
<h1>Voice test (Part 2, decisions 10 and 12)</h1>
<p>Made {when}. Every clip is 8 kHz: exactly what a caller hears on the line.</p>
<table><tr><th>Voice</th><th>What it is</th><th>Time to first sound (measured)</th><th>Clips</th></tr>{setups}</table>
<p><b>How to judge:</b> play each clip and read its "Listen for". Mark ✅ if it is natural <i>and</i> says that
thing right, ❌ if not. Use headphones if you can. Your marks are kept in this browser while you judge.</p>
<div id="summary"><b>Your marks</b>
<button id="copy" type="button">Copy results</button>
<textarea id="out" readonly></textarea></div>
{cards}
<script>
const names = {names};
const store = "voice-test-marks";
function load() {{ try {{ return JSON.parse(localStorage.getItem(store) || "{{}}"); }} catch {{ return {{}}; }} }}
function save(marks) {{ try {{ localStorage.setItem(store, JSON.stringify(marks)); }} catch {{}} }}
const marks = load();
document.querySelectorAll('input[type=radio]').forEach((input) => {{
  if (marks[input.name] === input.value) input.checked = true;
  input.addEventListener('change', () => {{ marks[input.name] = input.value; save(marks); show(); }});
}});
function show() {{
  const lines = [];
  for (const name of names) {{
    const mine = Object.entries(marks).filter(([key]) => key.startsWith(name + "__"));
    const good = mine.filter(([, value]) => value === "good").length;
    const wrong = mine.filter(([, value]) => value === "wrong").map(([key]) => key.split("__")[1]);
    lines.push(name + ": " + good + " good, " + wrong.length + " wrong (of " + mine.length + " judged)" + (wrong.length ? " | wrong: " + wrong.join(" ") : ""));
  }}
  document.getElementById("out").value = lines.join("\\n");
}}
document.getElementById("copy").addEventListener("click", async () => {{
  const text = document.getElementById("out").value;
  try {{ await navigator.clipboard.writeText(text); }} catch {{ document.getElementById("out").select(); }}
}});
show();
</script></body></html>
"""


def write_page(
    lines: list[VoiceLine],
    setups: list[Setup],
    timings: dict[str, Any],
    out_dir: Path = OUT_DIR,
    name: str = "listen.html",
) -> Path:
    # Each kind of run gets its own page, so a small rerun of a few lines
    # can't replace the page of a whole round.
    path = out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(listening_page(lines, setups, timings, datetime.now()), encoding="utf-8")
    return path
