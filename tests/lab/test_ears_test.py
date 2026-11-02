"""The ears test, offline: each service against a fake that speaks its documented
protocol over a real WebSocket, so no key or request is spent. What the real
services do is only proved by the first real clip."""

import asyncio
import base64
import http
import io
import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile
import websockets
from websockets.asyncio.server import serve

from common.audio.phone_line import band_limit, mu_law_round_trip
from common.audio.resample import PHONE_RATE, resample
from common.audio.sound_files import read_sound, write_sound
from common.retry import asked_wait_s, worth_retrying
from lab.ears_test.clips import (
    BABBLE_VOICES,
    WORK_RATE,
    Detail,
    clip_path,
    load_lines,
    make_babble,
    phone_version,
    prepare,
    raw_path,
    speech_span,
)
from lab.ears_test.record_server import BadRecording, make_server, save_recording
from lab.ears_test.runner import _row, listen_once, load_results, run_service
from lab.ears_test.score import flat, heard_detail, latin_share, report, score_service, with_digits
from lab.ears_test.services import Deepgram, EarsError, GoogleLive, Heard, Sarvam, Soniox

ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")
LINES = {line.id: line for line in load_lines()}
LIST = list(LINES.values())


def tone(seconds=1.0, rate=WORK_RATE, hz=440.0, level=0.3):
    return (level * np.sin(2 * np.pi * hz * np.arange(int(seconds * rate)) / rate)).astype(np.float32)


# ---- the lines -----------------------------------------------------------------------


def test_every_line_says_something_and_each_kind_of_key_detail_is_covered():
    kinds = {detail.kind for line in LIST for detail in line.expect}
    assert kinds == {"number", "name", "amount", "haan", "nahi", "ruko", "word"} and len(LIST) >= 30


def test_every_line_passes_on_its_own_words():
    # An answer key that can't accept the line itself would mark every service wrong.
    for line in LIST:
        for detail in line.expect:
            assert heard_detail(detail, line.say), (line.id, detail)


def test_the_order_numbers_said_are_real_seed_orders():
    seed = json.loads((ROOT / "packs" / "food" / "seed.json").read_text(encoding="utf-8"))
    orders = {str(order["order_no"]) for order in seed["orders"]}
    for line in LIST:
        for detail in line.expect:
            if detail.kind == "number" and detail.forms[0].startswith("45"):
                assert detail.forms[0] in orders, line.id


# ---- marking -------------------------------------------------------------------------


@pytest.mark.parametrize("heard", [
    "Mera order number 4521 hai", "order 4,521", "4 5 2 1", "chaar paanch do ek", "char panch do ek",
    "चार पांच दो एक", "४५२१", "45 21",
])
def test_an_order_number_counts_however_it_is_written(heard):
    assert heard_detail(Detail("number", ("4521",)), heard)


@pytest.mark.parametrize("heard", ["4512", "4 5 2", "chaar paanch ek do", "45210", "", "order"])
def test_a_wrong_digit_or_a_part_of_the_number_never_counts(heard):
    assert not heard_detail(Detail("number", ("4521",)), heard)


def test_a_comma_ends_a_number_so_the_next_one_cant_merge_into_it():
    assert heard_detail(Detail("number", ("4530",)), "nahi nahi, 4530, 45 tees nahi")
    assert not heard_detail(Detail("number", ("4530",)), "nahi nahi, 453045 tees nahi")


def test_the_word_do_is_still_a_word():
    # "do" is also the digit 2: reading it only as a digit would lose "kar do".
    assert with_digits("kar do") == "kar 2"
    assert heard_detail(Detail("word", ("kar do",)), "theek hai kar do")


def test_a_word_only_counts_whole():
    ruko = LINES["Y5"].expect[0]
    assert heard_detail(ruko, "Ruko!") and heard_detail(ruko, "रुको")
    assert not heard_detail(ruko, "Rukhsana") and not heard_detail(ruko, "chalo")


def test_punctuation_and_case_dont_matter_but_an_apostrophe_stays():
    assert flat("Wait, DON’T cancel!") == "wait don't cancel"


def test_english_letters_are_measured_on_letters_only():
    assert latin_share("haan 4521") == 1.0 and latin_share("हां") == 0.0
    assert latin_share("4521 ...") is None and latin_share("") is None


CLIPS = {"s1/Y5": {"speech_start_s": 0.3, "speech_end_s": 1.0, "seconds": 2.8}}


def row(final, partial="", partials=1, first=0.5, after=0.4, retries=0):
    return {"final": final, "last_partial": partial, "partials": partials, "first_text_s": first,
            "final_after_speech_s": after, "retries": retries}


def test_a_service_is_scored_on_key_details_with_every_miss_kept():
    results = {"a/s1/Y5/phone": row("Ruko", "रुको"), "a/s1/Y5/noise": row("chalo", first=1.5, after=2.0, retries=2)}
    score = score_service("a", LIST, results, CLIPS)
    assert str(score.details) == "1/2 (50%)"
    assert str(score.by_version["phone"]) == "1/1 (100%)" and str(score.by_version["noise"]) == "0/1 (0%)"
    assert score.misses == [{"clip": "s1/Y5/noise", "kind": "ruko", "said": "Ruko!", "heard": "chalo"}]
    assert score.retries == 2 and score.final_latin == [1.0, 1.0] and score.partial_latin == [0.0]
    assert sorted(score.first_text_s) == [0.5, 1.5] and score.audio_minutes == pytest.approx(5.6 / 60)


def test_a_failed_clip_and_a_clip_never_sent_are_not_counted_as_misses():
    score = score_service("a", LIST, {"a/s1/Y5/phone": {"error": "RuntimeError: x", "retries": 3}}, CLIPS)
    assert score.details.total == 0 and score.failed == 1 and score.retries == 3 and not score.misses


def test_the_report_ranks_by_key_details_and_never_guesses_a_price():
    good = score_service("good", LIST, {"good/s1/Y5/phone": row("ruko"), "good/s1/Y5/noise": row("ruko")}, CLIPS)
    bad = score_service("bad", LIST, {"bad/s1/Y5/phone": row("no"), "bad/s1/Y5/noise": row("no|pipe")}, CLIPS)
    text = report([bad, good], 2, {"good": None, "bad": 0.01}, "babble", "today")
    assert text.index("| good |") < text.index("| bad |")
    assert "price not checked" in text and "$0.00" in text and "no/pipe" in text


# ---- recordings and clips ------------------------------------------------------------


def make_recordings(sounds, ids=("Y1", "Y2", "Y3", "Y5", "Y7", "N1"), rate=48000):
    for number, line_id in enumerate(ids):
        path = raw_path("s1", line_id, sounds)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_sound(path, tone(1.0, rate, 300 + 60 * number), rate)


def test_prepare_makes_both_phone_versions_at_8khz_and_they_are_repeatable(tmp_path):
    make_recordings(tmp_path)
    first = prepare(tmp_path, tmp_path / "none")
    clean, rate = read_sound(clip_path("phone", "s1", "Y1", tmp_path))
    noisy, _ = read_sound(clip_path("noise", "s1", "Y1", tmp_path))
    assert rate == PHONE_RATE and len(clean) == len(noisy) and not np.array_equal(clean, noisy)
    assert "babble" in first["noise"] and first["clips"]["s1/Y1"]["seconds"] == pytest.approx(2.8)
    prepare(tmp_path, tmp_path / "none")
    assert np.array_equal(noisy, read_sound(clip_path("noise", "s1", "Y1", tmp_path))[0])  # every service, the same clip


def test_a_phone_version_goes_through_the_band_limit_and_the_codec():
    voice = tone(0.5)
    expected = mu_law_round_trip(band_limit(resample(voice, WORK_RATE, PHONE_RATE)))
    assert np.array_equal(phone_version(voice), expected)
    assert not np.array_equal(phone_version(voice), band_limit(resample(voice, WORK_RATE, PHONE_RATE)))


def test_noise_also_plays_in_the_callers_silence(tmp_path):
    make_recordings(tmp_path)
    prepare(tmp_path, tmp_path / "none")
    clean, _ = read_sound(clip_path("phone", "s1", "Y1", tmp_path))
    noisy, _ = read_sound(clip_path("noise", "s1", "Y1", tmp_path))
    tail = int(1.0 * PHONE_RATE)
    assert np.abs(clean[-tail:]).max() < 1e-3 < np.abs(noisy[-tail:]).max()


def test_real_noise_files_win_over_babble(tmp_path):
    make_recordings(tmp_path, ids=("Y1",))
    noise_dir = tmp_path / "noise_files"
    noise_dir.mkdir()
    write_sound(noise_dir / "traffic.wav", tone(2.0, 44100, 900), 44100)
    assert prepare(tmp_path, noise_dir)["noise"] == "1 noise files"


def test_babble_needs_enough_voices_and_says_what_to_do():
    with pytest.raises(ValueError, match="Record more"):
        make_babble([tone()] * (BABBLE_VOICES - 1))


def test_prepare_refuses_when_nothing_is_recorded(tmp_path):
    with pytest.raises(ValueError, match="no recordings"):
        prepare(tmp_path, tmp_path)


def test_the_speech_span_ignores_the_silence_around_it():
    samples = np.concatenate([np.zeros(800), tone(0.5, PHONE_RATE), np.zeros(1600)])
    assert speech_span(samples, PHONE_RATE) == pytest.approx((0.1, 0.6), abs=0.021)
    assert speech_span(np.zeros(800), PHONE_RATE) == (0.0, 0.0) and speech_span(np.zeros(0), PHONE_RATE) == (0.0, 0.0)


def wav_bytes(samples, rate=48000):
    out = io.BytesIO()
    soundfile.write(out, samples, rate, format="WAV", subtype="PCM_16")
    return out.getvalue()


def test_a_good_recording_is_saved_and_a_redo_replaces_it(tmp_path):
    path = save_recording("s1", "Y1", wav_bytes(tone(1.0, 48000)), LIST, tmp_path)
    assert path == raw_path("s1", "Y1", tmp_path) and read_sound(path)[1] == 48000
    save_recording("s1", "Y1", wav_bytes(tone(2.0, 48000)), LIST, tmp_path)
    assert len(read_sound(path)[0]) == 96000 and not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("speaker, line, data, why", [
    ("s1", "Y1", wav_bytes(np.zeros(48000, np.float32)), "too quiet"),
    ("s1", "Y1", wav_bytes(tone(1.0, 48000, level=0.015)), "too quiet"),
    ("s1", "Y1", wav_bytes(tone(0.1, 48000)), "lasted"),
    ("s1", "Y1", wav_bytes(tone(31.0, 48000)), "lasted"),
    ("s1", "Y1", wav_bytes(np.repeat(tone(0.5, 48000)[:128 * 187].reshape(-1, 128), 2, axis=0).ravel()), "doubled"),
    ("s1", "Y1", b"not sound", "readable"),
    ("s1", "ZZ", wav_bytes(tone(1.0, 48000)), "unknown line"),
    ("../evil", "Y1", wav_bytes(tone(1.0, 48000)), "short code"),
    ("Rahul Verma", "Y1", wav_bytes(tone(1.0, 48000)), "short code"),
    ("", "Y1", wav_bytes(tone(1.0, 48000)), "short code"),
])
def test_a_bad_recording_is_refused_and_nothing_is_written(tmp_path, speaker, line, data, why):
    with pytest.raises(BadRecording, match=why):
        save_recording(speaker, line, data, LIST, tmp_path)
    assert not list(tmp_path.rglob("*.wav"))


@pytest.fixture
def page(tmp_path):
    server = make_server(0, tmp_path, LIST)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", tmp_path
    server.shutdown()
    server.server_close()


def post(url, data, headers=None):
    request = urllib.request.Request(url, data=data, method="POST", headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_page_saves_through_its_server_and_reports_progress(page):
    base, sounds = page
    assert post(f"{base}/save?speaker=s1&line=Y1", wav_bytes(tone(1.0, 48000))) == (200, {"saved": True})
    assert raw_path("s1", "Y1", sounds).exists()
    with urllib.request.urlopen(f"{base}/lines?speaker=s1", timeout=5) as response:
        done = {line["id"]: line["done"] for line in json.loads(response.read())}
    assert done["Y1"] is True and done["Y2"] is False


def test_the_server_explains_a_refusal_and_refuses_other_websites(page):
    base, sounds = page
    status, body = post(f"{base}/save?speaker=s1&line=Y1", b"junk")
    assert status == 400 and "readable" in body["error"]
    refused = post(f"{base}/save?speaker=s1&line=Y1", wav_bytes(tone(1.0, 48000)), {"Origin": "https://evil.example"})
    assert refused[0] == 403 and not raw_path("s1", "Y1", sounds).exists()


def test_the_page_and_its_scripts_are_served_with_the_right_type(page):
    base, _ = page
    for name, kind in (("", "text/html"), ("record.js", "text/javascript"), ("wav.js", "text/javascript"),
                       ("capture-worklet.js", "text/javascript")):
        with urllib.request.urlopen(f"{base}/{name}", timeout=5) as response:
            assert response.headers["Content-Type"].startswith(kind), name


needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


@needs_node
def test_the_javascript_tests_pass():
    result = subprocess.run([NODE, "--test", "lab/ears_test/record_page/wav.test.js", "lab/ears_test/record_page/take.test.js"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@needs_node
def test_python_accepts_a_wav_made_by_the_page(tmp_path):
    script = ("import('./lab/ears_test/record_page/wav.js').then(m => {"
              "const s = new Float32Array(48000).map((_, i) => 0.3 * Math.sin(i / 20));"
              "process.stdout.write(Buffer.from(m.encodeWav(s, 48000)).toString('base64'))})")
    result = subprocess.run([NODE, "-e", script], cwd=ROOT, capture_output=True, text=True, check=True)
    path = save_recording("s1", "Y1", base64.b64decode(result.stdout), LIST, tmp_path)
    samples, rate = read_sound(path)
    assert rate == 48000 and len(samples) == 48000 and np.abs(samples).max() == pytest.approx(0.3, abs=0.01)


# ---- the four services, over real WebSockets -----------------------------------------


def heard():
    result = Heard(asyncio.get_running_loop().time)
    result.started = result.clock()
    return result


async def pieces(*items):
    for item in items:
        yield item


class FakeService:
    """A local WebSocket server that keeps what it was told and runs a script."""

    def __init__(self, script):
        self.script = script
        self.path, self.headers, self.received = None, None, []

    async def handle(self, ws):
        self.path, self.headers = ws.request.path, ws.request.headers
        await self.script(ws, self)

    async def __aenter__(self):
        self.server = await serve(self.handle, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}/listen"
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()


async def test_sarvam_sends_base64_audio_and_reads_partial_and_final_words():
    async def script(ws, fake):
        async for raw in ws:
            message = json.loads(raw)
            fake.received.append(message)
            if message["event"] == "audio_input":
                await ws.send(json.dumps({"event": "transcript.partial", "utterance_idx": 0, "text": "ru"}))
            elif message["event"] == "end":
                await ws.send(json.dumps({"event": "transcript.final", "utterance_idx": 0, "text": "ruko"}))
                await ws.send(json.dumps({"event": "session.end"}))
                return

    async with FakeService(script) as fake:
        result = heard()
        await Sarvam("the-key", fake.url).stream(pieces(b"\x01\x02"), result)
    assert result.text == "ruko" and result.last_partial == "ru" and result.partials == 1
    assert fake.received == [{"event": "audio_input", "audio": base64.b64encode(b"\x01\x02").decode()}, {"event": "end"}]
    assert fake.headers["API-SUBSCRIPTION-KEY"] == "the-key"
    for part in ("model=saaras%3Av3-realtime", "mode=translit", "sample_rate=8000", "encoding=linear16", "language_code=hi-IN"):
        assert part in fake.path


async def test_sarvams_speed_settings_are_only_sent_when_chosen():
    async def script(ws, fake):
        await ws.send(json.dumps({"event": "session.end"}))

    async with FakeService(script) as fake:
        await Sarvam("k", fake.url).stream(pieces(), heard())
        assert "silence_duration_ms" not in fake.path and "stream_type=balanced" in fake.path
        await Sarvam("k", fake.url, stream_type="fast", silence_ms=250).stream(pieces(), heard())
        assert "silence_duration_ms=250" in fake.path and "stream_type=fast" in fake.path


@pytest.mark.parametrize("close_code, retry", [(1011, True), (1003, False)])
async def test_sarvams_close_codes_are_judged_right(close_code, retry):
    async def script(ws, fake):
        await ws.close(close_code, "x")

    async with FakeService(script) as fake:
        with pytest.raises(EarsError) as caught:
            await Sarvam("k", fake.url).stream(pieces(b"\x00\x00"), heard())
    assert worth_retrying(caught.value) is retry


async def test_a_fatal_sarvam_error_comes_out_with_its_status_and_a_harmless_one_doesnt():
    async def script(ws, fake):
        await ws.send(json.dumps({"event": "error", "is_fatal": False, "code": "invalid_config", "message": "meh"}))
        await ws.send(json.dumps({"event": "error", "is_fatal": True, "message": "slow down", "status_code": 429}))
        await ws.wait_closed()

    async with FakeService(script) as fake:
        with pytest.raises(EarsError, match="slow down") as caught:
            await Sarvam("k", fake.url).stream(pieces(b"\x00\x00"), heard())
    assert caught.value.status_code == 429 and worth_retrying(caught.value)


async def test_soniox_sets_up_first_ends_with_an_empty_text_frame_and_keeps_words_whole():
    async def script(ws, fake):
        fake.received.append(json.loads(await ws.recv()))
        fake.received.append(await ws.recv())
        await ws.send(json.dumps({"tokens": [{"text": "ha", "is_final": False}]}))
        fake.received.append(await ws.recv())
        await ws.send(json.dumps({"tokens": [{"text": "ha", "is_final": True}, {"text": "an", "is_final": True}]}))
        await ws.send(json.dumps({"tokens": [{"text": " ruko", "is_final": True}]}))
        await ws.send(json.dumps({"tokens": [], "finished": True}))

    async with FakeService(script) as fake:
        result = heard()
        await Soniox("the-key", fake.url).stream(pieces(b"\x01\x02"), result)
    config, audio, end = fake.received
    assert config == {"model": "stt-rt-v5", "audio_format": "s16le", "sample_rate": 8000, "num_channels": 1,
                      "language_hints": ["hi", "en"]}
    assert audio == b"\x01\x02" and end == ""  # an empty binary frame wouldn't end the stream
    assert result.text == "haan ruko" and result.last_partial == "ha"  # no space put inside "haan"
    assert fake.headers["Authorization"] == "Bearer the-key"


async def test_a_soniox_error_comes_out_with_its_status():
    async def script(ws, fake):
        await ws.recv()
        await ws.send(json.dumps({"tokens": [], "error_code": 503, "error_type": "busy", "error_message": "later"}))

    async with FakeService(script) as fake:
        with pytest.raises(EarsError, match="busy") as caught:
            await Soniox("k", fake.url).stream(pieces(b"\x00\x00"), heard())
    assert worth_retrying(caught.value)


async def test_deepgram_keeps_interim_and_final_apart_and_closes_properly():
    def result_message(text, final):
        return json.dumps({"type": "Results", "is_final": final, "channel": {"alternatives": [{"transcript": text}]}})

    async def script(ws, fake):
        await ws.send(json.dumps({"type": "Metadata"}))
        fake.received.append(await ws.recv())
        await ws.send(result_message("wai", False))
        fake.received.append(json.loads(await ws.recv()))
        await ws.send(result_message("wait", True))
        await ws.send(result_message("", True))  # an empty final is not a word

    async with FakeService(script) as fake:
        result = heard()
        await Deepgram("the-key", fake.url).stream(pieces(b"\x01\x02"), result)
    assert result.text == "wait" and result.last_partial == "wai" and result.finals == ["wait"]
    assert fake.received == [b"\x01\x02", {"type": "CloseStream"}]
    assert fake.headers["Authorization"] == "Token the-key"
    for part in ("model=nova-3", "language=multi", "encoding=linear16", "sample_rate=8000", "interim_results=true"):
        assert part in fake.path


async def test_a_refused_connection_keeps_its_status_and_wait_for_the_retry_rules():
    def refuse(connection, request):
        response = connection.respond(http.HTTPStatus.TOO_MANY_REQUESTS, "slow down\n")
        response.headers["Retry-After"] = "7"
        return response

    async def unused(ws):
        pass

    async with serve(unused, "127.0.0.1", 0, process_request=refuse) as server:
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        with pytest.raises(websockets.InvalidStatus) as caught:
            await Deepgram("k", url).stream(pieces(b"\x00\x00"), heard())
    assert worth_retrying(caught.value) and asked_wait_s(caught.value) == 7.0


class FakeLive:
    def __init__(self, turns):
        self.turns, self.sent, self.config, self.model = turns, [], None, None

    def connect(self, model, config):
        self.model, self.config = model, config
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def send_realtime_input(self, **kwargs):
        self.sent.append(kwargs)

    async def receive(self):
        if self.turns:
            for message in self.turns.pop(0):
                yield message


def live_message(partial_text=None, final=None):
    def text(value):
        return SimpleNamespace(text=value) if value is not None else None

    return SimpleNamespace(server_content=SimpleNamespace(interim_input_transcription=text(partial_text),
                                                          input_transcription=text(final)))


async def test_google_gets_16khz_reads_every_turn_and_ends_its_stream():
    live = FakeLive([[SimpleNamespace(server_content=None), live_message(partial_text="ha")],
                     [live_message(final="haan")], [live_message(final="ruko")]])
    result = heard()
    service = GoogleLive(SimpleNamespace(aio=SimpleNamespace(live=live)))
    await service.stream(pieces(b"\x01\x02"), result)
    assert service.rate == 16000 and live.model == "gemini-3.5-transcribe-live"
    assert result.text == "haan ruko" and result.last_partial == "ha"
    assert live.sent[0]["audio"].mime_type == "audio/pcm;rate=16000" and live.sent[0]["audio"].data == b"\x01\x02"
    assert live.sent[-1] == {"audio_stream_end": True}
    assert live.config.input_audio_transcription.mode.value == "VERBATIM"
    assert live.config.response_modalities[0].value == "TEXT"


# ---- the runner ----------------------------------------------------------------------


class Scripted:
    """A service that takes the pieces at the speed they come."""

    name, rate = "fake", 8000

    def __init__(self, text="ruko", hang=False, fail=None):
        self.text, self.hang, self.fail, self.got = text, hang, fail, []

    async def stream(self, stream, heard):
        if self.fail:
            raise self.fail
        async for piece in stream:
            self.got.append(piece)
            heard.partial(self.text[:1])
        heard.final(self.text)
        if self.hang:
            await asyncio.Event().wait()


async def test_audio_goes_at_real_speed_in_100_ms_pieces():
    service = Scripted()
    audio = bytes(8000)  # 0.5 s at 8 kHz
    loop = asyncio.get_running_loop()
    start = loop.time()
    result = await listen_once(service, audio)
    taken = loop.time() - start
    assert 0.38 <= taken < 1.0 and [len(p) for p in service.got] == [1600] * 5
    assert result.text == "ruko" and result.partials == 5 and result.first_text_at is not None


async def test_the_clock_starts_with_the_first_sound_not_with_opening_the_line():
    class SlowToConnect(Scripted):
        async def stream(self, stream, heard):
            await asyncio.sleep(0.4)  # the line takes this long to open
            async for piece in stream:
                heard.partial("ru")
                break
            heard.final("ruko")

    result = await listen_once(SlowToConnect(), bytes(1600), quiet_s=0.1)
    assert result.first_text_at < 0.2


async def test_a_service_that_never_closes_the_line_is_let_go_once_it_goes_quiet():
    result = await listen_once(Scripted(hang=True), bytes(1600), quiet_s=0.1)
    assert result.text == "ruko"


async def test_a_stuck_service_is_a_timeout_which_the_retry_rules_try_again(monkeypatch):
    monkeypatch.setattr("lab.ears_test.runner.STUCK_AFTER_S", 0.1)

    class NeverReads(Scripted):
        async def stream(self, stream, heard):
            await asyncio.Event().wait()

    with pytest.raises(TimeoutError) as caught:
        await listen_once(NeverReads(), bytes(1600), quiet_s=100)
    assert worth_retrying(caught.value)


async def test_an_error_from_a_service_comes_out_of_listening():
    with pytest.raises(RuntimeError, match="bad"):
        await listen_once(Scripted(fail=RuntimeError("bad")), bytes(1600))


def setup_clips(tmp_path, ids=("Y1", "Y2", "Y5")):
    make_recordings(tmp_path, ids=(*ids, "Y3", "Y7", "N1", "P1", "P2", "P3"))
    clips = prepare(tmp_path, tmp_path / "none")["clips"]
    return {f"s1/{line}": clips[f"s1/{line}"] for line in ids}


def fast_listen(answers, calls):
    """Stands in for streaming at real speed; answers can be errors to raise."""

    async def listen(service, audio):
        calls.append(len(audio))
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        result = Heard(lambda: 1.0)
        result.final(answer)
        return result

    return listen


async def no_sleep(_):
    pass


async def run(tmp_path, clips, answers, calls, logs=None, versions=("phone",), service=None):
    await run_service(service or Scripted(), clips, load_results(tmp_path / "r.json"), tmp_path / "r.json", tmp_path, versions,
                      (logs if logs is not None else []).append, no_sleep, fast_listen(answers, calls))
    return load_results(tmp_path / "r.json")


async def test_every_clip_is_heard_saved_as_it_goes_and_never_sent_twice(tmp_path):
    clips, calls = setup_clips(tmp_path), []
    results = await run(tmp_path, clips, ["a", "b", "c"], calls)
    assert results["fake/s1/Y1/phone"]["final"] == "a" and len(results) == 3 and len(calls) == 3
    calls.clear()
    await run(tmp_path, clips, [], calls)
    assert not calls


async def test_a_failed_clip_is_tried_again_on_the_next_run(tmp_path):
    clips, calls = setup_clips(tmp_path), []
    results = await run(tmp_path, clips, [RuntimeError("once"), "b", "c"], calls)
    assert "RuntimeError: once" in results["fake/s1/Y1/phone"]["error"]
    calls.clear()
    results = await run(tmp_path, clips, ["a"], calls)
    assert len(calls) == 1 and results["fake/s1/Y1/phone"]["final"] == "a"


class Limited(Exception):
    status_code = 429


async def test_a_daily_limit_stops_the_service_after_one_request(tmp_path):
    clips, calls, logs = setup_clips(tmp_path), [], []
    results = await run(tmp_path, clips, [Limited("Please retry in 24h23m46s"), "x", "y"], calls, logs)
    assert len(calls) == 1 and len(results) == 1 and any("asks to wait 24.4 h" in line for line in logs)


async def test_a_short_limit_is_waited_out_and_logged(tmp_path):
    clips, calls, logs = setup_clips(tmp_path, ids=("Y1",)), [], []
    results = await run(tmp_path, clips, [Limited("Please retry in 54s"), "a"], calls, logs)
    assert results["fake/s1/Y1/phone"]["final"] == "a" and results["fake/s1/Y1/phone"]["retries"] == 1
    assert any("retrying in 54 s" in line for line in logs)


async def test_the_same_lasting_failure_stops_the_service_after_three_clips(tmp_path):
    make_recordings(tmp_path, ids=("Y1", "Y2", "Y5", "Y8", "Y3", "Y7"))
    clips = prepare(tmp_path, tmp_path / "none")["clips"]
    calls, logs = [], []
    await run(tmp_path, clips, [EarsError("bad key")] * 6, calls, logs)
    assert len(calls) == 3 and any("3 failures in a row" in line for line in logs)


async def test_a_clip_that_works_clears_the_count_of_failures(tmp_path):
    # Only failures in a row mean something lasting; scattered ones must not stop the run.
    make_recordings(tmp_path, ids=("Y1", "Y2", "Y5", "Y8", "Y3", "Y7"))
    clips = prepare(tmp_path, tmp_path / "none")["clips"]
    calls = []
    answers = [EarsError("x"), EarsError("x"), "ok", EarsError("x"), EarsError("x"), "ok"]
    await run(tmp_path, clips, answers, calls)
    assert len(calls) == 6


def test_speed_is_timed_from_the_speech_not_from_the_start_of_the_clip():
    result = Heard(lambda: 0.0)
    result.first_text_at, result.last_final_at = 1.0, 2.5
    row_ = _row(result, {"speech_start_s": 0.3, "speech_end_s": 2.0}, 0)
    assert row_["first_text_s"] == 0.7 and row_["final_after_speech_s"] == 0.5


async def test_google_is_sent_clips_raised_to_its_rate(tmp_path):
    clips, calls = setup_clips(tmp_path, ids=("Y1",)), []

    class Wide(Scripted):
        rate = 16000

    await run(tmp_path, clips, ["a"], calls, service=Wide())
    assert calls[0] == pytest.approx(2.8 * 16000 * 2, abs=8)


async def test_a_whole_clip_goes_through_a_fake_service_to_a_score(tmp_path):
    # The whole path: a recording, its phone version, a real stream, a mark, the report.
    clips = setup_clips(tmp_path, ids=("Y5",))
    logs = []
    await run_service(Scripted("Ruko"), clips, {}, tmp_path / "r.json", tmp_path, ("phone",), logs.append, no_sleep,
                      partial(listen_once, quiet_s=0.1))
    results = load_results(tmp_path / "r.json")
    score = score_service("fake", LIST, results, clips)
    assert str(score.details) == "1/1 (100%)" and score.gave_partials == 1 and score.first_text_s
    assert "| fake | 1/1 (100%)" in report([score], 1, {}, "babble", "today")


# ---- the command -----------------------------------------------------------------------


def test_a_service_without_a_key_is_skipped_and_named_and_no_key_is_printed(monkeypatch):
    from cli import ears_test

    monkeypatch.setattr(ears_test, "load_env", lambda: None)
    monkeypatch.setenv("SARVAM_API_KEY", "secret-value")
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    services, skipped = ears_test.make_services(["sarvam", "soniox", "deepgram"])
    assert [s.name for s in services] == ["sarvam"]
    assert skipped == ["soniox (no SONIOX_API_KEY in .env)", "deepgram (no DEEPGRAM_API_KEY in .env)"]
    assert "secret-value" not in " ".join(skipped)


def test_real_voices_and_results_stay_out_of_git():
    for path in ("lab/sounds/ears/raw/s1/Y1.wav", "runs/week1/ears_test/results.json"):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0, path
