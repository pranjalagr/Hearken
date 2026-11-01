"""The week-1 voice test, without the network: its lines, both service
adapters (with fake responses shaped like the real ones, and the real Google
library over a fake network), timeouts, retries, resuming, and the listening
page."""

import base64
import io
import json
import urllib.error
from datetime import datetime
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from engine.contract.spoken import LANGUAGES
from lab.week1.voice_test import (
    CATEGORIES,
    REQUEST_TIMEOUT_S,
    SCREEN_IDS,
    SETUPS,
    SETUPS_BY_NAME,
    Spoken,
    VoiceLine,
    asked_wait_s,
    google_client,
    google_speak,
    listening_page,
    load_lines,
    run_round,
    sarvam_request,
    sarvam_speak,
    timing_summary,
    with_retries,
    worth_retrying,
    write_page,
)
from packs.food.pack import FoodPack

LINES = {line.id: line for line in load_lines()}
SARVAM = SETUPS_BY_NAME["sarvam-sunny-enhi"]
GOOGLE = SETUPS_BY_NAME["google-csagent-1"]


# ---- the lines -----------------------------------------------------------------


def test_76_lines_in_every_group_with_unique_ids():
    assert len(LINES) == 76
    assert {line.category for line in LINES.values()} == set(CATEGORIES)
    assert set(SCREEN_IDS) <= set(LINES)
    assert all(line.listen for line in LINES.values())


def test_the_money_fix_lines_all_say_the_amount_that_failed():
    """E8-E11 are only worth comparing with E3 if they carry the same amount."""
    assert "₹249.50" in LINES["E3"].text and "₹249.50" in LINES["E8"].text
    for line_id in ("E9", "E10", "E11"):
        assert "249" in LINES[line_id].text and "50 paise" in LINES[line_id].text
        assert "." not in LINES[line_id].text.rstrip(".")  # no decimal point left to misread
    assert [LINES[i].lang for i in ("E3", "E8", "E9", "E10", "E11")] == ["hi", "en", "hi", "hi", "en"]


def test_only_the_script_pairs_use_hindi_letters():
    """Decision 12: English letters everywhere, except the lines that test that rule."""
    devanagari = {i for i, line in LINES.items() if any("ऀ" <= ch <= "ॿ" for ch in line.text)}
    assert devanagari == {"N2", "N4", "N6"}


async def test_the_yes_questions_are_exactly_what_the_backend_says(open_call):
    """Prepared through the real backend, so the voice test never drifts
    from what the robot will really ask (decision 9)."""
    kit = await open_call(FoodPack())

    async def asked(tool, **inputs):
        prepared = await kit.tool(tool, **inputs)
        assert prepared.status == "ok", prepared.message
        return {language: await kit.guard.question(prepared.data["token"], language) for language in LANGUAGES}

    cancel_4521 = await asked("prepare_cancel_order", order_no=4521, customer_name="Rahul Verma")
    cancel_4530 = await asked("prepare_cancel_order", order_no=4530, customer_name="Aman Gupta")
    cancel_4533 = await asked("prepare_cancel_order", order_no=4533, customer_name="Neha Singh")
    refund_4512 = await asked(
        "prepare_refund_order", order_no=4512, customer_name="Priya Sharma", amount_rupees=40, reason="missing_item"
    )
    move_4561 = await asked(
        "prepare_change_address", order_no=4561, customer_name="Priya Sharma",
        new_address="Flat 2C, Rose Apartments, Model Town, Ludhiana",
    )
    paise_4512 = await asked(  # decision 10: an amount with paise
        "prepare_refund_order", order_no=4512, customer_name="Priya Sharma", amount_rupees=120.5, reason="damaged_item"
    )
    assert LINES["F1"].text == cancel_4521["hinglish"]
    assert LINES["F2"].text == cancel_4530["hinglish"]
    assert LINES["F3"].text == cancel_4533["hinglish"]
    assert LINES["F4"].text == refund_4512["hinglish"]
    assert LINES["F5"].text == move_4561["hinglish"]
    assert LINES["F6"].text == cancel_4521["english"]
    assert LINES["F7"].text == move_4561["english"]
    assert LINES["F8"].text == paise_4512["hinglish"]
    assert LINES["F9"].text == paise_4512["english"]


def test_the_status_lines_are_exactly_the_packs_phrases():
    words = FoodPack().spoken_words()
    statuses = ("placed", "preparing", "out_for_delivery", "delivered", "cancelled")
    for n, status in enumerate(statuses, 1):
        assert LINES[f"G{n}"].text == words.phrase("status", status, "hinglish")
    assert LINES["G6"].text == words.phrase("status", "preparing", "english")


def test_every_candidate_is_a_male_voice_from_the_live_lists():
    assert {s.voice for s in SETUPS} == {
        "shubh_hi_customer", "sunny_enhi_customer", "en-in-csagent-1", "en-in-csagent-7",
    }
    assert len({s.name for s in SETUPS}) == len(SETUPS)


# ---- Sarvam --------------------------------------------------------------------


def test_the_sarvam_request_asks_for_raw_8khz_audio_with_preprocessing():
    request = sarvam_request(SARVAM, LINES["N2"], "test-key")
    body = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "https://api.sarvam.ai/text-to-speech/stream"
    assert request.get_header("Api-subscription-key") == "test-key"
    assert body == {
        "text": LINES["N2"].text,  # Hindi letters sent as real UTF-8, not \u escapes
        "language_code": "hi-IN",
        "model": "bulbul:v4-flash",
        "speaker": "sunny_enhi_customer",
        "speech_sample_rate": 8000,
        "output_audio_codec": "linear16",
        "enable_preprocessing": True,
    }
    assert "आपका".encode("utf-8") in request.data
    assert json.loads(sarvam_request(SARVAM, LINES["C1"], "k").data)["language_code"] == "en-IN"


class FakeResponse:
    def __init__(self, body: bytes, content_type: str = "audio/pcm"):
        self._body = io.BytesIO(body)
        self.headers = {"Content-Type": content_type}

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Clock:
    def __init__(self, *times):
        self._times = list(times)

    def __call__(self):
        return self._times.pop(0)


def test_sarvam_audio_and_its_timing_are_read_from_the_stream():
    audio = bytes(range(256)) * 10  # 2560 bytes, even
    spoken = sarvam_speak(SARVAM, LINES["D1"], "k", opener=lambda request, timeout: FakeResponse(audio),
                          clock=Clock(10.0, 10.4, 11.0))
    assert spoken == Spoken(audio, pytest.approx(0.4), pytest.approx(1.0))


@pytest.mark.parametrize(("body", "kind", "message"), [
    (b"\x00" * 100, "application/json", "expected raw PCM"),
    (b"\x00" * 101, "audio/pcm", "broken audio"),
    (b"", "audio/pcm", "broken audio"),
])
def test_sarvam_refuses_anything_but_clean_raw_audio(body, kind, message):
    with pytest.raises(RuntimeError, match=message):
        sarvam_speak(SARVAM, LINES["D1"], "k", opener=lambda request, timeout: FakeResponse(body, kind),
                     clock=Clock(0.0, 0.1, 0.2))


# ---- Google --------------------------------------------------------------------


def tone_24k(seconds: float) -> bytes:
    t = np.arange(int(seconds * 24000)) / 24000
    return (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes()


class FakeGoogle:
    """Shaped like the real stream checked on 1 Nov 2026: audio deltas that
    declare 24 kHz, mixed with other events."""

    def __init__(self, chunks, rate=24000):
        self.chunks, self.rate, self.asked = chunks, rate, None
        self.interactions = self

    def create(self, **kwargs):
        self.asked = kwargs
        yield SimpleNamespace(event_type="interaction.start", delta=None)
        for n, chunk in enumerate(self.chunks):
            rate = self.rate[n] if isinstance(self.rate, list) else self.rate
            delta = SimpleNamespace(type="audio", data=base64.b64encode(chunk).decode(), sample_rate=rate, channels=1)
            yield SimpleNamespace(event_type="step.delta", delta=delta)
        yield SimpleNamespace(event_type="interaction.complete", delta=None)


def test_google_audio_is_turned_into_8khz_and_timed_from_the_first_chunk():
    audio = tone_24k(0.6)
    fake = FakeGoogle([audio[:9600], audio[9600:]])
    spoken = google_speak(GOOGLE, LINES["A1"], fake, clock=Clock(5.0, 6.5, 7.0))
    assert len(spoken.pcm) // 2 == pytest.approx(0.6 * 8000, abs=2)  # resampled 24 kHz -> 8 kHz
    assert (spoken.first_audio_s, spoken.total_s) == (pytest.approx(1.5), pytest.approx(2.0))
    assert fake.asked["model"] == "gemini-3.8-flash-tts" and fake.asked["stream"] is True
    assert fake.asked["generation_config"] == {"speech_config": [{"voice": "en-in-csagent-1"}]}
    assert fake.asked["input"][0]["content"][0]["text"] == LINES["A1"].text


def test_google_uses_each_chunks_declared_rate_and_refuses_a_change_or_silence():
    with pytest.raises(RuntimeError, match="sample rate changed"):
        google_speak(GOOGLE, LINES["A1"], FakeGoogle([tone_24k(0.1), tone_24k(0.1)], rate=[24000, 16000]),
                     clock=Clock(0, 1, 2))
    with pytest.raises(RuntimeError, match="no audio"):
        google_speak(GOOGLE, LINES["A1"], FakeGoogle([]), clock=Clock(0, 1))
    eight_k = FakeGoogle([b"\x01\x00" * 800], rate=8000)  # if it ever does send 8 kHz, keep it as is
    assert len(google_speak(GOOGLE, LINES["A1"], eight_k, clock=Clock(0, 1, 2)).pcm) == 1600


# ---- the real Google library over a fake network ----------------------------------
# The library's own habits (hidden retries, no time limit, its error shapes)
# surprised us once, so these run the real library with only the network faked.

STREAM_HEADERS = {"content-type": "text/event-stream"}


def sse(event: dict) -> bytes:
    return f"event: {event['event_type']}\ndata: {json.dumps(event)}\n\n".encode()


def audio_events(pcm: bytes, pieces: int = 2) -> list[dict]:
    size = len(pcm) // pieces
    chunks = [pcm[n * size:(n + 1) * size] for n in range(pieces)]
    return [
        {"event_type": "interaction.created", "interaction": {"id": "x", "status": "in_progress"}},
        *({"event_type": "step.delta", "index": 0,
           "delta": {"type": "audio", "data": base64.b64encode(chunk).decode(), "mime_type": "audio/l16",
                     "sample_rate": 24000}} for chunk in chunks),
        {"event_type": "interaction.completed", "interaction": {"id": "x", "status": "completed"}},
    ]


class ReplyStream(httpx.SyncByteStream):
    def __init__(self, events, stalls=False):
        self.events, self.stalls = events, stalls

    def __iter__(self):
        for event in self.events:
            yield sse(event)
        if self.stalls:  # the server goes quiet mid-reply until the time limit
            raise httpx.ReadTimeout("no bytes for 30 s")


def fake_google(reply):
    """The real Google client over a fake network; reply(request) answers each
    request. Returns the client and every request it sent."""
    sent = []

    def handle(request):
        sent.append(request)
        return reply(request)

    return google_client("not-a-real-key", httpx.Client(transport=httpx.MockTransport(handle))), sent


def google_error(status):
    return lambda request: httpx.Response(status, json={"error": {"code": status, "message": "x", "status": "X"}})


def stuck_before_reply(request):
    raise httpx.ReadTimeout("no reply for 30 s", request=request)


def test_the_real_google_library_brings_audio_to_google_speak():
    client, sent = fake_google(
        lambda request: httpx.Response(200, headers=STREAM_HEADERS, stream=ReplyStream(audio_events(tone_24k(0.1)))))
    spoken = google_speak(GOOGLE, LINES["A1"], client)
    assert len(spoken.pcm) == 1600  # 0.1 s at 24 kHz -> 800 samples at 8 kHz
    body = json.loads(sent[0].content)
    assert body["model"] == "gemini-3.8-flash-tts" and body["stream"] is True
    assert body["response_format"] == {"type": "audio", "mime_type": "audio/l16", "sample_rate": 8000}
    assert sent[0].extensions["timeout"]["read"] == REQUEST_TIMEOUT_S  # a stuck reply can't hang the run


@pytest.mark.parametrize(("reply", "retry"), [
    (google_error(503), True),  # the library alone sends this 4 times, 6 s of hidden waiting
    (google_error(429), True),
    (google_error(400), False),  # a wrong voice name: retrying can't help
    (stuck_before_reply, True),
    (lambda request: httpx.Response(200, headers=STREAM_HEADERS,
                                    stream=ReplyStream(audio_events(tone_24k(0.1))[:2], stalls=True)), True),
], ids=["server-error", "rate-limit", "bad-request", "stuck-before-reply", "stuck-mid-stream"])
def test_google_failures_come_out_after_one_request_and_are_judged_right(reply, retry):
    client, sent = fake_google(reply)
    with pytest.raises(Exception) as caught:
        google_speak(GOOGLE, LINES["A1"], client)
    assert len(sent) == 1  # no hidden retries: with_retries sees and logs every one
    assert worth_retrying(caught.value) is retry


# ---- retries, resuming, the page ----------------------------------------------------


def http_error(code):
    return urllib.error.HTTPError("https://x", code, "msg", {}, None)


@pytest.mark.parametrize(("error", "retry"), [
    (http_error(429), True),
    (http_error(503), True),
    (http_error(400), False),  # a wrong speaker name
    (http_error(401), False),  # a bad key
    (TimeoutError("The read operation timed out"), True),  # urllib, mid-reply
    (urllib.error.URLError(TimeoutError("timed out")), True),  # urllib, while connecting
    (urllib.error.URLError("getaddrinfo failed"), False),  # no network: fail now, rerun later
    (RuntimeError("broken audio: 101 bytes"), False),
])
def test_sarvam_failures_are_judged_right(error, retry):
    assert worth_retrying(error) is retry


def test_sarvam_waits_no_longer_than_the_time_limit():
    asked = []
    sarvam_speak(SARVAM, LINES["D1"], "k", clock=Clock(0.0, 0.1, 0.2),
                 opener=lambda request, timeout: asked.append(timeout) or FakeResponse(b"\x00\x01" * 160))
    assert asked == [REQUEST_TIMEOUT_S]


def test_retries_wait_longer_each_time_are_all_heard_and_stop_at_a_lasting_error():
    waits, heard, calls = [], [], []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise http_error(429)
        if len(calls) == 2:
            raise TimeoutError("stuck")
        return "ok"

    def note(error, wait):
        heard.append((type(error).__name__, wait))

    assert with_retries(flaky, sleep=waits.append, on_retry=note) == "ok"
    assert waits == [15.0, 30.0] and heard == [("HTTPError", 15.0), ("TimeoutError", 30.0)]

    def wrong_voice():
        raise http_error(400)

    with pytest.raises(urllib.error.HTTPError):
        with_retries(wrong_voice, sleep=waits.append, on_retry=note)
    assert waits == [15.0, 30.0]  # no waiting for an error that won't go away

    def always_stuck():
        raise TimeoutError("stuck")

    with pytest.raises(TimeoutError):
        with_retries(always_stuck, sleep=waits.append, on_retry=note)
    assert waits[2:] == [15.0, 30.0, 60.0]  # three retries, then it gives up


def test_a_round_makes_missing_clips_skips_made_ones_and_records_failures(tmp_path):
    lines = [LINES["D1"], LINES["E3"]]
    made = []

    def speak(setup, line):
        made.append((setup.name, line.id))
        if line.id == "E3" and setup.company == "google":
            raise RuntimeError("voice refused")
        return Spoken(b"\x00\x01" * 800, 0.5, 1.0)

    setups = [SARVAM, GOOGLE]
    pauses = []
    timings = run_round(lines, setups, speak, out_dir=tmp_path, sleep=pauses.append, log=lambda text: None)
    assert (tmp_path / "sarvam-sunny-enhi" / "D1.wav").exists()
    # 1,600 bytes = 800 samples = 0.1 s at 8 kHz.
    assert timings["sarvam-sunny-enhi/D1"] == {"first_audio_s": 0.5, "total_s": 1.0, "seconds": 0.1, "retries": 0}
    assert timings["google-csagent-1/E3"] == {"error": "RuntimeError: voice refused", "retries": 0}
    assert pauses == [0.5, 6.0]  # between clips of one voice, never before the first
    made.clear()
    run_round(lines, setups, speak, out_dir=tmp_path, sleep=pauses.append, log=lambda text: None)
    assert made == [("google-csagent-1", "E3")]  # only the failed clip is tried again


DAILY = "Rate limit exceeded for model gemini-3.8-flash-tts (limit: 10 requests per day on Free Tier). " \
        "Please retry in 24h23m46s or upgrade your tier at https://ai.dev/rate-limit."
PER_MINUTE = "Rate limit exceeded for model gemini-3.8-flash-lite-tts (limit: 3 requests per minute on Free Tier). " \
             "Please retry in 54s or upgrade your tier at https://ai.dev/rate-limit."


def http_error_with(code, headers):
    return urllib.error.HTTPError("https://x", code, "msg", headers, None)


@pytest.mark.parametrize(("error", "seconds"), [
    (RuntimeError(DAILY), 24 * 3600 + 23 * 60 + 46),  # the real messages, 1 Nov 2026
    (RuntimeError(PER_MINUTE), 54),
    (RuntimeError("Please retry in 1m5s"), 65),
    (RuntimeError("Please retry in 2.5s"), 2.5),
    (RuntimeError("Please retry in 3h"), 3 * 3600),
    (http_error_with(429, {"Retry-After": "30"}), 30),  # the standard header
    (http_error(429), None),  # says nothing: our own waits apply
    (RuntimeError("retry in a moment"), None),
])
def test_the_wait_a_service_asks_for_is_read(error, seconds):
    assert asked_wait_s(error) == seconds


class RateLimited(Exception):
    """Shaped like the real Google error: the status in .status_code, the
    wait only in the message (checked with the real library below)."""

    status_code = 429


def test_a_per_minute_limit_is_waited_out_and_a_daily_one_fails_at_once():
    waits, heard, calls = [], [], []

    def per_minute():
        calls.append(1)
        if len(calls) == 1:
            raise RateLimited(PER_MINUTE)
        return "ok"

    assert with_retries(per_minute, sleep=waits.append, on_retry=lambda error, wait: heard.append(wait)) == "ok"
    assert waits == [54.0] and heard == [54.0]  # the service's 54 s, not our 15 s

    def daily():
        raise RateLimited(DAILY)

    with pytest.raises(RateLimited):
        with_retries(daily, sleep=waits.append)
    assert waits == [54.0]  # no waiting at all for a 24-hour limit


def test_the_real_google_library_carries_the_daily_wait_in_its_error():
    client, sent = fake_google(lambda request: httpx.Response(
        429, json={"error": {"message": DAILY, "code": "too_many_requests"}}))
    with pytest.raises(Exception) as caught:
        google_speak(GOOGLE, LINES["A1"], client)
    assert worth_retrying(caught.value) and asked_wait_s(caught.value) == 87826


def test_a_round_leaves_a_models_clips_unmade_once_it_runs_out_for_the_day(tmp_path):
    lite = SETUPS_BY_NAME["google-lite-csagent-1"]
    seven = SETUPS_BY_NAME["google-csagent-7"]  # the same model as GOOGLE
    asked = []

    def speak(setup, line):
        asked.append((setup.name, line.id))
        if setup.model == GOOGLE.model:
            raise RateLimited(DAILY)
        return Spoken(b"\x00\x01" * 800, 0.5, 1.0)

    said = []
    lines = [LINES["A1"], LINES["B4"]]
    timings = run_round(lines, [GOOGLE, seven, lite], speak, out_dir=tmp_path, sleep=lambda s: None, log=said.append)
    # One try at the out-of-quota model, then nothing more from it; the other model goes on.
    assert asked == [("google-csagent-1", "A1"), ("google-lite-csagent-1", "A1"), ("google-lite-csagent-1", "B4")]
    assert timings["google-csagent-1/A1"]["error"].startswith("RateLimited: Rate limit exceeded")
    assert "google-csagent-1/B4" not in timings and "google-csagent-7/A1" not in timings  # unmade, not failed
    assert any("asks to wait 24.4 h" in line for line in said)
    assert "  gemini-3.8-flash-tts: 3 clips left unmade (out of quota); run the same command again later" in said


def test_a_round_logs_and_counts_every_retry(tmp_path):
    tries = {"D1": 0, "E3": 0}

    def speak(setup, line):
        tries[line.id] += 1
        if line.id == "E3" or tries[line.id] == 1:  # D1 sticks once; E3 is always stuck
            raise TimeoutError("stuck")
        return Spoken(b"\x00\x01" * 800, 0.5, 1.0)

    said = []
    timings = run_round([LINES["D1"], LINES["E3"]], [SARVAM], speak, out_dir=tmp_path,
                        sleep=lambda seconds: None, log=said.append)
    assert timings["sarvam-sunny-enhi/D1"]["retries"] == 1
    assert timings["sarvam-sunny-enhi/E3"] == {"error": "TimeoutError: stuck", "retries": 3}
    assert "  sarvam-sunny-enhi/D1: retrying in 15 s after TimeoutError: stuck" in said
    assert not (tmp_path / "sarvam-sunny-enhi" / "E3.wav").exists()


def test_the_listening_page_shows_every_clip_and_escapes_text():
    tricky = VoiceLine("Z1", "everyday", "hi", 'Flat <2C> & "Rose"', "quotes & <tags>")
    timings = {"sarvam-sunny-enhi/Z1": {"first_audio_s": 0.4, "total_s": 1.0, "seconds": 1.2, "retries": 2},
               "google-csagent-1/Z1": {"error": "RuntimeError: voice refused"}}
    page = listening_page([tricky], [SARVAM, GOOGLE, SETUPS_BY_NAME["google-csagent-7"]], timings,
                          datetime(2026, 11, 1, 12, 0))
    assert 'src="sarvam-sunny-enhi/Z1.wav"' in page and "1 made, 0 failed, retries: 2" in page
    assert "Flat &lt;2C&gt; &amp; &quot;Rose&quot;" in page and "<2C>" not in page
    assert "voice refused" in page and "not made yet" in page
    assert page.count('name="sarvam-sunny-enhi__Z1"') == 2  # one ✅ and one ❌, grouped


def test_each_kind_of_run_writes_its_own_page(tmp_path):
    write_page([LINES["E3"]], [SARVAM], {}, tmp_path, name="listen-screen.html")
    write_page([LINES["E9"]], [SARVAM], {}, tmp_path, name="listen-lines.html")
    assert "E3" in (tmp_path / "listen-screen.html").read_text(encoding="utf-8")  # not replaced by the small run
    assert "E9" in (tmp_path / "listen-lines.html").read_text(encoding="utf-8")


def test_timing_summary_counts_only_this_runs_lines_and_all_their_retries():
    timings = {"sarvam-sunny-enhi/A1": {"first_audio_s": 0.3, "total_s": 1, "seconds": 1, "retries": 2},
               "sarvam-sunny-enhi/B4": {"first_audio_s": 0.7, "total_s": 1, "seconds": 1},  # made before retries were counted
               "sarvam-sunny-enhi/C1": {"error": "x", "retries": 3},
               "sarvam-sunny-enhi/D1": {"first_audio_s": 5.0, "total_s": 5, "seconds": 1},  # another run's line
               "google-csagent-1/A1": {"first_audio_s": 9, "total_s": 9, "seconds": 1, "retries": 1}}
    this_run = [LINES["A1"], LINES["B4"], LINES["C1"], LINES["E9"]]  # E9 not made yet
    assert timing_summary(SARVAM, timings, this_run) == {"clips": 2, "errors": 1, "retries": 5,
                                                         "median_first_s": 0.5, "slowest_first_s": 0.7}
