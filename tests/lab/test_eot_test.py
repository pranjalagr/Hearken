"""The end-of-turn test's clips and review page, with a made-up voice (no requests)."""

import json
import threading
import urllib.error
import urllib.request
from collections import Counter

import numpy as np
import pytest
import soundfile

from lab.eot_test.clips import (
    GUARD_S,
    LEAD_S,
    MIXES,
    TRAIL_S,
    ClipPlan,
    build_pair,
    clip_path,
    find_pause,
    load_lines,
    load_manifest,
    make_clips,
    on_the_phone,
    plan_clips,
)
from lab.ears_test.record_server import make_server as make_record_server
from lab.eot_test.human import build_human_pair, find_human_pause, followed_by_quiet, human_lines, prepare_human
from lab.eot_test.models import WINDOW, window_of
from lab.eot_test.review_server import load_verdicts, make_server, save_verdict
from lab.eot_test.score import FINISHED, UNFINISHED, Row, approved, choose_point, cross_marked, mark, report, run_models, separation, until_quiet_for

RATE = 8000


def tone(seconds):
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def quiet(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def speech(*parts):
    """Positive numbers are seconds of sound, negative are seconds of quiet."""
    return np.concatenate([tone(p) if p > 0 else quiet(-p) for p in parts])


def test_the_real_lines_each_have_one_pause_and_are_all_different():
    lines = load_lines()
    assert len(lines) == 60 and len(set(lines)) == 60


def test_each_line_gets_two_voices_and_the_sounds_are_spread_evenly():
    plans = plan_clips(load_lines())
    assert len(plans) == 120 and len({p.id for p in plans}) == 120
    assert all(n == 2 for n in Counter(p.line for p in plans).values())
    assert set(Counter((p.noise, p.level) for p in plans).values()) == {120 // len(MIXES)}


def test_the_longest_pause_is_found_and_edges_dont_count():
    voice = speech(-0.2, 0.5, -0.2, 0.5, -0.6, 0.5, -0.3)
    first, start, end, last = find_pause(voice, RATE)
    assert first == pytest.approx(0.2 * RATE, abs=320) and last == pytest.approx(2.5 * RATE, abs=320)
    assert start == pytest.approx(1.4 * RATE, abs=320) and end == pytest.approx(2.0 * RATE, abs=320)


def test_no_pause_or_a_pause_at_the_very_edge_is_no_pause():
    assert find_pause(tone(2.0), RATE) is None
    assert find_pause(speech(0.1, -0.5, 1.0), RATE) is None  # too little speech before it
    assert find_pause(quiet(1.0), RATE) is None


def test_the_unfinished_clip_stops_at_the_pause_and_both_end_in_the_same_quiet():
    voice = speech(-0.1, 0.8, -0.4, 0.6, -0.2)
    unfinished, finished, unfinished_end, finished_end = build_pair(voice, RATE, 1.0)
    assert unfinished_end == pytest.approx(LEAD_S + 0.8 + GUARD_S, abs=0.03)
    assert finished_end == pytest.approx(LEAD_S + 0.8 + 1.0 + 0.6 + GUARD_S, abs=0.05)
    for audio, end in ((unfinished, unfinished_end), (finished, finished_end)):
        assert len(audio) / RATE == pytest.approx(end + TRAIL_S, abs=0.001)
        assert np.abs(audio[int(end * RATE) + 1:]).max() == 0  # nothing after the speech but quiet
    held = finished[int((LEAD_S + 0.8 + GUARD_S + 0.05) * RATE):int((LEAD_S + 0.8 + 0.95) * RATE)]
    assert np.abs(held).max() == 0  # the pause is the length asked for


def test_background_sound_plays_through_the_pause(tmp_path):
    bed = np.random.default_rng(1).normal(0, 0.1, 8000)
    soundfile.write(tmp_path / "traffic.wav", bed, RATE)
    plan = ClipPlan("L01_aviraj", 1, "aviraj", "a... b", "traffic", "medium", 0.5, 100)
    assert np.abs(on_the_phone(quiet(1.0), plan, tmp_path)).max() > 0
    clean = ClipPlan("L01_aviraj", 1, "aviraj", "a... b", None, None, 0.5, 100)
    assert np.abs(on_the_phone(quiet(1.0), clean, tmp_path)).max() < 0.001


async def test_clips_are_made_and_a_bad_one_is_reported_and_left_out(tmp_path):
    plans = [ClipPlan(f"L0{n}_aviraj", n, "aviraj", "a... b", None, None, 0.5, 0) for n in (1, 2, 3)]

    async def raw(plan):
        if plan.line == 2:
            raise ConnectionError("no line")
        return (tone(2.0) if plan.line == 3 else speech(0.8, -0.4, 0.6)), RATE

    logged = []
    made = await make_clips(plans, raw, tmp_path, log=logged.append)
    assert [c.plan.line for c in made] == [1]
    assert clip_path(tmp_path, "L01_aviraj", "unfinished").exists() and clip_path(tmp_path, "L01_aviraj", "finished").exists()
    assert any("L02" in line and "no recording" in line for line in logged) and any("L03" in line and "no usable pause" in line for line in logged)
    assert [c["plan"]["id"] for c in load_manifest(tmp_path)] == ["L01_aviraj"]


async def test_a_run_of_failures_stops_asking(tmp_path):
    plans = [ClipPlan(f"L{n:02d}_aviraj", n, "aviraj", "a... b", None, None, 0.5, 0) for n in range(1, 11)]
    asked = []

    async def raw(plan):
        asked.append(plan.id)
        raise ConnectionError("out of credit")

    await make_clips(plans, raw, tmp_path, log=lambda line: None, give_up_after=3)
    assert len(asked) == 3


def test_verdicts_are_kept_and_nonsense_is_refused(tmp_path):
    save_verdict("a", "ok", {"a", "b"}, tmp_path)
    save_verdict("b", "reject", {"a", "b"}, tmp_path)
    save_verdict("a", "reject", {"a", "b"}, tmp_path)  # changing your mind replaces it
    assert load_verdicts(tmp_path) == {"a": "reject", "b": "reject"}
    with pytest.raises(ValueError):
        save_verdict("zzz", "ok", {"a"}, tmp_path)
    with pytest.raises(ValueError):
        save_verdict("a", "maybe", {"a"}, tmp_path)


@pytest.fixture
async def page(tmp_path):
    plans = [ClipPlan("L01_aviraj", 1, "aviraj", "a... b", None, None, 0.5, 0)]

    async def raw(plan):
        return speech(0.8, -0.4, 0.6), RATE

    await make_clips(plans, raw, tmp_path, log=lambda line: None)
    server = make_server(0, tmp_path)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", tmp_path
    server.shutdown()
    server.server_close()


def _status(url, **kwargs):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, **kwargs), timeout=5) as reply:
            return reply.status, reply.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


async def test_the_page_lists_plays_and_saves(page):
    base, sounds = page
    status, body = _status(f"{base}/clips")
    assert status == 200 and json.loads(body)[0]["id"] == "L01_aviraj"
    status, body = _status(f"{base}/audio/L01_aviraj_unfinished.wav")
    assert status == 200 and body[:4] == b"RIFF"
    status, _ = _status(f"{base}/verdict?clip=L01_aviraj&verdict=ok", method="POST")
    assert status == 200 and load_verdicts(sounds) == {"L01_aviraj": "ok"}
    status, body = _status(f"{base}/")
    assert status == 200 and b"EOT clip review" in body


async def test_the_page_refuses_other_paths_and_other_websites(page):
    base, sounds = page
    assert _status(f"{base}/audio/..%2F..%2Fverdicts.json")[0] == 404
    assert _status(f"{base}/audio/L99_aviraj_finished.wav")[0] == 404
    status, _ = _status(f"{base}/verdict?clip=L01_aviraj&verdict=ok", method="POST", headers={"Origin": "http://evil.example"})
    assert status == 403 and load_verdicts(sounds) == {}
    assert _status(f"{base}/verdict?clip=nope&verdict=ok", method="POST")[0] == 400


def _row(line, kind, chance, model="m", wait=0.2):
    return Row(model, wait, f"L{line:02d}_v", line, kind, "v", "none", chance, 1.0)


def test_only_pairs_a_person_approved_are_used():
    manifest = [{"plan": {"id": "a"}}, {"plan": {"id": "b"}}, {"plan": {"id": "c"}}]
    assert [c["plan"]["id"] for c in approved(manifest, {"a": "ok", "b": "reject"})] == ["a"]


def test_every_clip_ends_in_the_same_quiet():
    audio = np.zeros(RATE * 5, dtype=np.float32)
    assert len(until_quiet_for(audio, RATE, 1.0, 0.2)) == int(1.2 * RATE)
    assert len(until_quiet_for(audio, RATE, 3.0, 0.5)) == int(3.5 * RATE)


def test_a_short_turn_is_padded_on_the_right_side_and_a_long_one_keeps_its_end():
    short = np.ones(100, dtype=np.float32)
    assert window_of(short, True)[-1] == 1 and window_of(short, True)[0] == 0 and len(window_of(short, True)) == WINDOW
    assert window_of(short, False)[0] == 1 and window_of(short, False)[-1] == 0
    long = np.arange(WINDOW + 50, dtype=np.float32)
    assert window_of(long, True)[-1] == WINDOW + 49 and window_of(long, True)[0] == 50


def test_the_point_keeps_cutoffs_within_the_limit_and_waits_as_few_as_it_can():
    unfinished = [i / 100 for i in range(100)]  # 0.00 to 0.99
    point = choose_point(unfinished, 0.05)
    assert sum(c >= point for c in unfinished) == 5  # exactly the five allowed
    assert choose_point(unfinished, 0.0) > 0.99  # none allowed: above the highest
    assert choose_point([0.5], 1.0) == 0.0  # all allowed: no wait needed


def test_a_model_is_never_marked_on_the_sentences_its_point_came_from():
    # Odd sentences would give a point of 0.9, even ones 0.1. If a half were marked on its
    # own point its cut-offs would be 0; marked on the other half's point, odd ones are all cut off.
    rows = [_row(n, UNFINISHED, 0.85 if n % 2 else 0.05) for n in range(1, 41)] + [_row(n, FINISHED, 0.95) for n in range(1, 41)]
    got, points, _ = cross_marked(rows, 0.0)
    assert got.unfinished == 40 and got.finished == 40
    assert got.cutoffs == 20  # the odd half, judged by the point the even half chose
    assert min(points) < 0.1 < 0.8 < max(points)


def test_marking_counts_each_mistake_on_its_own_side():
    rows = [_row(1, UNFINISHED, 0.9), _row(2, UNFINISHED, 0.1), _row(3, FINISHED, 0.9), _row(4, FINISHED, 0.1)]
    got = mark(rows, 0.5)
    assert (got.cutoffs, got.wasted) == (1, 1)
    assert separation([_row(1, UNFINISHED, 0.1), _row(2, FINISHED, 0.9)]) == 1.0
    assert separation([_row(1, UNFINISHED, 0.9), _row(2, FINISHED, 0.1)]) == 0.0


async def test_the_models_are_asked_at_each_wait_and_the_report_names_them(tmp_path):
    plans = [ClipPlan(f"L0{n}_aviraj", n, "aviraj", "a... b", None, None, 0.5, 0) for n in (1, 2)]

    async def raw(plan):
        return speech(0.8, -0.4, 0.6), RATE

    await make_clips(plans, raw, tmp_path, log=lambda line: None)
    lengths = []

    def model(audio):
        lengths.append(len(audio))
        return 0.7

    rows = run_models(load_manifest(tmp_path), {"fake": model}, tmp_path, log=lambda line: None)
    assert len(rows) == 2 * 2 * 2  # two clips, two kinds, two waits
    assert len(set(lengths)) > 1 and all(r.ms >= 0 for r in rows)
    text = report(rows, "now")
    assert "fake" in text and "Cut-offs" in text and "week 11" in text


def _talking(*parts, rate=RATE):
    """Like speech() but with sound that never repeats, as the recording check requires."""
    noise = np.random.default_rng(3)
    return np.concatenate([noise.normal(0, 0.15, int(p * rate)).astype(np.float32) if p > 0 else np.zeros(int(-p * rate), dtype=np.float32) for p in parts])


def _wav(samples, rate=RATE):
    import io
    out = io.BytesIO()
    soundfile.write(out, samples, rate, format="WAV", subtype="PCM_16")
    return out.getvalue()


def test_two_thirds_of_the_lines_are_recorded_from_every_group():
    lines = human_lines()
    assert len(lines) == 40 and len({l.id for l in lines}) == 40 and all("⏸" in l.say and "..." not in l.say for l in lines)
    numbers = {l.number for l in lines}
    assert all(numbers & set(group) for group in (range(1, 16), range(16, 28), range(28, 40), range(40, 51), range(51, 61)))


def test_the_longest_pause_is_chosen_and_edges_and_clicks_dont_count():
    assert find_human_pause([(0.3, 1.3), (1.5, 2.5), (3.4, 4.4)]) == (0.3, 2.5, 3.4, 4.4)
    assert find_human_pause([(0.3, 1.5)]) is None  # one stretch: no pause
    assert find_human_pause([(0.3, 1.3), (1.35, 2.3)]) is None  # too short to be a pause
    assert find_human_pause([(0.3, 0.4), (1.0, 2.5)]) is None  # too little speech before it
    assert find_human_pause([(0.1, 0.13), (1.0, 2.0), (2.5, 3.5)]) == (1.0, 2.0, 2.5, 3.5)  # a click isn't speech


def test_a_short_natural_pause_is_enough_and_the_next_words_never_leak_into_the_cut():
    voice = _talking(-0.3, 1.0, -0.2, 1.0, -0.3)
    unfinished, finished, cut_s, stop_s, pause_s = build_human_pair(voice, (0.3, 1.3, 1.5, 2.5))
    assert pause_s == pytest.approx(0.2) and cut_s == pytest.approx(1.34) and stop_s == pytest.approx(2.54)
    for audio, end in ((unfinished, cut_s), (finished, stop_s)):
        assert len(audio) / RATE == pytest.approx(end + 0.6, abs=0.01)
        assert np.abs(audio[int((end + 0.02) * RATE):]).max() < 0.05  # quiet only: 0.15-level speech would show


def test_the_quiet_is_what_really_followed_then_more_of_the_room():
    out = followed_by_quiet(np.ones(2), np.array([0.5, 0.5]), np.array([0.1]), 5)
    assert out.tolist() == pytest.approx([1, 1, 0.5, 0.5, 0.1, 0.1, 0.1])
    assert len(followed_by_quiet(np.ones(2), np.zeros(9), np.array([0.1]), 5)) == 7  # long pauses are trimmed


def test_a_take_with_no_pause_is_left_out_and_reported_not_refused(tmp_path):
    (tmp_path / "raw" / "s1").mkdir(parents=True)
    lines = human_lines()
    soundfile.write(tmp_path / "raw" / "s1" / f"{lines[0].id}.wav", _talking(-0.3, 1.0, -0.2, 1.0, -0.3), RATE, subtype="PCM_16")
    soundfile.write(tmp_path / "raw" / "s1" / f"{lines[1].id}.wav", _talking(2.0), RATE, subtype="PCM_16")

    def detect(audio):
        return [(0.3, 1.3), (1.5, 2.5)] if len(audio) > 2.2 * 16000 else [(0.0, 2.0)]

    logged = []
    made = prepare_human("s1", tmp_path, log=logged.append, detect=detect)
    assert [c.plan.line for c in made] == [lines[0].number]
    assert any(lines[1].id in line and "left out" in line for line in logged)
    assert load_verdicts(tmp_path) == {f"{lines[0].id}_s1": "ok"}


def test_the_record_page_shows_its_own_instructions_and_takes_any_pause(tmp_path):
    lines = human_lines()
    index = tmp_path / "page.html"
    index.write_text("<title>EOT recording</title>", encoding="utf-8")
    server = make_record_server(0, tmp_path, lines, index)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"EOT recording" in _status(f"{base}/")[1]
        assert _status(f"{base}/record.js")[0] == 200  # the shared recording code still comes through
        listed = json.loads(_status(f"{base}/lines?speaker=s1")[1])
        assert len(listed) == 40 and "⏸" in listed[0]["say"]
        short_pause = _wav(_talking(-0.3, 1.0, -0.2, 1.0, -0.3))
        assert _status(f"{base}/save?speaker=s1&line={lines[0].id}", method="POST", data=short_pause)[0] == 200
        assert (tmp_path / "raw" / "s1" / f"{lines[0].id}.wav").exists()
    finally:
        server.shutdown()
        server.server_close()
