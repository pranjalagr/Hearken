"""The haan vs ruko phrase list and clip maker, with a made-up voice (no requests)."""

from collections import Counter
from pathlib import Path

import pytest
import yaml

import numpy as np
import torch

from lab.haan_ruko.clips import load_manifest, make_clips
from lab.haan_ruko.model import HaanRuko, OnnxModel, export_onnx
from lab.haan_ruko.score import Played, answer_at, check_times, mark, separation, zero_miss_line
from lab.haan_ruko.sound import FRAMES, LEAD_S, N_MELS, RATE, Roughness, log_mel, on_a_rough_line, window
from lab.haan_ruko.speech import Timing, opening_listening_word, time_clip
from lab.haan_ruko.windows import Helpers, training_windows
from lab.haan_ruko.phrases import KEEP, MIXED, SENTENCE_VOICES, STOP, VOICES, load_phrases, pick_sentences, plan_clips


def _phrases(tmp_path, **changes):
    data = {"keep": {"train": ["haan", "hmm"], "held_back": ["achha ji"]},
            "stop": {"train": ["ruko"], "held_back": ["hold on"]},
            "mixed": {"train": ["haan, lekin kab?"], "held_back": ["ok, ek sawaal"]},
            "sentences": {"from": [], "count": 10, "held_back_share": 0.2}}
    data.update(changes)
    path = tmp_path / "phrases.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_the_real_list_loads_and_has_every_group():
    data = load_phrases()
    for group in (KEEP, STOP, MIXED):
        assert data[group]["train"] and data[group]["held_back"]


def test_a_bare_yes_a_repeat_or_a_mixed_phrase_without_its_comma_is_refused(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("keep: {train: [yes], held_back: []}\nstop: {train: [], held_back: []}\nmixed: {train: [], held_back: []}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="quote"):
        load_phrases(path)
    with pytest.raises(ValueError, match="twice"):
        load_phrases(_phrases(tmp_path, stop={"train": ["Haan"], "held_back": []}))
    with pytest.raises(ValueError, match="comma"):
        load_phrases(_phrases(tmp_path, mixed={"train": ["haan lekin kab"], "held_back": []}))


def test_short_phrases_get_all_voices_and_sentences_two_each(tmp_path):
    plans = plan_clips(load_phrases(_phrases(tmp_path)), sentences=[f"sentence number {n}" for n in range(30)])
    by_group = Counter(p.group for p in plans)
    assert by_group == {"keep": 3 * len(VOICES), "stop": 2 * len(VOICES), "mixed": 2 * len(VOICES), "sentence": 10 * SENTENCE_VOICES}
    assert len({p.id for p in plans}) == len(plans)
    assert {p.answer for p in plans if p.group == "sentence"} == {STOP}
    # held back by phrase: every voice of a held-back phrase is held back, so no voice of it is trained on
    held_texts = {p.text for p in plans if p.held_back}
    assert not any(p.text in held_texts and not p.held_back for p in plans)
    assert sum(1 for p in plans if p.group == "sentence" and p.held_back) == 2 * SENTENCE_VOICES


def test_the_same_list_always_plans_the_same_clips(tmp_path):
    data = load_phrases(_phrases(tmp_path))
    pool = [f"sentence {n}" for n in range(30)]
    assert plan_clips(data, pool) == plan_clips(data, list(reversed(pool)))


def test_editing_the_list_keeps_the_other_clips_names(tmp_path):
    before = {p.text + p.voice: p.id for p in plan_clips(load_phrases(_phrases(tmp_path)), sentences=[])}
    edited = _phrases(tmp_path, keep={"train": ["ji", "haan", "hmm"], "held_back": ["achha ji"]})
    after = {p.text + p.voice: p.id for p in plan_clips(load_phrases(edited), sentences=[])}
    assert all(after[k] == v for k, v in before.items())


def test_picking_sentences_tidies_spaces_and_drops_repeats():
    trained, held = pick_sentences(["a  b", "a b", "c", " ", "d"], 10, 0.0)
    assert sorted(trained) == ["a b", "c", "d"] and held == []


async def test_clips_are_listed_with_their_answers_and_a_failure_is_reported(tmp_path):
    plans = plan_clips(load_phrases(_phrases(tmp_path)), sentences=[])[:4]

    async def make(plan):
        if plan.voice == VOICES[1]:
            raise ConnectionError("no line")
        path = tmp_path / "raw" / plan.voice / f"{plan.id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF")
        return path

    logged = []
    made = await make_clips(plans, make, tmp_path, log=logged.append)
    assert len(made) == 3 and all(m["answer"] == KEEP and m["file"].startswith("raw/") for m in made)
    assert any("no clip" in line for line in logged)
    assert load_manifest(tmp_path) == made


async def test_a_run_of_failures_stops_spending(tmp_path):
    plans = plan_clips(load_phrases(_phrases(tmp_path)), sentences=[])
    asked = []

    async def make(plan):
        asked.append(plan.id)
        raise ConnectionError("credit run out")

    await make_clips(plans, make, tmp_path, log=lambda line: None, give_up_after=3)
    assert len(asked) == 3


LISTENING = ["haan", "haan ji", "hmm", "achha"]


def test_a_listening_sound_is_timed_and_one_too_long_is_left_out():
    t = time_clip("a", KEEP, "haan", [(0.1, 0.5)], LISTENING)
    assert (t.start, t.end, t.why_left_out) == (0.1, 0.5, "")
    assert "too long" in time_clip("b", KEEP, "haan haan", [(0.1, 2.5)], LISTENING).why_left_out
    assert time_clip("c", KEEP, "hmm", [], LISTENING).why_left_out == "no speech found"


def test_a_mixed_clip_switches_at_the_pause_near_its_comma():
    # "haan, lekin kab aayega?": the comma is about a quarter of the way in
    t = time_clip("m", MIXED, "haan, lekin kab aayega?", [(0.1, 0.4), (0.6, 1.0), (1.05, 1.6)], LISTENING)
    assert t.switch == 0.6 and not t.why_left_out
    assert time_clip("m", MIXED, "haan, lekin kab aayega?", [(0.1, 1.6)], LISTENING).why_left_out == "no pause near the comma"


def test_speech_opening_with_a_listening_word_teaches_nothing_at_first():
    assert opening_listening_word("Haan ji, mera order", LISTENING) == "haan ji"
    assert opening_listening_word("Haanji bolo", LISTENING) is None  # a different word, not "haan"
    t = time_clip("s", STOP, "Haan ji, mera order kahan hai", [(0.1, 0.5), (0.7, 2.0)], LISTENING)
    assert t.ignore_until == pytest.approx(0.95)
    assert time_clip("s", STOP, "Order kahan hai", [(0.1, 2.0)], LISTENING).ignore_until is None


def test_checks_come_every_tenth_of_a_second_from_three_tenths_and_stop_when_the_caller_does():
    assert check_times(Timing("a", 0.1, 0.5)) == [0.4, 0.5, 0.6, 0.7]
    assert check_times(Timing("a", 0.0, 9.0))[-1] == 2.0  # never past 2 s of speech


def test_the_right_answer_at_each_moment():
    mixed = Timing("m", 0.1, 1.6, switch=0.6)
    assert answer_at(MIXED, mixed, 0.5) == 0 and answer_at(MIXED, mixed, 0.7) is None and answer_at(MIXED, mixed, 0.9) == 1
    opening = Timing("s", 0.1, 2.0, ignore_until=0.95)
    assert answer_at(STOP, opening, 0.5) is None and answer_at(STOP, opening, 1.0) == 1
    assert answer_at(KEEP, Timing("k", 0.1, 0.5), 0.4) == 0


def test_the_sound_picture_and_its_window():
    features = log_mel(np.zeros(int(3 * RATE), np.float32))
    assert features.shape[0] == N_MELS and window(features, 2.5).shape == (N_MELS, FRAMES)
    with pytest.raises(ValueError):
        window(features, 1.0)  # less than 2 s heard


def test_a_sound_lands_where_the_rough_line_says_it_will():
    clean = np.zeros(RATE, np.float32)
    # A 500 Hz tone burst at 0.5 s: the phone's band keeps it (a flat burst would be filtered out).
    clean[int(0.5 * RATE):int(0.6 * RATE)] = 0.5 * np.sin(2 * np.pi * 500 * np.arange(int(0.1 * RATE)) / RATE)
    for speed in (0.9, 1.0, 1.1):
        line, factor = on_a_rough_line(clean, Roughness(speed=speed), None)
        loud = np.flatnonzero(np.abs(line) > 0.05)
        assert loud[0] / RATE == pytest.approx(LEAD_S + 0.5 * factor, abs=0.02)


def _played(answer, chances, switch=None):
    clip = {"id": "x", "answer": answer, "group": answer, "text": "t", "voice": "v"}
    timing = Timing("x", 0.0, 1.0, switch=switch)
    return Played(clip, timing, "phone", [0.3 + 0.1 * i for i in range(len(chances))], chances)


def test_a_call_stops_at_the_first_stop_and_the_line_misses_no_speech():
    played = [_played(KEEP, [0.1, 0.2]), _played(KEEP, [0.1, 0.7]), _played(STOP, [0.2, 0.8]), _played(STOP, [0.6, 0.3])]
    line = zero_miss_line(played)
    assert line == 0.6  # the quietest real speech still stops
    got = mark(played, line)
    assert (got.wrong_stops, got.keep, got.missed_stops) == (1, 2, 0)
    assert got.stop_times == pytest.approx([0.4, 0.3])
    assert separation([_played(KEEP, [0.1]), _played(STOP, [0.9])]) == 1.0


def test_a_mixed_phrase_is_timed_from_the_switch_and_an_early_stop_is_not_a_miss():
    got = mark([_played(MIXED, [0.1, 0.1, 0.9], switch=0.4)], 0.5)
    assert got.missed_stops == 0 and got.stop_times == pytest.approx([0.1])


def test_training_windows_skip_unfair_moments_and_add_background_only(tmp_path):
    talk = np.random.default_rng(0).normal(0, 0.2, int(1.0 * RATE)).astype(np.float32)
    clips = [({"id": "k", "answer": KEEP}, Timing("k", 0.1, 0.5), talk[: int(0.6 * RATE)]),
             ({"id": "m", "answer": MIXED}, Timing("m", 0.1, 0.95, switch=0.5), talk)]

    class Quiet:
        def stretch(self, kind, level, start, count):
            return np.zeros(count, np.float32)

    x, y = training_windows(clips, Quiet(), np.random.default_rng(1))
    assert x.shape[1:] == (N_MELS, FRAMES) and set(y.tolist()) <= {0.0, 1.0}
    assert y.sum() >= 1 and (y == 0).sum() >= 4


def test_the_saved_model_answers_exactly_as_the_trained_one(tmp_path):
    model = HaanRuko(-5.0, 3.0)
    model.eval()
    export_onnx(model, tmp_path / "m.onnx")
    x = np.random.default_rng(0).normal(-5, 3, (2, N_MELS, FRAMES)).astype(np.float32)
    trained = torch.sigmoid(model(torch.from_numpy(x)[:, None])).detach().numpy()
    assert OnnxModel(tmp_path / "m.onnx").chances(x) == pytest.approx(trained, abs=1e-5)


def test_the_real_voice_list_has_every_held_back_phrase_and_about_a_hundred_takes():
    from lab.haan_ruko.human import human_takes

    data = load_phrases()
    takes = human_takes(data)
    held = {t.text for t in takes if t.held_back}
    assert all(p in held for g in (KEEP, STOP, MIXED) for p in data[g]["held_back"])
    assert 90 <= len(takes) <= 110 and len({t.id for t in takes}) == len(takes)
    assert {t.answer for t in takes if t.group == "sentence"} == {STOP}


def test_a_soft_take_is_accepted_and_a_silent_one_refused():
    import io

    import soundfile

    from lab.ears_test.record_server import BadRecording
    from lab.haan_ruko.human import check_take

    def wav(peak):
        out = io.BytesIO()
        samples = peak * np.sin(np.arange(16000) * 0.05 + np.random.default_rng(0).normal(0, 0.3, 16000))
        soundfile.write(out, samples.astype(np.float32), 16000, format="WAV", subtype="PCM_16")
        return out.getvalue()

    check_take(wav(0.03))  # a soft "hmm": the ears test's 0.05 rule would have refused it
    with pytest.raises(BadRecording, match="nothing was heard"):
        check_take(wav(0.002))


def test_real_pieces_are_labelled_from_what_was_said():
    from lab.haan_ruko.indicvoices import answer_for, opens_with_listening

    assert answer_for("हाँ", 0.6) == KEEP and answer_for("जी जी।", 0.8) == KEEP
    assert answer_for("नहीं", 0.5) == STOP and answer_for("मेरा ऑर्डर कहाँ है", 2.0) == STOP
    assert answer_for("हाँ", 0.1) is None  # cut too tight to hear
    assert answer_for("<unintelligible>", 1.0) is None and answer_for("हाँ", 3.0) is None
    assert answer_for("हम", 0.5) == STOP  # "we": not a listening sound
    assert opens_with_listening("हाँ जी मेरा ऑर्डर") and not opens_with_listening("हाँ जी")


def test_real_speakers_split_the_same_way_every_time():
    from lab.haan_ruko.indicvoices import role_of

    roles = [role_of(f"S{n}") for n in range(2000)]
    assert roles == [role_of(f"S{n}") for n in range(2000)]
    counts = Counter(roles)
    assert 200 < counts["test"] < 400 and 100 < counts["check"] < 300  # about 15% and 10%


def test_speech_is_sampled_about_twice_the_listening_short_speech_first():
    from lab.haan_ruko.indicvoices import pick

    rows = [{"answer": KEEP, "duration": 0.5}] * 10 + [{"answer": STOP, "duration": 1.0}] * 50 + [{"answer": STOP, "duration": 4.0}] * 50
    chosen = pick(rows, np.random.default_rng(0))
    speech = [r for r in chosen if r["answer"] == STOP]
    assert sum(r["answer"] == KEEP for r in chosen) == 10 and len(speech) == 20
    assert sum(r["duration"] <= 1.5 for r in speech) == 10


def test_each_clip_gives_at_most_a_few_training_moments():
    talk = np.random.default_rng(0).normal(0, 0.2, int(2.0 * RATE)).astype(np.float32)
    clips = [({"id": "s", "answer": STOP}, Timing("s", 0.1, 1.9), talk)]

    class Quiet:
        def stretch(self, kind, level, start, count):
            return np.zeros(count, np.float32)

    _, y = training_windows(clips, Quiet(), np.random.default_rng(1), checks_per_clip=4)
    assert len(y) == 4  # four of its eighteen checks; 10% of 4 rounds down to no background window


def _talk_clips(count):
    talk = np.random.default_rng(0).normal(0, 0.2, int(1.6 * RATE)).astype(np.float32)
    return [({"id": f"c{n}", "answer": STOP if n % 2 else KEEP}, Timing(f"c{n}", 0.1, 0.4 + 0.1 * (n % 10)), talk)
            for n in range(count)]


def test_helper_processes_make_exactly_the_windows_this_process_would():
    # Each clip has its own seed, so who makes it can't change the round (11 Nov 2026).
    from lab.eot_test.clips import NOISE_FOLDER
    from lab.haan_ruko.sound import Noises

    if not NOISE_FOLDER.exists():
        pytest.skip("the noise recordings are not on this computer (lab/sounds is not in git)")
    clips = _talk_clips(230)  # more than two chunks, so the order of chunks is tested too
    here = training_windows(clips, Noises(NOISE_FOLDER), np.random.default_rng(5))
    with Helpers(NOISE_FOLDER, count=2) as helpers:
        there = training_windows(clips, Noises(NOISE_FOLDER), np.random.default_rng(5), helpers=helpers)
    assert np.array_equal(here[0], there[0]) and np.array_equal(here[1], there[1])
    assert (here[1] == 0).sum() >= int(len(here[1]) / 1.1 * 0.1)  # the background-only windows are there too


def test_asking_in_small_batches_gives_the_same_answers_as_one_big_batch(monkeypatch):
    import lab.haan_ruko.score as score
    from lab.haan_ruko.sound import TEST_LINES

    heard = score.hear(_talk_clips(12), TEST_LINES, None)
    # A made-up model: the answer depends only on the window itself, as a real one does.
    chances_of = lambda windows: windows.mean(axis=(1, 2))  # noqa: E731
    together = score.ask(heard, chances_of)
    monkeypatch.setattr(score, "ASK_AT_ONCE", 7)
    apart = score.ask(heard, chances_of)
    assert [(p.clip["id"], p.line, p.times, p.chances) for p in together] == [(p.clip["id"], p.line, p.times, p.chances) for p in apart]
    assert len(together) == 12 * len(TEST_LINES)


def test_any_mix_of_listening_words_is_listening_but_an_answer_or_a_lone_polite_word_is_speech():
    # The word rule of 14 Nov 2026: listening sounds come in endless combinations.
    from lab.haan_ruko.indicvoices import answer_for, opens_with_listening

    for text in ("ठीक है सर ठीक है", "जी जी बिल्कुल", "ओके ओके ठीक है", "हाँ जी मैम", "हम्म हम्म", "सही है"):
        assert answer_for(text, 1.0) == KEEP, text
    for text in ("हाँ है", "है जी", "सर", "अच्छा है", "नहीं", "हाँ नहीं", "क्या", "हलो"):
        assert answer_for(text, 1.0) == STOP, text
    assert opens_with_listening("ठीक है सर मेरा ऑर्डर कहाँ है") and not opens_with_listening("ठीक है सर")


def test_short_real_no_pieces_are_practised_more_and_nothing_else_is():
    from lab.haan_ruko.indicvoices import is_short_no
    from lab.haan_ruko.train import practise_more

    def real(text, duration=1.0, answer=STOP):
        return ({"text": text, "duration": duration, "answer": answer, "role": "train"}, None, None)

    machine = ({"text": "नहीं", "answer": STOP}, None, None)  # machine clips have no role
    train = [real("नहीं"), real("नहीं नहीं मेरा ऑर्डर अभी तक नहीं आया", duration=2.4), real("हाँ"), machine]
    assert practise_more(train, is_short_no, 4) == 1
    assert len(train) == 7 and sum(item[0]["text"] == "नहीं" and "role" in item[0] for item in train) == 4


def test_only_pieces_that_start_with_a_no_word_count_as_a_short_no():
    # 14 Nov 2026: "any no word" also caught "है ना" ("isn't it?") and "कोई बात नहीं" ("no problem").
    from lab.haan_ruko.indicvoices import is_short_no

    def real(text, duration=1.0, answer=STOP):
        return {"text": text, "duration": duration, "answer": answer, "role": "train"}

    for text in ("नहीं", "नहीं नहीं", "नही नही", "नई", "ना ना", "रुको", "रुकिए", "नहीं सर", "मत करो"):
        assert is_short_no(real(text)), text
    for text in ("है ना", "ठीक है ना", "कोई बात नहीं", "ठीक है कोई बात नहीं", "कोई दिक्कत नहीं है", "कम नहीं होंगे",
                 "वो देखो ना आप", "तो लेना ही है ना आपको"):
        assert not is_short_no(real(text)), text
    assert not is_short_no(real("नहीं", duration=1.6))  # longer than short speech


# --- files 10-81: cut one at a time, raw file deleted (14 Nov 2026) ------------------------


def _fake_raw_file(path, rows):
    """A tiny stand-in for one IndicVoices parquet file: 16 kHz audio bytes and the same columns."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile

    audio = []
    for n, row in enumerate(rows):
        tone = (0.3 * np.sin(2 * np.pi * (200 + 10 * n) * np.arange(int(row["duration"] * 16000)) / 16000)).astype(np.float32)
        buffer = io.BytesIO()
        soundfile.write(buffer, tone, 16000, format="WAV", subtype="PCM_16")
        audio.append({"bytes": buffer.getvalue(), "path": f"{n}.wav"})
    table = pa.table({"audio_filepath": audio, "verbatim": [r["text"] for r in rows], "duration": [r["duration"] for r in rows],
                      "scenario": [r.get("scenario", "Conversation") for r in rows], "speaker_id": [r["speaker"] for r in rows]})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


_ROWS = [{"text": "हाँ जी", "duration": 0.6, "speaker": "S1"}, {"text": "नहीं", "duration": 0.5, "speaker": "S2"},
         {"text": "ओके ओके", "duration": 0.9, "speaker": "S3"}, {"text": "मेरा ऑर्डर कहाँ है", "duration": 1.4, "speaker": "S1"},
         {"text": "भैया मेरा खाना अभी तक नहीं आया और बहुत देर हो गई है", "duration": 3.2, "speaker": "S4"},
         {"text": "हम लोग कल सुबह आएंगे और फिर बात करेंगे", "duration": 2.8, "speaker": "S5"},
         {"text": "<noise> हाँ", "duration": 0.7, "speaker": "S6"}, {"text": "हाँ", "duration": 0.1, "speaker": "S6"},
         {"text": "पढ़ा हुआ वाक्य", "duration": 1.0, "speaker": "S7", "scenario": "Read"}]


def test_sixteen_bit_sound_goes_on_the_line_exactly_as_the_float_read_of_the_same_file(tmp_path):
    import soundfile

    from lab.haan_ruko.sound import as_float

    wav = tmp_path / "a.wav"
    soundfile.write(wav, (0.2 * np.sin(np.arange(8000) / 7)).astype(np.float32), RATE, subtype="PCM_16")
    as_int, _ = soundfile.read(wav, dtype="int16")
    as_flt, _ = soundfile.read(wav, dtype="float32")
    assert as_int.dtype == np.int16 and np.array_equal(as_float(as_int), as_flt)
    line_a, _ = on_a_rough_line(as_int, Roughness(gain_db=-3), None, dither_seed=4)
    line_b, _ = on_a_rough_line(as_flt, Roughness(gain_db=-3), None, dither_seed=4)
    assert np.array_equal(line_a, line_b)


def test_a_newer_file_keeps_every_short_piece_and_as_many_long_ones_as_listening_ones():
    from lab.haan_ruko.indicvoices import store_rows

    rows = [{"duration": d, "answer": a} for d, a in [(0.5, KEEP), (1.9, STOP), (2.0, KEEP), (2.1, STOP), (3.0, STOP), (4.0, STOP)]]
    kept = store_rows(rows, np.random.default_rng(1))
    assert [r["duration"] for r in kept[:3]] == [0.5, 1.9, 2.0]  # all up to 2 s
    assert len(kept) == 5 and all(r["duration"] > 2.0 for r in kept[3:])  # 2 listening, so 2 long
    assert kept == store_rows(rows, np.random.default_rng(1))


def test_a_cut_file_reads_back_as_labelled_sixteen_bit_pieces(tmp_path):
    from common.audio.resample import PHONE_RATE, resample
    from lab.haan_ruko.indicvoices import cut_file, load_more, role_of

    raw = _fake_raw_file(tmp_path / "raw.parquet", _ROWS)
    packs = tmp_path / "store" / "packs"
    packs.mkdir(parents=True)
    stats = cut_file(raw, 10, packs)
    # the read speech, the noise-tagged piece and the too-short piece are not kept; the 4 short
    # ones are, and 2 long ones, as there are 2 listening pieces
    assert stats == {"kept": 6, "keep": 2, "stop": 4, "conversation": 6}
    items = load_more(tmp_path / "store", per_file=100)
    by_text = {c["text"]: (c, t, a) for c, t, a in items}
    assert set(by_text) >= {"हाँ जी", "नहीं", "ओके ओके", "मेरा ऑर्डर कहाँ है"}
    clip, timing, audio = by_text["हाँ जी"]
    assert clip["answer"] == KEEP and clip["role"] == role_of("S1") and audio.dtype == np.int16
    assert by_text["नहीं"][0]["answer"] == STOP and by_text["मेरा ऑर्डर कहाँ है"][0]["answer"] == STOP
    assert abs(timing.end - 0.6) < 0.01 and timing.start == 0.0
    expected = resample(0.3 * np.sin(2 * np.pi * 200 * np.arange(int(0.6 * 16000)) / 16000), 16000, PHONE_RATE)
    assert np.max(np.abs(audio / 32768 - expected[: len(audio)])) < 2e-3  # the same sound, as 16-bit


def test_deleting_a_download_removes_that_raw_file_and_its_note_and_nothing_else(tmp_path):
    from lab.haan_ruko.indicvoices import file_name, forget_download

    (tmp_path / ".cache" / "huggingface" / "download" / "hindi").mkdir(parents=True)
    (tmp_path / "hindi").mkdir()
    for name in (file_name(10), file_name(11)):
        (tmp_path / name).write_bytes(b"raw")
        (tmp_path / ".cache" / "huggingface" / "download" / f"{name}.metadata").write_text("note")
    (tmp_path / "keep_me.txt").write_text("not ours")
    gone = forget_download(tmp_path, file_name(10))
    assert len(gone) == 2 and not (tmp_path / file_name(10)).exists()
    assert (tmp_path / file_name(11)).exists() and (tmp_path / "keep_me.txt").exists()
    assert (tmp_path / ".cache" / "huggingface" / "download" / f"{file_name(11)}.metadata").exists()


def test_importing_cuts_each_file_deletes_its_raw_file_and_skips_finished_files(tmp_path):
    import shutil

    from lab.haan_ruko.indicvoices import file_name, forget_download, import_more

    source = _fake_raw_file(tmp_path / "source.parquet", _ROWS)
    downloads, asked = tmp_path / "downloading", []

    def download(name):
        asked.append(name)
        (downloads / name).parent.mkdir(parents=True, exist_ok=True)
        return Path(shutil.copy(source, downloads / name))

    def forget(name):
        return forget_download(downloads, name)

    done = import_more(10, 13, download, forget, store=tmp_path / "store", ahead=1, log=lambda line: None)
    assert [d["file"] for d in done] == [10, 11, 12] and asked == [file_name(n) for n in (10, 11, 12)]
    assert not list(downloads.rglob("*.parquet"))  # every raw file deleted
    assert sorted(p.name for p in (tmp_path / "store" / "packs").iterdir()) == [
        "iv-10.flac", "iv-10.json", "iv-11.flac", "iv-11.json", "iv-12.flac", "iv-12.json"]
    asked.clear()
    assert import_more(10, 14, download, forget, store=tmp_path / "store", log=lambda line: None)[0]["file"] == 13
    assert asked == [file_name(13)]  # finished files are not downloaded again


def test_a_newer_file_gives_every_weak_spot_piece_then_listening_and_twice_the_speech():
    from lab.haan_ruko.indicvoices import training_share

    clips = ([{"answer": KEEP, "text": "ओके सर", "duration": 0.8, "role": "train"}] * 3
             + [{"answer": STOP, "text": "नहीं", "duration": 0.5, "role": "train"}] * 2
             + [{"answer": KEEP, "text": "हाँ", "duration": 0.4, "role": "train"}] * 50
             + [{"answer": STOP, "text": "कब आएगा", "duration": 1.0, "role": "train"}] * 50
             + [{"answer": STOP, "text": "बहुत लंबी बात", "duration": 2.4, "role": "train"}] * 50)
    chosen = training_share(clips, np.random.default_rng(0), per_file=35)
    assert {0, 1, 2, 3, 4} <= set(chosen) and len(chosen) == 35
    picked = Counter((clips[i]["answer"], clips[i]["duration"]) for i in chosen if i > 4)
    assert picked[(KEEP, 0.4)] == 10 and picked[(STOP, 1.0)] == 10 and picked[(STOP, 2.4)] == 10
    assert chosen == training_share(clips, np.random.default_rng(0), per_file=35)
    assert len(training_share(clips[:5], np.random.default_rng(0), per_file=2)) == 5  # weak spots always all


def test_each_round_trains_on_all_clips_when_they_fit_else_a_fixed_size_share():
    from lab.haan_ruko.train import round_share

    small = list(range(10))
    assert round_share(small, np.random.default_rng(0), count=10) is small
    share = round_share(list(range(100)), np.random.default_rng(0), count=30)
    assert len(share) == 30 == len(set(share)) and share == sorted(share)
    assert share == round_share(list(range(100)), np.random.default_rng(0), count=30)
