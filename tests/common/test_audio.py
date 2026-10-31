"""Audio helpers: PCM bytes, pieces, resampling, phone line, noise, files."""

import numpy as np
import pytest

from common.audio.noise import NOISE_LEVELS_DB, SPEECH_LEVEL_DBFS, NoiseBed, active_rms, set_speech_level
from common.audio.pcm import float_to_pcm16, pcm16_to_float, split_pieces
from common.audio.phone_line import band_limit, mu_law_round_trip
from common.audio.resample import PHONE_RATE, resample
from common.audio.sound_files import read_sound, write_sound


def tone(freq: float, seconds: float, rate: int, level: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def peak_hz(samples: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.fft.rfftfreq(len(samples), 1 / rate)[np.argmax(spectrum)])


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))


def db(ratio: float) -> float:
    return 20 * np.log10(ratio)


# ---- PCM and pieces -------------------------------------------------------------


def test_pcm_round_trip_is_exact_for_values_that_came_from_pcm():
    original = np.arange(-32768, 32768, 7, dtype=np.int16)
    data = original.astype("<i2").tobytes()
    assert float_to_pcm16(pcm16_to_float(data)) == data


def test_pcm_is_16_bit_little_endian():
    assert float_to_pcm16(np.array([0.5], dtype=np.float32)) == (16384).to_bytes(2, "little", signed=True)
    assert float_to_pcm16(np.array([-1.0], dtype=np.float32)) == (-32768).to_bytes(2, "little", signed=True)


def test_pcm_clips_instead_of_wrapping_around():
    data = float_to_pcm16(np.array([1.5, -1.5, 1.0], dtype=np.float32))
    assert list(np.frombuffer(data, dtype="<i2")) == [32767, -32768, 32767]


def test_odd_pcm_byte_count_is_refused():
    with pytest.raises(ValueError):
        pcm16_to_float(b"\x00\x01\x02")


def test_split_pieces_pads_only_the_last_piece_with_silence():
    pieces = split_pieces(np.ones(350, dtype=np.float32), 160)
    assert [len(piece) for piece in pieces] == [160, 160, 160]
    assert pieces[2][:30].tolist() == [1.0] * 30 and not pieces[2][30:].any()


def test_split_pieces_exact_multiple_and_empty():
    assert len(split_pieces(np.zeros(320, dtype=np.float32), 160)) == 2
    assert split_pieces(np.zeros(0, dtype=np.float32), 160) == []
    with pytest.raises(ValueError):
        split_pieces(np.zeros(10, dtype=np.float32), 0)


# ---- resampling ----------------------------------------------------------------


@pytest.mark.parametrize("rate", [16000, 22050, 24000, 44100, 48000])
def test_resampling_to_8k_keeps_the_pitch_and_the_length(rate):
    out = resample(tone(1000, 1.0, rate), rate, PHONE_RATE)
    assert abs(len(out) - PHONE_RATE) <= 1
    assert peak_hz(out, PHONE_RATE) == pytest.approx(1000, abs=2)
    assert out.dtype == np.float32


def test_sound_above_4khz_is_removed_not_folded_back():
    """Without the filter, a 6 kHz whistle at 48 kHz would come back as 2 kHz."""
    out = resample(tone(6000, 1.0, 48000), 48000, PHONE_RATE)
    assert rms(out[400:-400]) < 0.005  # the tone had RMS 0.35


def test_same_rate_returns_a_copy():
    samples = tone(500, 0.1, 8000)
    out = resample(samples, 8000, 8000)
    assert np.array_equal(out, samples) and out is not samples


# ---- phone line ----------------------------------------------------------------


def test_phone_band_keeps_speech_and_removes_rumble_and_hiss():
    passed = band_limit(tone(1000, 1.0, PHONE_RATE))
    rumble = band_limit(tone(80, 1.0, PHONE_RATE))
    edge = band_limit(tone(3900, 1.0, PHONE_RATE))
    middle = slice(800, -800)
    assert db(rms(passed[middle]) / rms(tone(1000, 1.0, PHONE_RATE))) == pytest.approx(0, abs=0.5)
    assert db(rms(rumble[middle]) / 0.3536) < -20
    assert db(rms(edge[middle]) / 0.3536) < -10


@pytest.mark.parametrize("length", [0, 1, 2, 5, 27, 28, 100])
def test_phone_band_works_on_short_blocks(length):
    assert len(band_limit(np.ones(length, dtype=np.float32))) == length


def test_mu_law_keeps_speech_level_sound_close_and_quiet_stays_quiet():
    signal = tone(440, 1.0, PHONE_RATE, level=0.3)
    out = mu_law_round_trip(signal)
    error = out - signal
    assert db(rms(signal) / rms(error)) > 30  # G.711 gives about 38 dB on speech
    assert mu_law_round_trip(np.zeros(5, dtype=np.float32)).tolist() == [0.0] * 5
    # A tiny sample may round to exactly 0, as in a real 8-bit codec, but a
    # sample never comes back with the opposite sign.
    assert np.all(np.sign(out) * np.sign(signal) >= 0)


def test_mu_law_has_at_most_256_output_values_like_an_8_bit_codec():
    ramp = np.linspace(-1, 1, 100_000, dtype=np.float32)
    assert len(np.unique(mu_law_round_trip(ramp))) <= 256


def test_mu_law_works_sample_by_sample_so_pieces_change_nothing():
    signal = tone(300, 0.2, PHONE_RATE)
    whole = mu_law_round_trip(signal)
    pieced = np.concatenate([mu_law_round_trip(p) for p in np.split(signal, 10)])
    assert np.array_equal(whole, pieced)


# ---- levels and noise ----------------------------------------------------------


def test_active_level_ignores_the_pauses_between_words():
    speech = tone(500, 0.5, PHONE_RATE, level=0.2)
    with_pauses = np.concatenate([speech, np.zeros(PHONE_RATE * 2, dtype=np.float32), speech])
    assert active_rms(with_pauses, PHONE_RATE) == pytest.approx(rms(speech), rel=0.01)


def test_set_speech_level_hits_the_target():
    speech = np.concatenate([tone(500, 0.5, PHONE_RATE, 0.05), np.zeros(4000, dtype=np.float32)])
    out = set_speech_level(speech, PHONE_RATE)
    assert db(active_rms(out, PHONE_RATE)) == pytest.approx(SPEECH_LEVEL_DBFS, abs=0.1)


def test_set_speech_level_leaves_silence_alone():
    assert not set_speech_level(np.zeros(800, dtype=np.float32), PHONE_RATE).any()


@pytest.mark.parametrize("level", sorted(NOISE_LEVELS_DB))
def test_noise_bed_sits_the_right_distance_below_speech(level):
    rng = np.random.default_rng(1)
    bed = NoiseBed(rng.normal(0, 0.3, PHONE_RATE * 3).astype(np.float32), NOISE_LEVELS_DB[level])
    noise = bed.next(PHONE_RATE * 3)  # exactly one loop
    assert db(rms(noise)) == pytest.approx(SPEECH_LEVEL_DBFS - NOISE_LEVELS_DB[level], abs=0.01)


def test_noise_bed_loops_seamlessly_from_its_start_point():
    recording = np.arange(1000, dtype=np.float32) + 1
    bed = NoiseBed(recording, speech_to_noise_db=10, start_sample=900)
    pieces = np.concatenate([bed.next(160) for _ in range(10)])  # 1600 samples, wraps twice
    expected = np.concatenate([recording[900:], recording, recording[:500]])
    assert np.allclose(pieces / pieces[0], expected / expected[0])


def test_same_start_point_gives_the_same_noise():
    recording = np.random.default_rng(2).normal(0, 0.2, 5000).astype(np.float32)
    first = NoiseBed(recording, 10, start_sample=1234).next(3000)
    second = NoiseBed(recording, 10, start_sample=1234).next(3000)
    assert np.array_equal(first, second)


def test_noise_bed_refuses_empty_or_silent_recordings():
    with pytest.raises(ValueError):
        NoiseBed(np.zeros(0, dtype=np.float32), 10)
    with pytest.raises(ValueError):
        NoiseBed(np.zeros(100, dtype=np.float32), 10)


# ---- sound files ---------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".wav", ".flac"])
def test_sound_files_round_trip(tmp_path, suffix):
    samples = tone(700, 0.3, 16000, level=0.6)
    path = tmp_path / f"clip{suffix}"
    write_sound(path, samples, 16000)
    back, rate = read_sound(path)
    assert rate == 16000 and back.dtype == np.float32
    assert np.max(np.abs(back - samples)) < 1 / 16000  # 16-bit rounding only


def test_stereo_is_averaged_to_mono(tmp_path):
    import soundfile

    left, right = np.full(100, 0.5, dtype=np.float32), np.full(100, -0.25, dtype=np.float32)
    path = tmp_path / "stereo.wav"
    soundfile.write(str(path), np.stack([left, right], axis=1), 8000, subtype="PCM_16")
    mono, _ = read_sound(path)
    assert np.allclose(mono, 0.125, atol=1 / 16000)
