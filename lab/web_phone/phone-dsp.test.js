// Tests for the web phone's logic, run by Node: node --test lab/web_phone/phone-dsp.test.js
// (the Python suite runs this too, in tests/lab/test_web_phone.py).

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  Downsampler,
  PhoneCore,
  PieceMaker,
  ProtocolError,
  Speaker,
  decodeRobotMessage,
  encodeCleared,
  encodeEnd,
  encodeKey,
  encodePlayed,
  encodeStart,
  floatToPcm16,
  pcm16ToFloat,
} from "./phone-dsp.js";

const sine = (freq, seconds, rate, level = 0.5) =>
  Float32Array.from({ length: Math.round(seconds * rate) }, (_, i) => level * Math.sin((2 * Math.PI * freq * i) / rate));

const rms = (samples) => Math.sqrt(samples.reduce((sum, x) => sum + x * x, 0) / samples.length);

const signChanges = (samples) => {
  let count = 0;
  for (let i = 1; i < samples.length; i++) if (Math.sign(samples[i]) !== Math.sign(samples[i - 1])) count++;
  return count;
};

const piece = (value) => floatToPcm16(new Float32Array(160).fill(value));

// ---- messages -----------------------------------------------------------------

test("messages are byte-for-byte what the Python side writes", () => {
  assert.equal(encodeStart("c1"), '{"type":"start","call_id":"c1","sample_rate":8000,"piece_ms":20}');
  assert.equal(encodePlayed("q1", 40), '{"type":"played","name":"q1","caller_ms":40}');
  assert.equal(encodeCleared(3, 60, ["q1"]), '{"type":"cleared","played_pieces":3,"caller_ms":60,"dropped_marks":["q1"]}');
  assert.equal(encodeKey("#", 20), '{"type":"key","digit":"#","caller_ms":20}');
  assert.equal(encodeEnd("bye"), '{"type":"end","reason":"bye"}');
});

test("bad caller times, keys and empty fields are refused", () => {
  assert.throws(() => encodePlayed("q1", 30), ProtocolError);
  assert.throws(() => encodePlayed("q1", -20), ProtocolError);
  assert.throws(() => encodeKey("A", 0), ProtocolError);
  assert.throws(() => encodeKey("12", 0), ProtocolError);
  assert.throws(() => encodeStart(""), ProtocolError);
  assert.throws(() => encodeEnd(""), ProtocolError);
});

test("only mark, clear and end are accepted from a robot", () => {
  assert.deepEqual(decodeRobotMessage('{"type":"mark","name":"q1"}'), { type: "mark", name: "q1" });
  assert.deepEqual(decodeRobotMessage('{"type":"clear"}'), { type: "clear" });
  assert.deepEqual(decodeRobotMessage('{"type":"end","reason":"bye"}'), { type: "end", reason: "bye" });
  for (const bad of [
    "not json",
    "[1]",
    '{"type":"start","call_id":"c","sample_rate":8000,"piece_ms":20}',
    '{"type":"mark","name":""}',
    '{"type":"mark","name":"q","extra":1}',
    '{"type":"end"}',
  ]) {
    assert.throws(() => decodeRobotMessage(bad), ProtocolError, bad);
  }
});

// ---- samples ------------------------------------------------------------------

test("PCM is 16-bit little-endian and clips instead of wrapping", () => {
  const bytes = new Uint8Array(floatToPcm16(Float32Array.from([0.5, 1.5, -1.5])));
  assert.deepEqual([...bytes], [0x00, 0x40, 0xff, 0x7f, 0x00, 0x80]);
  const back = pcm16ToFloat(floatToPcm16(Float32Array.from([0.25, -0.75])));
  assert.deepEqual([...back], [0.25, -0.75]);
  assert.throws(() => pcm16ToFloat(new ArrayBuffer(3)), ProtocolError);
});

// ---- microphone ---------------------------------------------------------------

for (const rate of [48000, 44100]) {
  test(`a ${rate} Hz microphone becomes 8 kHz with the same pitch`, () => {
    const out = new Downsampler(rate).process(sine(1000, 1, rate));
    assert.ok(Math.abs(out.length - 8000) <= 1, `length ${out.length}`);
    // 1000 Hz = 2000 sign changes per second.
    assert.ok(Math.abs(signChanges(out.slice(100)) - 2000 * (out.length - 100) / 8000) <= 4);
  });
}

test("sound above 4 kHz is removed before it can fold back as noise", () => {
  const out = new Downsampler(48000).process(sine(6000, 1, 48000));
  assert.ok(rms(out.slice(100)) < 0.01, `rms ${rms(out)}`); // the tone itself had rms 0.35
});

test("the downsampler gives the same result in small blocks as in one", () => {
  const input = sine(700, 0.2, 48000);
  const whole = new Downsampler(48000).process(input);
  const blocks = new Downsampler(48000);
  const parts = [];
  for (let i = 0; i < input.length; i += 128) parts.push(...blocks.process(input.slice(i, i + 128)));
  assert.equal(parts.length, whole.length);
  parts.forEach((x, i) => assert.ok(Math.abs(x - whole[i]) < 1e-6));
});

test("samples become whole 20 ms pieces; the rest waits for the next push", () => {
  const maker = new PieceMaker();
  assert.equal(maker.push(new Float32Array(350)).length, 2);
  assert.equal(maker.push(new Float32Array(129)).length, 0);
  assert.equal(maker.push(new Float32Array(1)).length, 1);
});

// ---- speaker --------------------------------------------------------------------

test("the speaker plays pieces in order and reaches a mark when its audio ends", () => {
  const speaker = new Speaker();
  speaker.addPiece(new Float32Array(160).fill(0.1));
  speaker.addMark("q1");
  let reached = [];
  for (let i = 0; i < 160; i++) {
    const sample = speaker.nextSample();
    assert.equal(sample.value, Math.fround(0.1));
    reached = sample.reached;
  }
  assert.deepEqual(reached, ["q1"]); // with the last sample
  assert.deepEqual(speaker.nextSample(), { value: 0, reached: [] });
  assert.equal(speaker.startedPieces, 1);
});

test("clear drops unplayed audio and its marks", () => {
  const speaker = new Speaker();
  for (let i = 0; i < 5; i++) speaker.addPiece(new Float32Array(160).fill(0.2));
  speaker.addMark("q1");
  for (let i = 0; i < 200; i++) speaker.nextSample(); // into the second piece
  assert.deepEqual(speaker.clear(), { startedPieces: 2, dropped: ["q1"] });
  assert.equal(speaker.isPlaying, false);
  assert.equal(speaker.nextSample().value, 0);
});

// ---- the whole phone ----------------------------------------------------------

function run(core, seconds, mic = null) {
  const out = { pieces: [], played: [], audio: [] };
  const quanta = Math.round((seconds * core.contextRate) / 128);
  for (let q = 0; q < quanta; q++) {
    const output = new Float32Array(128);
    const input = mic ? mic.subarray(q * 128, q * 128 + 128) : null;
    const result = core.process(input, output);
    out.pieces.push(...result.pieces);
    out.played.push(...result.played);
    out.audio.push(...output);
  }
  return out;
}

test("before the call starts the phone sends nothing and plays silence", () => {
  const core = new PhoneCore(48000);
  const out = run(core, 0.1);
  assert.equal(out.pieces.length, 0);
  assert.ok(out.audio.every((x) => x === 0));
});

test("the phone sends one piece per 20 ms, even with no microphone", () => {
  const core = new PhoneCore(48000);
  core.start();
  assert.equal(run(core, 1).pieces.length, 50);
  const withMic = new PhoneCore(48000);
  withMic.start();
  const out = run(withMic, 1, sine(500, 1, 48000));
  assert.equal(out.pieces.length, 50);
  assert.ok(out.pieces.every((buffer) => buffer.byteLength === 320));
});

for (const rate of [48000, 44100]) {
  test(`at ${rate} Hz a mark comes back exactly when its audio has played`, () => {
    const core = new PhoneCore(rate);
    core.start();
    for (let i = 0; i < 25; i++) core.addAudio(piece(0.25)); // half a second
    core.addMark("q1");
    const out = run(core, 0.7);
    assert.deepEqual(out.played, [{ name: "q1", caller_ms: 500 }]);
    const playing = out.audio.slice(Math.round(0.05 * rate), Math.round(0.45 * rate));
    assert.ok(playing.every((x) => Math.abs(x - 0.25) < 1e-6)); // the robot's sound, steady
    assert.ok(out.audio.slice(Math.round(0.55 * rate)).every((x) => x === 0)); // then silence
  });
}

test("a mark waits for the very end of its piece, even when the piece starts mid-window", () => {
  // The robot's piece arrives ~10 ms into the call, so it ends near sample
  // 247, in the second 20 ms window: the mark must say 40 ms, not 20.
  const core = new PhoneCore(48000);
  core.start();
  run(core, 512 / 48000); // 4 render quanta of silence
  core.addAudio(piece(0.25));
  core.addMark("q1");
  assert.deepEqual(run(core, 0.1).played, [{ name: "q1", caller_ms: 40 }]);
});

test("clear stops the robot at once and reports what was heard", () => {
  const core = new PhoneCore(48000);
  core.start();
  for (let i = 0; i < 100; i++) core.addAudio(piece(0.25)); // 2 seconds
  core.addMark("q1");
  run(core, 0.2);
  const cleared = core.clear();
  assert.ok(cleared.played_pieces >= 10 && cleared.played_pieces <= 11, JSON.stringify(cleared));
  assert.deepEqual(cleared.dropped_marks, ["q1"]);
  assert.equal(cleared.caller_ms % 20, 0);
  const after = run(core, 0.2);
  assert.ok(after.audio.slice(10).every((x) => x === 0)); // silence right after (one sample of glide)
  assert.deepEqual(after.played, []);
});

test("caller times round up to the next piece boundary", () => {
  const core = new PhoneCore(48000);
  assert.equal(core.callerMs(0), 0);
  assert.equal(core.callerMs(160), 20);
  assert.equal(core.callerMs(161), 40);
  assert.throws(() => core.addAudio(new ArrayBuffer(100)), ProtocolError);
});
