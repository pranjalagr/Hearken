// The web phone's logic: the wire's messages, sound conversion, and the
// phone itself (microphone in, speaker out), written without any browser
// parts so Node can test it.
//
// In the browser this same file is also the AudioWorklet: it registers the
// processor at the bottom when it runs inside an audio worklet. One file
// means the worklet never depends on module imports inside worklets.
//
// Everything here must match common/wire/protocol.py exactly: 8 kHz,
// 20 ms pieces of 160 samples, 16-bit little-endian PCM, and the same JSON.

export const SAMPLE_RATE = 8000;
export const PIECE_MS = 20;
export const PIECE_SAMPLES = 160;
export const PIECE_BYTES = 320;
const KEYPAD = new Set("0123456789*#");

export class ProtocolError extends Error {}

// ---- the wire's messages (caller side) ---------------------------------------

function checkCallerMs(ms) {
  // Every caller time is a piece boundary: 20 ms x pieces sent.
  if (!Number.isInteger(ms) || ms < 0 || ms % PIECE_MS !== 0) {
    throw new ProtocolError(`caller_ms must be a non-negative multiple of ${PIECE_MS}`);
  }
}

export function encodeStart(callId) {
  if (!callId) throw new ProtocolError("start needs a call id");
  return JSON.stringify({ type: "start", call_id: callId, sample_rate: SAMPLE_RATE, piece_ms: PIECE_MS });
}

export function encodePlayed(name, callerMs) {
  checkCallerMs(callerMs);
  return JSON.stringify({ type: "played", name, caller_ms: callerMs });
}

export function encodeCleared(playedPieces, callerMs, droppedMarks) {
  checkCallerMs(callerMs);
  return JSON.stringify({
    type: "cleared",
    played_pieces: playedPieces,
    caller_ms: callerMs,
    dropped_marks: droppedMarks,
  });
}

export function encodeKey(digit, callerMs) {
  if (!KEYPAD.has(digit) || digit.length !== 1) throw new ProtocolError(`not a keypad key: ${digit}`);
  checkCallerMs(callerMs);
  return JSON.stringify({ type: "key", digit, caller_ms: callerMs });
}

export function encodeEnd(reason) {
  if (!reason) throw new ProtocolError("end needs a reason");
  return JSON.stringify({ type: "end", reason });
}

// A robot may send only mark, clear and end; anything else breaks the wire's
// rules, so it is refused rather than half-understood.
export function decodeRobotMessage(text) {
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    throw new ProtocolError("not JSON");
  }
  if (body === null || typeof body !== "object" || Array.isArray(body)) {
    throw new ProtocolError("a message must be a JSON object");
  }
  const keys = Object.keys(body).sort().join(",");
  if (body.type === "mark" && keys === "name,type" && typeof body.name === "string" && body.name) {
    return { type: "mark", name: body.name };
  }
  if (body.type === "clear" && keys === "type") return { type: "clear" };
  if (body.type === "end" && keys === "reason,type" && typeof body.reason === "string" && body.reason) {
    return { type: "end", reason: body.reason };
  }
  throw new ProtocolError(`a robot may not send this: ${text.slice(0, 80)}`);
}

// ---- samples <-> bytes ------------------------------------------------------------

// One scale both ways, as in common/audio/pcm.py.
export function floatToPcm16(samples) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32768)));
    view.setInt16(i * 2, value, true); // little-endian, whatever the machine is
  }
  return buffer;
}

export function pcm16ToFloat(buffer) {
  if (buffer.byteLength % 2) throw new ProtocolError("16-bit PCM needs an even number of bytes");
  const view = new DataView(buffer);
  const out = new Float32Array(buffer.byteLength / 2);
  for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}

// ---- microphone: any rate -> 8 kHz ----------------------------------------------

// A windowed-sinc low-pass filter (Blackman window), normalised to gain 1.
export function lowPassTaps(rate, cutoffHz, count = 63) {
  const taps = new Float64Array(count);
  const mid = (count - 1) / 2;
  const fc = cutoffHz / rate;
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const x = i - mid;
    const sinc = x === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x);
    const window = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (count - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (count - 1));
    taps[i] = sinc * window;
    sum += taps[i];
  }
  for (let i = 0; i < count; i++) taps[i] /= sum;
  return taps;
}

// Filter first, then pick 8 kHz samples. Without the filter, sound above
// 4 kHz (a hiss, a whistle) would fold back into the call as noise.
export class Downsampler {
  constructor(inRate, outRate = SAMPLE_RATE) {
    this.step = inRate / outRate;
    this.taps = lowPassTaps(inRate, 3600); // just under the 4 kHz limit of 8 kHz sound
    this.history = new Float64Array(this.taps.length * 2); // doubled: no wrap-around maths per tap
    this.pos = 0;
    this.index = 0; // input samples seen so far
    this.nextT = 0; // where the next output sample falls, in input samples
    this.prevY = 0;
  }

  process(input) {
    const out = [];
    const n = this.taps.length;
    for (let i = 0; i < input.length; i++) {
      this.history[this.pos] = input[i];
      this.history[this.pos + n] = input[i];
      this.pos = (this.pos + 1) % n;
      let y = 0;
      for (let k = 0; k < n; k++) y += this.taps[k] * this.history[this.pos + k];
      while (this.nextT <= this.index) {
        const frac = this.nextT - (this.index - 1);
        out.push(this.prevY + (y - this.prevY) * frac);
        this.nextT += this.step;
      }
      this.prevY = y;
      this.index += 1;
    }
    return Float32Array.from(out);
  }
}

// Collects 8 kHz samples into whole 20 ms pieces, as bytes for the wire.
export class PieceMaker {
  constructor() {
    this.buffer = new Float32Array(PIECE_SAMPLES);
    this.filled = 0;
  }

  push(samples) {
    const pieces = [];
    for (let i = 0; i < samples.length; i++) {
      this.buffer[this.filled++] = samples[i];
      if (this.filled === PIECE_SAMPLES) {
        pieces.push(floatToPcm16(this.buffer));
        this.filled = 0;
      }
    }
    return pieces;
  }
}

// ---- speaker: the robot's audio, played at real speed ---------------------------

// The same rules as common/wire/playback.py: a mark is reached once every
// piece queued before it has played; clear throws away what is unplayed.
export class Speaker {
  constructor() {
    this.queue = [];
    this.current = null;
    this.offset = 0;
    this.startedPieces = 0; // robot pieces the caller has started to hear
  }

  get isPlaying() {
    return this.current !== null || this.queue.some((item) => item instanceof Float32Array);
  }

  addPiece(samples) {
    if (samples.length !== PIECE_SAMPLES) throw new ProtocolError("a robot piece must be 160 samples");
    this.queue.push(samples);
  }

  addMark(name) {
    this.queue.push({ mark: name });
  }

  // The next 8 kHz sample, and the marks reached when it ends.
  nextSample() {
    // Marks at the head are reached now only if no piece is still playing:
    // a mark queued right after the current piece waits for its last sample.
    const reached = this.current === null ? this.popMarks() : [];
    if (this.current === null && this.queue.length && this.queue[0] instanceof Float32Array) {
      this.current = this.queue.shift();
      this.offset = 0;
      this.startedPieces += 1;
    }
    let value = 0;
    if (this.current !== null) {
      value = this.current[this.offset++];
      if (this.offset === PIECE_SAMPLES) {
        this.current = null;
        reached.push(...this.popMarks());
      }
    }
    return { value, reached };
  }

  clear() {
    const dropped = this.queue.filter((item) => !(item instanceof Float32Array)).map((item) => item.mark);
    this.queue = [];
    this.current = null;
    return { startedPieces: this.startedPieces, dropped };
  }

  popMarks() {
    const names = [];
    while (this.queue.length && !(this.queue[0] instanceof Float32Array)) names.push(this.queue.shift().mark);
    return names;
  }
}

// ---- the phone: one audio worklet's work, without the browser -------------------

export class PhoneCore {
  constructor(contextRate) {
    this.contextRate = contextRate;
    this.ratio = SAMPLE_RATE / contextRate; // 8 kHz samples per output frame
    this.reset();
  }

  reset() {
    this.capturing = false;
    this.down = new Downsampler(this.contextRate);
    this.maker = new PieceMaker();
    this.speaker = new Speaker();
    this.frames = 0; // output frames since the call started
    this.pulled = 0; // 8 kHz speaker samples pulled so far
    // The two 8 kHz samples around the current output frame: a at index
    // `base`, b at base + 1. Output frames are interpolated between them.
    this.base = 0;
    this.a = 0;
    this.b = 0;
    this.primed = false;
  }

  // The call starts: the caller's clock starts at 0 here.
  start() {
    this.reset();
    this.capturing = true;
  }

  stop() {
    this.reset();
  }

  // Now, on the caller's clock, rounded UP to a piece boundary. Up, not down:
  // a "played" time that is a little late can only make an early "haan"
  // count less, never more (Part 2 decision 22).
  callerMs(position8k = this.frames * this.ratio) {
    // The tiny subtraction absorbs float error (4000 computed as
    // 4000.0000001 must not round up a whole extra piece); max(0, ...) keeps
    // 0 from becoming -0.
    return Math.max(0, Math.ceil(position8k / PIECE_SAMPLES - 1e-9)) * PIECE_MS;
  }

  addAudio(buffer) {
    if (buffer.byteLength !== PIECE_BYTES) throw new ProtocolError(`a robot piece must be ${PIECE_BYTES} bytes`);
    this.speaker.addPiece(pcm16ToFloat(buffer));
  }

  addMark(name) {
    this.speaker.addMark(name);
  }

  clear() {
    const { startedPieces, dropped } = this.speaker.clear();
    return { played_pieces: startedPieces, caller_ms: this.callerMs(), dropped_marks: dropped };
  }

  // One render quantum: the microphone frames in, the speaker frames out.
  process(input, output) {
    const result = { pieces: [], played: [] };
    if (!this.capturing) {
      output.fill(0);
      return result;
    }
    // A missing microphone still sends silence: the caller's clock (pieces
    // sent x 20 ms) must keep running, as on a real line.
    const mic = input && input.length === output.length ? input : new Float32Array(output.length);
    result.pieces = this.maker.push(this.down.process(mic));
    if (!this.primed) {
      this.a = this.pull(result);
      this.b = this.pull(result);
      this.primed = true;
    }
    for (let k = 0; k < output.length; k++) {
      const t = (this.frames + k) * this.ratio; // where this frame falls, in 8 kHz samples
      const i0 = Math.floor(t);
      while (this.base < i0) {
        this.a = this.b;
        this.b = this.pull(result);
        this.base += 1;
      }
      output[k] = this.a + (this.b - this.a) * (t - i0);
    }
    this.frames += output.length;
    return result;
  }

  pull(result) {
    const index = this.pulled++;
    const { value, reached } = this.speaker.nextSample();
    for (const name of reached) result.played.push({ name, caller_ms: this.callerMs(index + 1) });
    return value;
  }
}

// ---- in the browser: this file is also the audio worklet -------------------------

if (typeof AudioWorkletProcessor !== "undefined" && typeof registerProcessor === "function") {
  class PhoneProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      // `sampleRate` is the audio worklet's global: the context's own rate.
      // eslint-disable-next-line no-undef
      this.core = new PhoneCore(sampleRate);
      this.port.onmessage = (event) => {
        const message = event.data;
        try {
          if (message.type === "start") this.core.start();
          else if (message.type === "stop") this.core.stop();
          else if (message.type === "audio") this.core.addAudio(message.buffer);
          else if (message.type === "mark") this.core.addMark(message.name);
          else if (message.type === "clear") this.port.postMessage({ type: "cleared", ...this.core.clear() });
        } catch (error) {
          this.port.postMessage({ type: "error", message: String(error) });
        }
      };
    }

    process(inputs, outputs) {
      const input = inputs[0] && inputs[0][0];
      const output = outputs[0][0];
      const result = this.core.process(input, output);
      for (const buffer of result.pieces) this.port.postMessage({ type: "piece", buffer }, [buffer]);
      for (const played of result.played) this.port.postMessage({ type: "played", ...played });
      return true;
    }
  }
  // eslint-disable-next-line no-undef
  registerProcessor("hearken-phone", PhoneProcessor);
}
