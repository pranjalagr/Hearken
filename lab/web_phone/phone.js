// The web phone's browser side: buttons, microphone permission, the
// WebSocket, and passing messages between the line and the audio worklet.
// All sound and protocol logic lives in phone-dsp.js (tested with Node).

import {
  PIECE_BYTES,
  PIECE_MS,
  decodeRobotMessage,
  encodeCleared,
  encodeEnd,
  encodeKey,
  encodePlayed,
  encodeStart,
} from "./phone-dsp.js";

const $ = (id) => document.getElementById(id);
const ui = {
  url: $("robot-url"),
  call: $("call"),
  hangUp: $("hang-up"),
  silent: $("go-silent"),
  status: $("status"),
  keys: document.querySelectorAll("[data-key]"),
};

function setStatus(text) {
  ui.status.textContent = text;
}

function setInCall(inCall) {
  ui.call.disabled = inCall;
  ui.url.disabled = inCall;
  ui.hangUp.disabled = !inCall;
  ui.silent.disabled = !inCall;
  ui.keys.forEach((key) => (key.disabled = !inCall));
}

class Call {
  constructor(url) {
    this.url = url;
    this.callId = `web-${crypto.randomUUID()}`;
    this.piecesSent = 0; // the caller's clock: 20 ms x pieces sent
    this.silent = false;
    this.closed = false;
  }

  async open() {
    // The device's own rate; the worklet turns it into the wire's 8 kHz.
    this.audio = new AudioContext();
    await this.audio.audioWorklet.addModule("phone-dsp.js");
    // Echo cancellation keeps the robot's own voice out of the microphone
    // (decision 22: a "haan" heard as an echo must never count).
    this.mic = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.node = new AudioWorkletNode(this.audio, "hearken-phone", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    this.node.port.onmessage = (event) => this.fromPhone(event.data);
    this.audio.createMediaStreamSource(this.mic).connect(this.node);
    this.node.connect(this.audio.destination);

    this.ws = new WebSocket(this.url);
    this.ws.binaryType = "arraybuffer";
    this.ws.onopen = () => {
      this.ws.send(encodeStart(this.callId));
      // The caller's clock starts now, at piece 0.
      this.node.port.postMessage({ type: "start" });
      setStatus("Connected. Talk!");
    };
    this.ws.onmessage = (event) => this.fromRobot(event.data);
    this.ws.onclose = (event) => this.close(`The line closed (code ${event.code}).`);
    this.ws.onerror = () => setStatus("Could not reach the robot at that address.");
  }

  fromPhone(message) {
    const open = this.ws && this.ws.readyState === WebSocket.OPEN;
    if (!open || this.silent) return; // a silent drop sends nothing at all
    if (message.type === "piece") {
      this.ws.send(message.buffer);
      this.piecesSent += 1;
    } else if (message.type === "played") {
      this.ws.send(encodePlayed(message.name, message.caller_ms));
    } else if (message.type === "cleared") {
      this.ws.send(encodeCleared(message.played_pieces, message.caller_ms, message.dropped_marks));
    } else if (message.type === "error") {
      this.brokeRules(message.message);
    }
  }

  fromRobot(data) {
    if (data instanceof ArrayBuffer) {
      if (data.byteLength !== PIECE_BYTES) return this.brokeRules(`audio piece of ${data.byteLength} bytes`);
      this.node.port.postMessage({ type: "audio", buffer: data }, [data]);
      return;
    }
    let message;
    try {
      message = decodeRobotMessage(data);
    } catch (error) {
      return this.brokeRules(error.message);
    }
    if (message.type === "mark") this.node.port.postMessage({ type: "mark", name: message.name });
    else if (message.type === "clear") this.node.port.postMessage({ type: "clear" });
    else if (message.type === "end") this.close(`The robot hung up: ${message.reason}`);
  }

  pressKey(digit) {
    if (!this.silent) this.ws.send(encodeKey(digit, this.piecesSent * PIECE_MS));
  }

  hangUp() {
    if (!this.silent && this.ws.readyState === WebSocket.OPEN) this.ws.send(encodeEnd("bye"));
    this.close("You hung up.");
  }

  // For testing a robot's "no audio for 30 s" rule: the line goes dead
  // without a goodbye, exactly like a network dying in a lift.
  goSilent() {
    this.silent = true;
    setStatus("Silent drop: sending nothing. The robot should notice by itself.");
  }

  brokeRules(text) {
    // Close code 1002: the robot broke the wire's rules.
    if (this.ws.readyState === WebSocket.OPEN) this.ws.close(1002, text.slice(0, 120));
    this.close(`The robot broke the wire's rules: ${text}`);
  }

  close(reason) {
    if (this.closed) return;
    this.closed = true;
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) this.ws.close(1000);
    if (this.node) {
      this.node.port.postMessage({ type: "stop" });
      this.node.disconnect();
    }
    if (this.mic) this.mic.getTracks().forEach((track) => track.stop());
    if (this.audio) this.audio.close();
    setStatus(reason);
    setInCall(false);
    if (current === this) current = null;
  }
}

let current = null;

ui.call.addEventListener("click", async () => {
  setInCall(true);
  setStatus("Calling...");
  current = new Call(ui.url.value.trim());
  try {
    await current.open();
  } catch (error) {
    // Most often: microphone permission was refused, or the address is wrong.
    current.close(`Could not start the call: ${error.message}`);
  }
});
ui.hangUp.addEventListener("click", () => current && current.hangUp());
ui.silent.addEventListener("click", () => current && current.goSilent());
ui.keys.forEach((key) => key.addEventListener("click", () => current && current.pressKey(key.dataset.key)));
setInCall(false);
