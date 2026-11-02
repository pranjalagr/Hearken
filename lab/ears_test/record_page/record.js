// The recording page: one line at a time, record, listen, save. The server keeps
// what is saved, so closing the page loses nothing and the next visit carries on.
import { startTake } from "./take.js";
import { encodeWav, mergeChunks } from "./wav.js";

const $ = (id) => document.getElementById(id);
let lines = [];
let index = 0;
let audioContext = null;
let microphone = null;
let source = null;
let take = null;
let wav = null;

function setState({ recording = false, hasTake = false } = {}) {
  $("record").disabled = recording || !lines.length;
  $("stop").disabled = !recording;
  $("save").disabled = !hasTake;
  $("take").hidden = !hasTake;
}

function show() {
  const done = lines.filter((line) => line.done).length;
  $("progress").textContent = `${done} of ${lines.length} saved`;
  const line = lines[index];
  if (!line) {
    $("line").textContent = "All done. Thank you!";
    $("line-id").textContent = "";
    $("record").disabled = true;
    return;
  }
  $("line-id").textContent = `Line ${line.id}${line.done ? " (already saved: recording again replaces it)" : ""}`;
  $("line").textContent = line.say;
}

function nextUnsaved(from) {
  const after = lines.findIndex((line, i) => i >= from && !line.done);
  return after >= 0 ? after : lines.findIndex((line) => !line.done);
}

function moveTo(position) {
  index = position < 0 ? lines.length : position;
  wav = null;
  $("message").textContent = "";
  setState();
  show();
}

async function loadLines() {
  const speaker = $("speaker").value.trim();
  const response = await fetch(`/lines?speaker=${encodeURIComponent(speaker)}`);
  lines = await response.json();
  moveTo(nextUnsaved(0));
}

async function start() {
  $("message").textContent = "";
  try {
    if (!audioContext) {
      // Echo cancelling and noise removal stay off: they would clean the voice
      // before the test, and our noise is added later on purpose. Automatic
      // volume stays on: with it off, the first real take peaked at only 1.5%
      // of full volume (2 Nov 2026), and phones level the voice too.
      microphone = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true },
      });
      audioContext = new AudioContext();
      await audioContext.audioWorklet.addModule("capture-worklet.js");
    }
    await audioContext.resume();
    source ??= audioContext.createMediaStreamSource(microphone);
    take = startTake(source, () => new AudioWorkletNode(audioContext, "capture"));
    setState({ recording: true });
  } catch (error) {
    $("message").textContent = `The microphone didn't start: ${error.message}`;
  }
}

function stop() {
  wav = encodeWav(mergeChunks(take.stop()), audioContext.sampleRate);
  take = null;
  $("take").src = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
  setState({ hasTake: true });
}

async function save() {
  const speaker = $("speaker").value.trim();
  const line = lines[index];
  const response = await fetch(`/save?speaker=${encodeURIComponent(speaker)}&line=${encodeURIComponent(line.id)}`, {
    method: "POST",
    body: wav,
  });
  if (!response.ok) {
    $("message").textContent = `Not saved: ${(await response.json()).error}`;
    return;
  }
  line.done = true;
  moveTo(nextUnsaved(index + 1));
}

$("record").addEventListener("click", start);
$("stop").addEventListener("click", stop);
$("save").addEventListener("click", save);
$("speaker").addEventListener("change", loadLines);
$("skip").addEventListener("click", () => moveTo(lines.length ? (index + 1) % lines.length : 0));
// To redo a take you weren't happy with: saving again replaces it.
$("back").addEventListener("click", () => moveTo(lines.length ? (index - 1 + lines.length) % lines.length : 0));
loadLines();
