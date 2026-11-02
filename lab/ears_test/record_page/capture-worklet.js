// Hands each block of microphone samples to the page. The block is copied
// because the browser reuses its memory for the next one.
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}

registerProcessor("capture", CaptureProcessor);
