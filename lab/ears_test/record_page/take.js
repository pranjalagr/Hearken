// One recording take. Kept apart from the browser parts so Node can test it.
//
// Stopping must unplug this take's capture from the microphone. The first
// version only disconnected the capture's own output, so old captures kept
// running: the second take wrote every block twice, which made it twice as long,
// slow and choppy (found by ear, 2 Nov 2026).
export function startTake(source, makeCapture) {
  const blocks = [];
  const capture = makeCapture();
  capture.port.onmessage = (event) => blocks.push(event.data);
  source.connect(capture);
  return {
    stop() {
      source.disconnect(capture);
      capture.port.onmessage = null;
      return blocks;
    },
  };
}
