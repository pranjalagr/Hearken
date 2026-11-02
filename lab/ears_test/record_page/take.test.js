import assert from "node:assert/strict";
import { test } from "node:test";

import { startTake } from "./take.js";

// A stand-in for the microphone: it sends each block to whatever is plugged in.
function fakeMicrophone() {
  const plugged = new Set();
  return {
    connect: (node) => plugged.add(node),
    disconnect: (node) => plugged.delete(node),
    say: (block) => plugged.forEach((node) => node.port.onmessage?.({ data: block })),
    plugged,
  };
}
const makeCapture = () => ({ port: { onmessage: null } });

test("every take holds each block exactly once, however many came before it", () => {
  const microphone = fakeMicrophone();
  for (let takeNumber = 1; takeNumber <= 3; takeNumber += 1) {
    const take = startTake(microphone, makeCapture);
    microphone.say("a");
    microphone.say("b");
    assert.deepEqual(take.stop(), ["a", "b"]);
  }
});

test("stopping unplugs the take from the microphone", () => {
  const microphone = fakeMicrophone();
  const take = startTake(microphone, makeCapture);
  assert.equal(microphone.plugged.size, 1);
  const blocks = take.stop();
  assert.equal(microphone.plugged.size, 0);
  microphone.say("later");
  assert.deepEqual(blocks, []);
});
