import assert from "node:assert/strict";
import { test } from "node:test";

import { encodeWav, mergeChunks } from "./wav.js";

test("chunks are joined in order", () => {
  const merged = mergeChunks([new Float32Array([0.1, 0.2]), new Float32Array([0.3])]);
  assert.equal(merged.length, 3);
  assert.ok(Math.abs(merged[2] - 0.3) < 1e-6);
});

test("the header states the rate, one channel and 16 bits", () => {
  const view = new DataView(encodeWav(new Float32Array(10), 48000).buffer);
  assert.equal(view.getUint32(24, true), 48000);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 20);
});

test("a sample past full scale is clipped, not wrapped around", () => {
  const view = new DataView(encodeWav(new Float32Array([2, -2]), 8000).buffer);
  assert.equal(view.getInt16(44, true), 32767);
  assert.equal(view.getInt16(46, true), -32768);
});
