import init, { AcousticModel } from './pkg/wasm_sovits.js';

self.onmessage = async ({ data: { weights, caseFile } }) => {
  let model;
  try {
    const { memory } = await init();
    const bytes = new Uint8Array(await weights.arrayBuffer());
    const input = new Uint8Array(await caseFile.arrayBuffer());
    const start = performance.now();
    model = new AcousticModel(bytes);
    const loadMs = performance.now() - start;
    const loadBytes = memory.buffer.byteLength;
    const renderStart = performance.now();
    const wav = model.render(input);
    const renderMs = performance.now() - renderStart;
    const peakBytes = memory.buffer.byteLength;
    self.postMessage({ wav, loadMs, renderMs, loadBytes, peakBytes }, [wav.buffer]);
  } catch (error) {
    self.postMessage({ error: error.message || String(error) });
  } finally {
    model?.free();
  }
};
