import init, { AcousticModel, TextVoice } from './pkg/wasm_sovits.js';
import { loadCheckpoint } from './checkpoint.js';

let voice;
let memory;

self.onmessage = async ({ data: { files, text } }) => {
  let loadMs;
  try {
    if (!voice) {
      ({ memory } = await init());
      const start = performance.now();
      self.postMessage({ stage: '正在加载 SoVITS…' });
      const sovits = AcousticModel.from_checkpoint(await loadCheckpoint(files.sovits));
      self.postMessage({ stage: '正在加载 GPT…' });
      const gpt = await loadCheckpoint(files.gpt);
      self.postMessage({ stage: '正在加载 BERT…' });
      const bert = await loadCheckpoint(files.bert);
      const tokenizer = new Uint8Array(await files.tokenizer.arrayBuffer());
      const reference = new Uint8Array(await files.reference.arrayBuffer());
      voice = new TextVoice(sovits, gpt, bert, tokenizer, reference);
      loadMs = performance.now() - start;
    }
    self.postMessage({ stage: '正在合成…' });
    const renderStart = performance.now();
    const wav = voice.render(text);
    const renderMs = performance.now() - renderStart;
    const peakBytes = memory.buffer.byteLength;
    self.postMessage({ wav, loadMs, renderMs, peakBytes }, [wav.buffer]);
  } catch (error) {
    self.postMessage({ error: error.message || String(error), reset: !voice || error instanceof WebAssembly.RuntimeError });
  }
};
