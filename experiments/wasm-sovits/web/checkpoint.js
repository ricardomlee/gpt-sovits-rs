import { checkpoint_header_end, CheckpointLoader } from './pkg/wasm_sovits.js';

async function read(file, start, end) {
  return new Uint8Array(await file.slice(start, end).arrayBuffer());
}

export async function loadCheckpoint(file) {
  if (file.size > 0xffff_ffff) throw new Error('Checkpoint exceeds the 4 GiB file limit');
  const end = checkpoint_header_end(await read(file, 0, 8), file.size);
  const loader = new CheckpointLoader(await read(file, 0, end), file.size);
  try {
    for (let range; (range = loader.next_range());) {
      loader.push(await read(file, range[0], range[1]));
    }
    return loader;
  } catch (error) {
    loader.free();
    throw error;
  }
}
