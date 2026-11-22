import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';

const files = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/en.html', ['en.html', 'text/html; charset=utf-8']],
  ['/lab.css', ['lab.css', 'text/css; charset=utf-8']],
  ['/lab.js', ['lab.js', 'text/javascript']],
  ['/worker.js', ['worker.js', 'text/javascript']],
  ['/checkpoint.js', ['checkpoint.js', 'text/javascript']],
  ['/pkg/wasm_sovits.js', ['pkg/wasm_sovits.js', 'text/javascript']],
  ['/pkg/wasm_sovits_bg.wasm', ['pkg/wasm_sovits_bg.wasm', 'application/wasm']],
]);
const port = Number(process.argv[2] ?? 9898);

createServer(async (req, res) => {
  if (req.method === 'GET' && req.url === '/en/') return res.writeHead(302, { Location: '/en.html' }).end();
  const file = req.method === 'GET' && files.get(req.url);
  if (!file) return res.writeHead(404).end();
  const [path, type] = file;
  try {
    const data = await readFile(new URL(`./web/${path}`, import.meta.url));
    res.writeHead(200, { 'Content-Type': type }).end(data);
  } catch (error) {
    console.error(error.message);
    res.writeHead(404).end();
  }
}).listen(port, '127.0.0.1', () => console.log(`http://127.0.0.1:${port}/`));
