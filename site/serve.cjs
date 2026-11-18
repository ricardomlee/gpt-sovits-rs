const { createServer } = require('node:http');
const { readFile } = require('node:fs/promises');
const path = require('node:path');

const prefix = '/gpt-sovits-rs/';
const mime = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript',
  '.wasm': 'application/wasm',
  '.wav': 'audio/wav',
  '.webp': 'image/webp',
};

async function serve(directory, port = 0) {
  const root = path.resolve(directory);
  const server = createServer(async (request, response) => {
    const route = new URL(request.url, 'http://localhost').pathname;
    if (request.method !== 'GET' || !route.startsWith(prefix)) return response.writeHead(404).end();
    const relative = route.slice(prefix.length) + (route.endsWith('/') ? 'index.html' : '');
    const file = path.resolve(root, relative);
    const type = mime[path.extname(file)];
    if (!file.startsWith(root + path.sep) || !type) return response.writeHead(404).end();
    try {
      const data = await readFile(file);
      response.writeHead(200, { 'Content-Type': type, 'Content-Length': data.length }).end(data);
    } catch {
      response.writeHead(404).end();
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', resolve);
  });
  return { server, url: `http://127.0.0.1:${server.address().port}${prefix}` };
}

module.exports = { serve };

if (require.main === module) {
  serve(process.argv[2] || '_site', Number(process.argv[3] || 9898))
    .then(({ url }) => console.log(url))
    .catch(error => { console.error(error.message); process.exitCode = 1; });
}
