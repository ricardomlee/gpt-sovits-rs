const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright-core');
const { serve } = require('./serve.cjs');
const { text: englishTranscript } = require('../examples/first-run/request.json');

function checkPackagerGuards(pkg) {
  const scratch = fs.mkdtempSync(path.join(tmpdir(), 'gpt-sovits-pages-'));
  const output = path.join(scratch, 'public site');
  const build = assets => spawnSync('bash', [path.join(__dirname, 'build.sh'), output, assets], { encoding: 'utf8' });
  try {
    const missing = build(path.join(scratch, 'missing package'));
    assert.equal(missing.status, 1);
    assert.match(missing.stderr, /Missing browser artifact/);
    assert(!fs.existsSync(output));

    fs.mkdirSync(output);
    const leftover = path.join(output, 'private.txt');
    fs.writeFileSync(leftover, 'keep unchanged');
    const occupied = build(pkg);
    assert.equal(occupied.status, 1);
    assert.match(occupied.stderr, /Refusing nonempty output directory/);
    assert.equal(fs.readFileSync(leftover, 'utf8'), 'keep unchanged');
  } finally {
    fs.rmSync(scratch, { recursive: true });
  }
}

(async () => {
  const directory = path.resolve(process.argv[2] || '_site');
  const files = fs.readdirSync(directory, { recursive: true })
    .filter(relative => fs.statSync(path.join(directory, relative)).isFile());
  assert.deepEqual(files.sort(), [
    '.nojekyll', 'assets/pages-browser-lab.webp', 'assets/pages-voice-console.webp',
    'audio/local-assistant-en.wav', 'audio/sun-greeting.wav', 'audio/sun-zh.wav', 'en/index.html', 'index.html', 'lab/checkpoint.js',
    'lab/en.html', 'lab/index.html', 'lab/lab.css', 'lab/lab.js', 'lab/pkg/wasm_sovits.js', 'lab/pkg/wasm_sovits_bg.wasm',
    'lab/worker.js', 'styles.css',
  ]);
  const englishWav = 'local-assistant-en.wav';
  assert.deepEqual(fs.readFileSync(path.join(directory, 'audio', englishWav)),
    fs.readFileSync(path.join(__dirname, '../examples/samples', englishWav)));
  checkPackagerGuards(path.join(directory, 'lab/pkg'));
  const browser = await chromium.launch({ channel: 'chrome', executablePath: process.env.CHROME_BIN });
  const { server, url } = await serve(directory);
  try {
    for (const [width, height, en] of [[1440, 1000, false], [1440, 1000, true], [320, 740, false], [320, 740, true]]) {
      const page = await browser.newPage({ viewport: { width, height } });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      const network = [];
      page.on('request', request => network.push([request.method(), request.url()]));
      const lang = en ? 'en' : 'zh-CN';
      const home = en ? 'en/' : '';
      const lab = en ? 'lab/en.html' : 'lab/';
      await page.goto(url + home);
      assert.equal(page.url(), url + home);
      assert.equal(await page.locator('html').getAttribute('lang'), lang);
      assert.equal(await page.locator('h1').innerText(), 'GPT-SoVITS-RS');
      assert.equal(await page.locator('link[rel="canonical"]').getAttribute('href'), 'https://tts.ricli.tech/' + home);
      const samples = await page.locator('.track').evaluateAll(tracks => tracks.map(track => ({
        wav: track.querySelector('audio').src, download: track.querySelector('a[download]').href,
        text: track.querySelector('.transcript').textContent,
      })));
      const names = en ? [englishWav, 'sun-greeting.wav', 'sun-zh.wav'] : ['sun-greeting.wav', 'sun-zh.wav', englishWav];
      assert.deepEqual(samples.map(s => s.wav), names.map(n => url + 'audio/' + n));
      assert(samples.every(s => s.download === s.wav));
      assert.equal(samples.find(s => s.wav.endsWith('/' + englishWav)).text, englishTranscript);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      const firstPlayer = await page.locator('audio').first().boundingBox();
      assert(firstPlayer.y + firstPlayer.height <= height, 'first player must be visible without scrolling');
      for (const player of await page.locator('audio').all()) {
        await player.evaluate(audio => audio.play());
        await page.waitForFunction(audio => audio.currentTime > 0.1, await player.elementHandle());
        assert(await player.evaluate(audio => Number.isFinite(audio.duration) && audio.duration > 2 && !audio.error));
        await player.evaluate(audio => { audio.pause(); audio.currentTime = 0; });
      }
      await page.locator('.lab-layout img').scrollIntoViewIfNeeded();
      await page.waitForFunction(() => [...document.images].every(i => i.complete && i.naturalWidth > 0));
      await page.evaluate(() => scrollTo(0, 0));
      assert.deepEqual(errors, []);

      await page.getByRole('link', { name: en ? 'Browser lab' : '浏览器实验', exact: true }).click();
      assert.equal(page.url(), url + lab);
      assert.equal(await page.locator('h1').innerText(), en ? 'Browser speech lab' : '浏览器语音实验');
      assert.equal(await page.locator('input[type="file"]').count(), 5);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      const render = page.getByRole('button', { name: en ? 'Generate WAV' : '生成 WAV' });
      await render.waitFor();
      if (width === 1440) {
        const fake = { name: 'fixture.safetensors', mimeType: 'application/octet-stream', buffer: Buffer.from('not a checkpoint') };
        for (const input of await page.locator('input[type="file"]').all()) await input.setInputFiles(fake);
        await page.locator('textarea').fill('你好。');
        const wasmResponse = page.waitForResponse(`${url}lab/pkg/wasm_sovits_bg.wasm`);
        await render.click();
        const wasm = await wasmResponse;
        assert(wasm.ok());
        assert.equal(wasm.headers()['content-type'], 'application/wasm');
        await page.waitForFunction(() => document.querySelector('#result').textContent.includes('Invalid safetensors checkpoint'));
        assert(await page.locator('input[name="sovits"]').isEnabled());
        assert(await page.locator('audio').isHidden());
      }
      const switcher = page.locator('a[hreflang]');
      await switcher.click();
      assert.equal(page.url(), url + (en ? 'lab/' : 'lab/en.html'));
      await switcher.click();
      await page.locator('.wordmark').click();
      assert.equal(page.url(), url + home);
      await switcher.click();
      assert.equal(page.url(), url + (en ? '' : 'en/'));
      assert(network.every(([method, target]) => method === 'GET' && target.startsWith(url)), JSON.stringify(network));
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('Pages: both locales, language switches, all three WAVs, and shared Worker/Wasm startup passed.');
  } finally {
    await browser.close();
    server.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
