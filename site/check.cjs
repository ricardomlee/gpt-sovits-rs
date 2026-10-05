const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('playwright');

(async () => {
  const directory = path.resolve(process.argv[2] || '_site');
  const screenshots = path.resolve(process.argv[3] || '/tmp/gpt-sovits-pages');
  fs.mkdirSync(screenshots, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  try {
    for (const [width, height] of [[1440, 1000], [390, 844], [320, 740]]) {
      const page = await browser.newPage({ viewport: { width, height } });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(pathToFileURL(path.join(directory, 'index.html')).href);
      await page.waitForFunction(() => [...document.images].every(i => i.complete && i.naturalWidth > 0));
      assert.equal(await page.locator('h1').innerText(), 'GPT-SoVITS-RS');
      assert.equal(await page.locator('audio').count(), 2);
      assert.deepEqual(await page.locator('audio').evaluateAll(players => players.map(a => a.getAttribute('src'))),
        ['audio/sun-greeting.wav', 'audio/sun-zh.wav']);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      const firstPlayer = await page.locator('audio').first().boundingBox();
      assert(firstPlayer.y < height, 'first player must be visible without scrolling');
      for (const player of await page.locator('audio').all()) {
        await player.evaluate(audio => audio.play());
        await page.waitForFunction(audio => audio.currentTime > 0.1, await player.elementHandle());
        assert(await player.evaluate(audio => Number.isFinite(audio.duration) && audio.duration > 2 && !audio.error));
        await player.evaluate(audio => { audio.pause(); audio.currentTime = 0; });
      }
      for (const link of await page.locator('a[download]').all()) {
        assert(fs.existsSync(path.join(directory, await link.getAttribute('href'))));
      }
      assert.deepEqual(errors, []);
      await page.screenshot({ path: path.join(screenshots, `page-${width}.png`), fullPage: true });
      await page.close();
    }
    console.log('Pages: audio playback, assets, responsive layout, and downloads passed.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
