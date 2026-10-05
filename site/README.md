# Public audio showcase

A buildless, static GitHub Pages site. No backend, analytics, inference endpoint,
third-party fonts, or browser framework. Only explicitly reviewed public audio is published.

The two showcase players use the maintainer's Sun v2Pro voice with its SV embedding.
The model, embedding, and reference recording are not published. The installable
first-run demo still uses standard v2 weights and an LJ Speech reference; it does
not install or reproduce Sun.

## Preview

```bash
bash site/build.sh /tmp/gpt-sovits-site
```

Open `/tmp/gpt-sovits-site/index.html` in a browser; no development server is needed.
The output contains only HTML, CSS, the existing banner, and the two public WAV files.

Browser checks (Playwright 1.59.1, Chromium) cover desktop and narrow mobile layouts,
both audio players actually advancing, loaded images, and download targets:

```bash
node site/check.cjs /tmp/gpt-sovits-site
```

Playwright is a development-only dependency. The Pages workflow installs it outside
the repository and uploads screenshots; the published site needs no JavaScript.

## Publish

In repository **Settings > Pages > Build and deployment**, select **GitHub Actions**
once. Merge this PR to `master`; the **Audio showcase** workflow validates and deploys.
Pull requests only build and test, never deploy. Later changes to public samples or
the site redeploy automatically. The expected URL is:

`https://ricardomlee.github.io/gpt-sovits-rs/`

Do not advertise that URL as live until the first deploy succeeds. Once it is live,
set the repository website field to it and link it beside the README's WAV download.
Forks need to adjust repository links in `index.html` and enable their own Pages site.
