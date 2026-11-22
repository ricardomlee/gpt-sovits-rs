const form = document.querySelector('form');
const inputs = form.querySelector('fieldset');
const button = form.querySelector('[type="submit"]');
const unload = document.querySelector('#unload');
const result = document.querySelector('#result');
const labels = result.dataset;
const audio = document.querySelector('audio');
const download = document.querySelector('#download');
const mib = (bytes) => (bytes / 2 ** 20).toFixed(1);

let worker;

function clearAudio() {
  if (audio.src) URL.revokeObjectURL(audio.src);
  audio.removeAttribute('src');
  audio.load();
  audio.hidden = download.hidden = true;
}

function reset() {
  worker?.terminate();
  worker = undefined;
  inputs.disabled = button.disabled = false;
  unload.disabled = true;
}

unload.addEventListener('click', () => {
  reset();
  result.textContent = labels.unloaded;
});

function startWorker() {
  const current = new Worker('./worker.js', { type: 'module' });
  worker = current;
  current.onmessage = ({ data }) => {
    if (current !== worker) return;
    if (data.stage) {
      result.textContent = data.stage === 'render' ? labels.rendering : `${labels.loading} ${data.stage}…`;
      return;
    }
    button.disabled = false;
    if (data.error) {
      if (data.reset) reset();
      result.textContent = data.error;
      return;
    }
    const { wav, loadMs, renderMs, peakBytes } = data;
    const url = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }));
    audio.src = download.href = url;
    audio.hidden = download.hidden = false;
    const load = loadMs === undefined ? '' : `${labels.load} ${Math.round(loadMs)} ms · `;
    result.textContent = `${load}${labels.render} ${Math.round(renderMs)} ms · ${labels.memory} ${mib(peakBytes)} MiB`;
  };
  current.onerror = (error) => {
    if (current !== worker) return;
    reset();
    result.textContent = error.message || labels.failed;
  };
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  clearAudio();
  let files;
  if (!worker) {
    files = Object.fromEntries(Array.from(inputs.querySelectorAll('input'), (input) => [input.name, input.files[0]]));
    startWorker();
  }
  inputs.disabled = button.disabled = true;
  unload.disabled = false;
  result.textContent = labels.rendering;
  worker.postMessage({ files, text: form.elements.text.value });
});
button.disabled = false;
