# Windows x64 packages

Download the CPU or CUDA ZIP from a release with Windows assets and extract it. All three ZIPs use the same top-level `GPT-SoVITS Windows` directory.

| ZIP suffix | Contents | When to download |
| --- | --- | --- |
| `windows-x64-cpu-runtime.zip` | CPU engine and HTTP server, converter, standalone HTTP client, audio DLLs | CPU inference or conversion |
| `windows-x64-cuda-sm89.zip` | CUDA engine and HTTP server, audio DLLs | An NVIDIA GPU with compute capability 8.9, such as the RTX 4060 Ti |
| `demo-v2-models.zip` | Converted public v2 weights, tokenizer, English reference audio and voice profile | First run without your own models |

For GPU inference, install a compatible NVIDIA driver and the [official CUDA Toolkit 12.5.1 for Windows x86_64](https://developer.nvidia.com/cuda-12-5-1-download-archive). This build uses the system copies of `cublas64_12.dll`, `cublasLt64_12.dll` and `curand64_10.dll`. The launchers check `CUDA_PATH_V12_5\bin`, `CUDA_PATH\bin` and PATH. For direct executable use, put the toolkit's `bin` directory on PATH. CPU inference needs only the CPU package and models.

CPU and GPU share one copy of `models/` and `voices/`. The CPU package includes the converter and HTTP client for either backend. Compare `Get-FileHash .\your-package.zip` with the release's `SHA256SUMS`.

## Download the public demo models

Open the extracted `GPT-SoVITS Windows` folder and run:

```powershell
.\download-models.cmd
```

The launcher downloads the matching release's model ZIP (about 1.1 GiB), checks its SHA-256 against `SHA256SUMS`, and installs the converted weights and English demo voice. Run `.\demo-cpu.cmd` or `.\demo-cuda.cmd` to hear it. Users with their own models and voice can go straight to inference.

For an offline install, put the model ZIP and its matching `SHA256SUMS` in the same folder and run `.\download-models.cmd -Source "D:\Downloads\GPT-SoVITS"`. With a CI candidate, use the model artifact from the same workflow run. A repeat install recognizes the verified model archive. If a demo file conflicts with an existing file, use a separate installation folder to keep both.

## Run

From the extracted `GPT-SoVITS Windows` directory, run `.\demo-cpu.cmd` or `.\demo-cuda.cmd` to create a WAV using the public voice. Run `.\serve-cpu.cmd` or `.\serve-cuda.cmd` to start the HTTP server at `http://127.0.0.1:9881`. Ctrl+C stops it. With your own models and voice, supply their paths explicitly.

```powershell
.\gpt-sovits.exe --doctor --device cpu --models-dir D:\tts\models --voices-dir D:\tts\voices --voice my_voice
.\gpt-sovits.exe --device cpu --models-dir D:\tts\models --voices-dir D:\tts\voices --voice my_voice --text "The rain has stopped." --output first.wav
```

For CUDA, use `.\cuda-sm89\gpt-sovits.exe --device cuda` with the same model and voice arguments. Keep the root directory as the working directory.

## Package contents and sources

The public weights come from [lj1995/GPT-SoVITS at revision 1cdb10a](https://huggingface.co/lj1995/GPT-SoVITS/tree/1cdb10a4f2b13b3e8103b873c4c70c9d54d92c85). The [model card](https://huggingface.co/lj1995/GPT-SoVITS/blob/1cdb10a4f2b13b3e8103b873c4c70c9d54d92c85/README.md) identifies the upstream license. The reference is LJ001-0001, read by Linda Johnson and aligned by Keith Ito, from [LJ Speech 1.1](https://keithito.com/LJ-Speech-Dataset/). The model archive includes the pinned source URLs and SHA-256 hashes in `model-notices/downloads.txt`. The build verifies them before conversion.

The audio libraries are replaceable DLLs beside each engine. Their source archives, the applied soxr patches, and notices are under `licenses/`.

## CI verification

The Windows workflow compiles CPU and CUDA on hosted x64 runners. It extracts the CPU runtime, installs the model archive with the bundled downloader, runs doctor and synthesis, and checks the output WAV. The CUDA build targets SM 8.9 explicitly so compilation can run without a GPU.

To exercise CUDA hardware, dispatch **Windows packages** on the default branch and enable **Run synthesis on a self-hosted Windows SM 8.9 GPU**. This optional job needs a self-hosted x64 Windows runner labeled `sm89`, a working NVIDIA driver, CUDA Toolkit 12.5.1, PowerShell 7, `nsys` and `sqlite3` on PATH. It extracts the GPU and model ZIPs, uses the system CUDA libraries and counts GPU kernel executions during synthesis.
