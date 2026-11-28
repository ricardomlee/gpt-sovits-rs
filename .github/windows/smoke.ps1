# Exercise an extracted package. The GPU test uses the host's CUDA Toolkit, Nsight Systems and sqlite3.
param(
    [Parameter(Mandatory)][ValidateSet('cpu', 'cuda')][string]$Backend,
    [Parameter(Mandatory)][string]$DistDir,
    [Parameter(Mandatory)][string]$InstallDir
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
if (Test-Path -LiteralPath $InstallDir) { throw "Use a fresh extraction directory: $InstallDir" }
$patterns = if ($Backend -eq 'cpu') {
    '*-cpu-runtime.zip', '*-demo-v2-models.zip'
} else {
    '*-cuda-sm89.zip', '*-demo-v2-models.zip'
}
foreach ($pattern in $patterns) {
    $files = @(Get-ChildItem -LiteralPath $DistDir -Filter $pattern -File)
    if ($files.Count -ne 1) { throw "Expected one $pattern in $DistDir, found $($files.Count)" }
    [IO.Compression.ZipFile]::ExtractToDirectory($files[0].FullName, $InstallDir)
}
$root = (Resolve-Path "$InstallDir/GPT-SoVITS Windows").Path
$program = "$root/gpt-sovits.exe"
if ($Backend -eq 'cuda') {
    $program = "$root/cuda-sm89/gpt-sovits.exe"
    $nsys = (Get-Command nsys.exe).Source
    $sqlite = (Get-Command sqlite3.exe).Source
    $toolkit = if ($env:CUDA_PATH_V12_5) { $env:CUDA_PATH_V12_5 } else { $env:CUDA_PATH }
    foreach ($dll in 'cublas64_12.dll', 'cublasLt64_12.dll', 'curand64_10.dll') {
        if (Test-Path -LiteralPath "$root/cuda-sm89/$dll") { throw "Bundled CUDA library: $dll" }
        if (!(Test-Path -LiteralPath "$toolkit/bin/$dll")) { throw "Missing system CUDA library: $toolkit/bin/$dll" }
    }
}
$savedPath = $env:PATH
Push-Location $root
try {
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    if ($Backend -eq 'cpu') {
        & ./gpt-sovits-convert.exe --version
        & ./gpt-sovits-client.exe --version
    } else {
        $env:PATH = "$toolkit\bin;$env:PATH"
    }
    & $program --doctor --device $Backend --voice demo
    $wav = "$root/demo-$Backend.wav"
    $synthesis = @('--device', $Backend, '--mode', 'kv', '--voice', 'demo', '--seed', '42',
                   '--text', 'The rain has stopped. We can walk home by the river.', '--output', $wav)
    if ($Backend -eq 'cpu') {
        & $program @synthesis
    } else {
        & $nsys profile --trace=cuda --sample=none --cpuctxsw=none --export=sqlite --output=cuda-trace $program @synthesis
        $kernels = & $sqlite -readonly cuda-trace.sqlite 'SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL'
        if ([long]$kernels -le 0) { throw 'CUDA synthesis did not execute any GPU kernels' }
        Write-Host "$kernels GPU kernel executions"
    }
    if ((Get-Item -LiteralPath $wav).Length -le 44) { throw 'Empty synthesis output' }
} finally {
    $env:PATH = $savedPath
    Pop-Location
}
