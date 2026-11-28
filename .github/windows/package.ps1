param(
    [Parameter(Mandatory)][ValidateSet('cpu', 'cuda')][string]$Backend,
    [Parameter(Mandatory)][string]$Version,
    [Parameter(Mandatory)][string]$BinDir,
    [Parameter(Mandatory)][string]$NativeDir,
    [Parameter(Mandatory)][string]$NativeSource,
    [Parameter(Mandatory)][string]$StageDir,
    [Parameter(Mandatory)][string]$DistDir
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$repo = (Resolve-Path "$PSScriptRoot/../..").Path
$root = Join-Path $StageDir 'app/GPT-SoVITS Windows'
$bin = if ($Backend -eq 'cpu') { $root } else { Join-Path $root 'cuda-sm89' }
if (Test-Path -LiteralPath $root) { throw "Use a fresh staging directory: $StageDir" }
[void][IO.Directory]::CreateDirectory("$bin/licenses")
[void][IO.Directory]::CreateDirectory($DistDir)
Copy-Item -LiteralPath "$BinDir/gpt-sovits.exe", "$BinDir/gpt-sovits-convert.exe", "$NativeDir/bin/soxr.dll", "$NativeDir/bin/libmp3lame.dll" -Destination $bin
Copy-Item -LiteralPath "$repo/LICENSE", "$repo/THIRD_PARTY.md" -Destination $bin
Copy-Item "$repo/licenses/*.txt" "$bin/licenses/"
Copy-Item -Recurse -LiteralPath $NativeSource -Destination "$bin/licenses/native-source"
Copy-Item -LiteralPath "$repo/docs/WINDOWS.md" -Destination "$bin/README-Windows.md"
Copy-Item -LiteralPath "$repo/examples/first-run/prepare-models.cmd" -Destination $root
$variant = 'cpu-runtime'
$program = 'gpt-sovits.exe'
$launcher = @('@echo off', 'setlocal', 'cd /d "%~dp0"')
if ($Backend -eq 'cpu') {
    Copy-Item -LiteralPath "$BinDir/gpt-sovits-client.exe" -Destination $bin
} else {
    $variant = 'cuda-sm89'
    $program = 'cuda-sm89\gpt-sovits.exe'
    $launcher += 'if defined CUDA_PATH if exist "%CUDA_PATH%\bin" set "PATH=%CUDA_PATH%\bin;%PATH%"'
    $launcher += 'if defined CUDA_PATH_V12_5 if exist "%CUDA_PATH_V12_5%\bin" set "PATH=%CUDA_PATH_V12_5%\bin;%PATH%"'
    $launcher += 'for %%L in (cublas64_12.dll cublasLt64_12.dll curand64_10.dll) do ('
    $launcher += '  where /q %%L || (echo Missing %%L. Install NVIDIA CUDA Toolkit 12.5.1 and reopen your terminal. & exit /b 1)'
    $launcher += ')'
}

# Follow PE imports of the executables and audio DLLs. Only the MSVC redistributable is copied.
$vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$msvc = (Get-Content "$vs/VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt" -Raw).Trim()
$redist = (Get-Content "$vs/VC/Auxiliary/Build/Microsoft.VCRedistVersion.default.txt" -Raw).Trim()
$crt = "$vs/VC/Redist/MSVC/$redist/x64/Microsoft.VC143.CRT"
$dumpbin = "$vs/VC/Tools/MSVC/$msvc/bin/Hostx64/x64/dumpbin.exe"
$queue = [Collections.Generic.Queue[string]]::new()
Get-ChildItem $bin -File | Where-Object Extension -In '.exe', '.dll' | ForEach-Object { $queue.Enqueue($_.FullName) }
while ($queue.Count) {
    $file = $queue.Dequeue()
    foreach ($line in (& $dumpbin /nologo /dependents $file)) {
        if ($line -notmatch '^\s+([\w.-]+\.dll)\s*$') { continue }
        $dll = $Matches[1]
        if (Test-Path -LiteralPath "$bin/$dll") { continue }
        if (Test-Path -LiteralPath "$crt/$dll") {
            Copy-Item -LiteralPath "$crt/$dll" -Destination $bin
            $queue.Enqueue("$bin/$dll")
        } elseif ($Backend -eq 'cuda' -and $dll -in 'nvcuda.dll', 'cublas64_12.dll', 'curand64_10.dll') {
            continue # Provided by the installed NVIDIA driver and CUDA Toolkit.
        } elseif ($dll -notmatch '^(api|ext)-ms-win-' -and !(Test-Path -LiteralPath "$env:SystemRoot/System32/$dll")) {
            throw "Unresolved import $dll in $file"
        }
    }
}

$demo = $launcher + @(
    'set "data=."',
    'if exist "demo-data\.gpt-sovits-demo" set "data=demo-data"',
    ('"%~dp0{0}" --device {1} --models-dir "%data%\models" --voices-dir "%data%\voices" --mode kv --voice demo --seed 42 --text "The rain has stopped. We can walk home by the river." --output "%~dp0demo-{1}.wav"' -f $program, $Backend),
    'exit /b %errorlevel%')
$serve = $launcher + @(
    'set "data=."',
    'if not exist "models\" if not exist "voices\" if exist "demo-data\.gpt-sovits-demo" set "data=demo-data"',
    ('"%~dp0{0}" --http --host 127.0.0.1 --port 9881 --device {1} --models-dir "%data%\models" --voices-dir "%data%\voices"' -f $program, $Backend),
    'exit /b %errorlevel%')
[IO.File]::WriteAllLines("$root/demo-$Backend.cmd", $demo, [Text.Encoding]::ASCII)
[IO.File]::WriteAllLines("$root/serve-$Backend.cmd", $serve, [Text.Encoding]::ASCII)
$zip = "gpt-sovits-$Version-windows-x64-$variant.zip"
[IO.Compression.ZipFile]::CreateFromDirectory($root, "$DistDir/$zip", 'Optimal', $true)
