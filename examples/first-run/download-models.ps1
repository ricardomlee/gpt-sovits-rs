param(
    [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [string]$InstallDir = $PSScriptRoot,
    [string]$Source = "https://github.com/ricardomlee/gpt-sovits-rs/releases/download/v$Version"
)

$ErrorActionPreference = 'Stop'
$root = (Get-Item -LiteralPath $InstallDir).FullName
$name = "gpt-sovits-$Version-demo-v2-models.zip"
$files = @(
    'models/gpt-model.safetensors', 'models/sovits-model.safetensors',
    'models/bert/bert.safetensors', 'models/bert/tokenizer.json', 'models/hubert/hubert.safetensors',
    'voices/demo/ref.wav', 'voices/demo/voice.json', 'model-notices/downloads.txt', 'model-notices/WINDOWS.md'
)
$receipt = Join-Path $root "model-notices/$name.sha256"
$work = New-Item -ItemType Directory -Path (Join-Path $root ('.models-' + [guid]::NewGuid()))

function Get-ReleaseAsset([string]$FileName) {
    if ($Source -notmatch '^https?://') { return (Get-Item -LiteralPath (Join-Path $Source $FileName)).FullName }
    $download = Join-Path $work.FullName $FileName
    $url = $Source.TrimEnd('/') + '/' + $FileName
    Write-Host "Downloading $url"
    & curl.exe --fail --location --retry 2 --connect-timeout 20 --output $download $url
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $FileName (curl exit $LASTEXITCODE)." }
    return $download
}

function Get-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '') }
    finally { $stream.Dispose(); $sha.Dispose() }
}

try {
    $sums = Get-ReleaseAsset 'SHA256SUMS'
    $expected = @(foreach ($line in (Get-Content -LiteralPath $sums)) {
        if ($line -cmatch '^([a-fA-F0-9]{64}) [ *](?:\./)?(.+)$' -and $Matches[2] -ceq $name) {
            $Matches[1]
        }
    })
    if ($expected.Count -ne 1) { throw "Expected exactly one SHA-256 entry for $name in SHA256SUMS." }
    $hash = $expected[0]
    if ([IO.File]::Exists($receipt) -and ([IO.File]::ReadAllText($receipt)).Trim() -eq $hash -and
        !($files | Where-Object { ![IO.File]::Exists((Join-Path $root $_)) })) {
        Write-Host "The v$Version demo models are already installed."
        return
    }

    $zip = Get-ReleaseAsset $name
    if ((Get-Sha256 $zip) -ne $hash) { throw "SHA-256 mismatch for $name." }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($zip, $work.FullName)
    $payload = Join-Path $work.FullName 'GPT-SoVITS Windows'

    # Check for conflicts before installing any files. A retry can reuse identical files.
    $missing = @(foreach ($relative in $files) {
        $staged = Get-Item -LiteralPath (Join-Path $payload $relative)
        $target = Join-Path $root $relative
        if (!(Test-Path -LiteralPath $target)) {
            $relative
        } else {
            $existing = Get-Item -LiteralPath $target
            if ($existing.PSIsContainer -or $existing.Length -ne $staged.Length -or
                (Get-Sha256 $target) -ne (Get-Sha256 $staged.FullName)) {
                throw "A different file already exists at $target. Use a separate installation directory."
            }
        }
    })
    foreach ($relative in $missing) {
        $target = Join-Path $root $relative
        [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target))
        [IO.File]::Move((Join-Path $payload $relative), $target)
    }
    [IO.File]::WriteAllText($receipt, $hash)
    Write-Host "Installed the v$Version public demo models in $root."
} finally {
    if ($work.Parent.FullName -eq $root) { Remove-Item -LiteralPath $work.FullName -Recurse -Force }
}
