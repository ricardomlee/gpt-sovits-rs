# Build the optional public demo archive with the converter from the extracted CPU package.
param(
    [Parameter(Mandatory)][string]$Converter,
    [Parameter(Mandatory)][string]$SourceDir,
    [Parameter(Mandatory)][string]$StageDir,
    [Parameter(Mandatory)][string]$Archive
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$repo = (Resolve-Path "$PSScriptRoot/../..").Path
$fixture = "$repo/examples/first-run"
$root = Join-Path $StageDir 'GPT-SoVITS Windows'
if (Test-Path -LiteralPath $root) { throw "Use a fresh model staging directory: $StageDir" }
foreach ($dir in 'models/bert', 'models/hubert', 'voices/demo', 'model-notices') {
    [void][IO.Directory]::CreateDirectory("$root/$dir")
}
foreach ($line in (Get-Content "$fixture/downloads.txt")) {
    if (!$line -or $line.StartsWith('#')) { continue }
    $sha, $relative, $url = $line -split '\s+', 3
    $file = Join-Path $SourceDir $relative
    [void][IO.Directory]::CreateDirectory((Split-Path -Parent $file))
    if (Test-Path -LiteralPath $file) {
        if ((Get-FileHash -LiteralPath $file).Hash -ne $sha) { throw "Invalid cached model input: $relative" }
    } else {
        curl.exe -fsSL --connect-timeout 20 --max-time 1200 --retry 2 --retry-all-errors --output "$file.part" $url
        if ((Get-FileHash -LiteralPath "$file.part").Hash -ne $sha) { throw "Invalid model download: $relative" }
        Move-Item -LiteralPath "$file.part" -Destination $file
    }
}
& $Converter gpt "$SourceDir/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt" "$root/models/gpt-model.safetensors"
& $Converter sovits "$SourceDir/gsv-v2final-pretrained/s2G2333k.pth" "$root/models/sovits-model.safetensors"
& $Converter bert "$SourceDir/chinese-roberta-wwm-ext-large/pytorch_model.bin" "$root/models/bert/bert.safetensors"
& $Converter hubert "$SourceDir/chinese-hubert-base/pytorch_model.bin" "$root/models/hubert/hubert.safetensors"
Copy-Item -LiteralPath "$SourceDir/chinese-roberta-wwm-ext-large/tokenizer.json" -Destination "$root/models/bert/"
Copy-Item -LiteralPath "$SourceDir/reference.wav" -Destination "$root/voices/demo/ref.wav"
Copy-Item -LiteralPath "$fixture/voice.json" -Destination "$root/voices/demo/"
Copy-Item -LiteralPath "$fixture/downloads.txt", "$repo/docs/WINDOWS.md" -Destination "$root/model-notices/"
[IO.Compression.ZipFile]::CreateFromDirectory($root, $Archive, 'Optimal', $true)
