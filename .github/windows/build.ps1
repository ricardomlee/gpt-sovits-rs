param(
    [Parameter(Mandatory)][ValidateSet('cpu', 'cuda')][string]$Backend,
    [Parameter(Mandatory)][string]$NativeDir
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
. "$PSScriptRoot/enter-msvc.ps1"
$prefix = (Resolve-Path $NativeDir).Path
$env:PKG_CONFIG_PATH = "$prefix/lib/pkgconfig" -replace '\\', '/'
$env:LAME_LIB_DIR = "$prefix/lib"
$env:CMAKE_GENERATOR = 'Ninja'
$env:CMAKE_POLICY_VERSION_MINIMUM = '3.5'
$features = 'http-api'
if ($Backend -eq 'cuda') {
    $features += ',cuda'
    $env:CUDA_COMPUTE_CAP = '89'
    $env:CUDAFORGE_THREADS = '4'
}
cargo build --release --locked --bins --features $features --target-dir target
if ($Backend -eq 'cpu') {
    cargo build --release --locked --manifest-path client/Cargo.toml --target-dir target
}
