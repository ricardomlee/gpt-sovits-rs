# Build the replaceable audio DLLs from hash-pinned upstream source.
param(
    [Parameter(Mandatory)][string]$WorkDir,
    [Parameter(Mandatory)][string]$SoxrPort
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
. "$PSScriptRoot/enter-msvc.ps1"
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
$SoxrPort = (Resolve-Path $SoxrPort).Path
$sources = Join-Path $WorkDir 'native-source'
$prefix = Join-Path $WorkDir 'native'
[void][IO.Directory]::CreateDirectory($sources)

function Get-NativeSource($name, $mirror, $upstream, $sha512) {
    $file = Join-Path $sources $name
    if (!(Test-Path -LiteralPath $file)) {
        $options = @('-fsSL', '--connect-timeout', '15', '--max-time', '90', '--retry', '1', '--output', "$file.part")
        try { curl.exe @options $mirror }
        catch { curl.exe @options $upstream }
        if ((Get-FileHash "$file.part" -Algorithm SHA512).Hash -ne $sha512) { throw "Invalid SHA512 for $name" }
        Move-Item -LiteralPath "$file.part" -Destination $file
    } elseif ((Get-FileHash $file -Algorithm SHA512).Hash -ne $sha512) {
        throw "Invalid cached source: $name"
    }
    # The upstream soxr test symlink needs elevated privileges on some Windows hosts.
    tar.exe -xf $file -C $WorkDir --exclude '*/inst-check-soxr-lsr'
}
Get-NativeSource 'soxr-0.1.3-Source.tar.xz' 'https://deb.debian.org/debian/pool/main/libs/libsoxr/libsoxr_0.1.3.orig.tar.xz' 'https://downloads.sourceforge.net/project/soxr/soxr-0.1.3-Source.tar.xz' 'f4883ed298d5650399283238aac3dbe78d605b988246bea51fa343d4a8ce5ce97c6e143f6c3f50a3ff81795d9c19e7a07217c586d4020f6ced102aceac46aaa8'
Get-NativeSource 'lame-3.100.tar.gz' 'https://deb.debian.org/debian/pool/main/l/lame/lame_3.100.orig.tar.gz' 'https://downloads.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz' '0844b9eadb4aacf8000444621451277de365041cc1d97b7f7a589da0b7a23899310afd4e4d81114b9912aa97832621d20588034715573d417b2923948c08634b'
Push-Location "$WorkDir/soxr-0.1.3-Source"
try {
    foreach ($patch in (Get-ChildItem $SoxrPort -Filter '*.patch' | Sort-Object Name)) {
        git apply --ignore-whitespace $patch.FullName
        Copy-Item -LiteralPath $patch.FullName -Destination $sources
    }
    cmake -S . -B "$WorkDir/soxr-build" -G 'NMake Makefiles' "-DCMAKE_INSTALL_PREFIX=$prefix" `
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF `
        -DBUILD_SHARED_LIBS=ON -DBUILD_SHARED_RUNTIME=ON -DWITH_OPENMP=OFF -DWITH_LSR_BINDINGS=OFF `
        -DCMAKE_DISABLE_FIND_PACKAGE_LibAVUtil=TRUE
    cmake --build "$WorkDir/soxr-build" --target install
} finally { Pop-Location }
Push-Location "$WorkDir/lame-3.100"
try {
    nmake /nologo /f Makefile.MSVC MSVCVER=Win64 ASM=NO MACHINE=/machine:x64 `
        'CC_OPTS=/nologo /O2 /MD /DWin64 /D_CRT_SECURE_NO_WARNINGS' libmp3lame.dll
    Copy-Item output/libmp3lame.dll "$prefix/bin/"
    Copy-Item output/libmp3lame.lib "$prefix/lib/mp3lame.lib"
} finally { Pop-Location }
