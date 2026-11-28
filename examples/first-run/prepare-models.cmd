@echo off
setlocal
pushd "%~dp0" || exit /b 1
set "converter=%~dp0gpt-sovits-convert.exe"
if not exist "%converter%" set "converter=%~dp0cuda-sm89\gpt-sovits-convert.exe"
"%converter%" prepare-demo %*
set "result=%errorlevel%"
popd
exit /b %result%
