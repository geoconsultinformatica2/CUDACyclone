@echo off
setlocal EnableExtensions

rem Bitcoin Puzzle 71 launcher for CUDACyclone-5090.
rem PowerShell: .\run-puzzle71.cmd
rem CMD:        run-puzzle71.cmd
rem Additional arguments are forwarded to CUDACyclone-5090.

pushd "%~dp0" >nul || (
    echo Error: cannot enter the CUDACyclone directory.
    exit /b 1
)

if not exist "CUDACyclone-5090" (
    echo Error: CUDACyclone-5090 was not found in:
    echo   %CD%
    echo Build it first with: make rtx5090
    popd
    exit /b 1
)

echo Bitcoin Puzzle 71
echo Range      : 400000000000000000:7fffffffffffffffff
echo Address    : 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
echo Grid       : 512,8
echo Slices     : 16
echo Block bits : 37
echo Checkpoint : cudacyclone-p71.checkpoint
echo Press Ctrl+C to stop safely; run this script again to resume.
echo.

wsl.exe --cd "%CD%" ./CUDACyclone-5090 ^
    --range 400000000000000000:7fffffffffffffffff ^
    --address 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU ^
    --grid 512,8 ^
    --slices 16 ^
    --random-blocks ^
    --checkpoint cudacyclone-p71.checkpoint ^
    --block-bits 37 %*

set "CUDACYCLONE_EXIT=%ERRORLEVEL%"
popd

if not "%CUDACYCLONE_EXIT%"=="0" (
    echo.
    echo CUDACyclone exited with code %CUDACYCLONE_EXIT%.
)

exit /b %CUDACYCLONE_EXIT%
