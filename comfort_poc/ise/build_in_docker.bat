@echo off
setlocal
set PKG=D:\_July2026\SENA_Project\comfort_poc
docker run --rm -v "%PKG%:/work" -w /work/ise daverichmond/xilinx-ise:14.7 xtclsh build_ise.tcl
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
dir "%PKG%\ise\*.bit"
endlocal
