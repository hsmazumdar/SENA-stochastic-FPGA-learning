@echo off
setlocal
set PKGWIN=D:\_July2026\SENA_Project\sena2026\fpga_papilio_mnist_online
echo Building papilio_mnist_online_top.bit via Docker ISE 14.7 ...
docker run --rm -v "%PKGWIN%:/work" -w /work/ise daverichmond/xilinx-ise:14.7 xtclsh build_ise.tcl
if errorlevel 1 (
  echo BUILD FAILED
  exit /b 1
)
dir "%PKGWIN%\ise\*.bit"
endlocal
