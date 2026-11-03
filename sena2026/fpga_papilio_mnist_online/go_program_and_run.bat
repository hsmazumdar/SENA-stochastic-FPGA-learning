@echo off
setlocal
set PROG=D:\_July2026\SENA_Papilio_FPGA\PapillioBlink\tools\papilio-prog\papilio-prog.exe
set BIT=D:\_July2026\SENA_Project\sena2026\fpga_papilio_mnist_online\ise\papilio_mnist_online_top.bit
set HOST=D:\_July2026\SENA_Project\sena2026\fpga_papilio_mnist_online\host\papilio_mnist_online.py
set PORT=%~1
if "%PORT%"=="" set PORT=COM11
if not exist "%BIT%" (
  echo Missing bitfile — run ise\build_in_docker.bat first
  exit /b 1
)
"%PROG%" -v -f "%BIT%"
if errorlevel 1 exit /b 1
timeout /t 2 /nobreak >nul
python "%HOST%" --port %PORT% --train 200 --test 100 --epochs 2
endlocal
