@echo off
REM Program Papilio with 3xNN logic-learn bit + start 30s PC logger
setlocal
set PROG=D:\_July2026\SENA_Papilio_FPGA\PapillioBlink\tools\papilio-prog\papilio-prog.exe
set BIT=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\ise\papilio_logic_learn_top.bit
set HOST=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\host\papilio_progress_30s.py
set PORT=%~1
if "%PORT%"=="" set PORT=COM11

echo === JTAG scan ===
"%PROG%" -j
if errorlevel 1 (
  echo ERROR: papilio-prog cannot open FTDI JTAG.
  echo Fix: unplug Papilio, reboot if COM is stuck Access Denied, rear USB 2.0, then retry.
  echo      Do NOT stack WinUSB on MI_00 if this PC BSODed on libusb — use known-good FTDI path from PAPER_BRING_BACK.
  exit /b 1
)

echo === Program %BIT% ===
"%PROG%" -v -f "%BIT%"
if errorlevel 1 exit /b 1
timeout /t 2 /nobreak >nul

echo === Host 30s logger on %PORT% ===
python "%HOST%" --port %PORT% --go
endlocal
