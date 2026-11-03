@echo off
REM After PHYSICAL unplug/replug of Papilio on rear USB 2.0, run this.
setlocal
set PROG=D:\_July2026\SENA_Papilio_FPGA\PapillioBlink\tools\papilio-prog\papilio-prog.exe
set BIT=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\ise\papilio_logic_learn_top.bit
set LOG=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\results\post_replug_log.txt
set HOST=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\host\papilio_progress_30s.py

echo ==== %DATE% %TIME% ==== > "%LOG%"
echo Enumerating COM... >> "%LOG%"
powershell -NoProfile -Command "python -c \"import serial.tools.list_ports as L; print([ (p.device,p.description) for p in L.comports() ])\"" >> "%LOG%"

echo === JTAG ===
"%PROG%" -j
if errorlevel 1 (
  echo JTAG FAIL - is board plugged into rear USB 2.0?
  echo jtag=FAIL >> "%LOG%"
  type "%LOG%"
  exit /b 1
)

echo === PROGRAM ===
"%PROG%" -v -f "%BIT%"
if errorlevel 1 exit /b 1
timeout /t 2 /nobreak >nul

echo === Find UART COM ===
for %%P in (COM11 COM10 COM4 COM3) do (
  python -c "import serial; s=serial.Serial('%%P',115200,timeout=0.5); s.write(b'V'); r=s.read(2); print(r.hex()); s.close()" 2>nul && (
    echo Using %%P
    python "%HOST%" --port %%P --go
    exit /b 0
  )
)
echo No UART replied with identity — try COM manually
type "%LOG%"
endlocal
