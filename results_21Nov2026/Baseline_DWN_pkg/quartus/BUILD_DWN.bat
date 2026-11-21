@echo off
REM Build both revisions and (optionally) program the DE2-115.
REM   BUILD_DWN.bat            -> compile + reports for de2_115_dwn and dwn_core_only
REM   BUILD_DWN.bat program    -> same, then program de2_115_dwn.sof via USB-Blaster
REM Requires gen\ (run dwn_to_verilog.py first). Edit QUARTUS if installed elsewhere.
set QUARTUS=E:\FPGA_Tools_115\IntelFPGA\25.1\quartus\bin64
cd /d "%~dp0"
if not exist gen\dwn_core.v (echo gen\dwn_core.v missing - run dwn_to_verilog.py first & exit /b 1)
set LOG=%~dp0BUILD_DWN_LOG.txt
echo === %DATE% %TIME% DWN baseline build === >> "%LOG%"
for %%R in (de2_115_dwn dwn_core_only) do (
  echo [compile] %%R ^(several minutes^)...
  "%QUARTUS%\quartus_sh.exe" -t compile_and_report.tcl %%R >> "%LOG%" 2>&1
  if errorlevel 1 (echo COMPILE FAILED for %%R - see %LOG% & exit /b 1)
)
if /I "%1"=="program" (
  "%QUARTUS%\jtagconfig.exe" >> "%LOG%" 2>&1
  "%QUARTUS%\quartus_pgm.exe" -c "USB-Blaster [USB-0]" -m jtag -o "p;output_files\de2_115_dwn.sof" >> "%LOG%" 2>&1
  if errorlevel 1 "%QUARTUS%\quartus_pgm.exe" -c USB-Blaster -m jtag -o "p;output_files\de2_115_dwn.sof" >> "%LOG%" 2>&1
  if errorlevel 1 (echo PROGRAM FAILED - see %LOG% & exit /b 1)
  echo Programmed. After ~1 ms: LEDG[8:1] ON = PASS ^(0 mismatches vs Python^). KEY2 cycles HEX modes.
)
echo DONE. Reports in quartus\reports\
exit /b 0
