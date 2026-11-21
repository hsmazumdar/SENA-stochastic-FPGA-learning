@echo off
REM Synthesis-only measurement of sena_learn_core on EP4CE115 (Quartus Lite 25.1).
set QUARTUS=E:\FPGA_Tools_115\IntelFPGA\25.1\quartus\bin64
cd /d %~dp0
"%QUARTUS%\quartus_sh.exe" --flow compile sena_core > build_log.txt 2>&1
echo EXIT %ERRORLEVEL% >> build_log.txt
