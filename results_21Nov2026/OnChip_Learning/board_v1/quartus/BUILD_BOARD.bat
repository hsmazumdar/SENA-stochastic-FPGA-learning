@echo off
REM Full compile of the SENA DE2-115 board top (Quartus Prime Lite 25.1) + PowerPlay power estimate.
REM Output: output_files\de2_115_sena.sof and *.rpt ; log in build_log.txt
set QUARTUS=E:\FPGA_Tools_115\IntelFPGA\25.1\quartus\bin64
cd /d %~dp0
"%QUARTUS%\quartus_sh.exe" --flow compile de2_115_sena > build_log.txt 2>&1
echo COMPILE_EXIT %ERRORLEVEL% >> build_log.txt
"%QUARTUS%\quartus_pow.exe" de2_115_sena --default_toggle_rate=12.5%% >> build_log.txt 2>&1
echo POWER_EXIT %ERRORLEVEL% >> build_log.txt
