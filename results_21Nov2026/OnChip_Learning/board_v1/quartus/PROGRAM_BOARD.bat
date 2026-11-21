@echo off
REM Program the DE2-115 over USB-Blaster (JTAG, volatile). Board: SW19 (RUN/PROG) in RUN position.
set QUARTUS=E:\FPGA_Tools_115\IntelFPGA\25.1\quartus\bin64
cd /d %~dp0
"%QUARTUS%\quartus_pgm.exe" -l
"%QUARTUS%\quartus_pgm.exe" -c "USB-Blaster" -m JTAG -o "p;output_files\de2_115_sena.sof@1"
