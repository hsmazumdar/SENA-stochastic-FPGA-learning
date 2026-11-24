@echo off
REM SENA V13d - regenerate the manuscript tables and Fig. 2 from saved results.
REM No FPGA board, training or GPU needed. Output: reproduce_output\run_<timestamp>\
cd /d "%~dp0"
where py >nul 2>nul && (py -3 reproduce_tables.py) || (python reproduce_tables.py)
set RC=%ERRORLEVEL%
if %RC%==0 (echo All checked values match the manuscript.) else (echo Some values did not match - see check.csv in the output folder.)
pause
exit /b %RC%
