@echo off
setlocal
set PROG=D:\_July2026\SENA_Papilio_FPGA\PapillioBlink\tools\papilio-prog\papilio-prog.exe
set BIT=D:\_July2026\SENA_Project\comfort_poc\ise\comfort_sena_top.bit
set HOST=D:\_July2026\SENA_Project\comfort_poc\host\papilio_comfort.py
set PORT=%~1
if "%PORT%"=="" set PORT=COM11
if not exist "%BIT%" (echo missing bitfile & exit /b 1)
"%PROG%" -v -f "%BIT%"
if errorlevel 1 exit /b 1
timeout /t 2 /nobreak >nul
python "%HOST%" --port %PORT% --t-passes 100 --ensemble --limit 40
endlocal
