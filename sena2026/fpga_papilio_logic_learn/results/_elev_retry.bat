@echo off
set LOG=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\results\retry_program_log.txt
set PROG=D:\_July2026\SENA_Papilio_FPGA\PapillioBlink\tools\papilio-prog\papilio-prog.exe
set BIT=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\ise\papilio_logic_learn_top.bit
echo ==== %DATE% %TIME% RETRY ==== > %LOG%
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *" >> %LOG% 2>&1
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| find "PID:"') do taskkill /F /PID %%a >> %LOG% 2>&1
echo --- disable USB --- >> %LOG%
powershell -NoProfile -Command "Get-PnpDevice | Where-Object { $_.InstanceId -like 'USB\VID_0403&PID_6010\*' -and $_.InstanceId -notlike '*MI_*' } | ForEach-Object { Disable-PnpDevice -InstanceId $_.InstanceId -Confirm:$false -EA SilentlyContinue; Write-Output $_.InstanceId }" >> %LOG% 2>&1
timeout /t 3 /nobreak >nul
echo --- enable USB --- >> %LOG%
powershell -NoProfile -Command "Get-PnpDevice | Where-Object { $_.InstanceId -like 'USB\VID_0403&PID_6010\*' -and $_.InstanceId -notlike '*MI_*' } | ForEach-Object { Enable-PnpDevice -InstanceId $_.InstanceId -Confirm:$false -EA SilentlyContinue; Write-Output $_.InstanceId }" >> %LOG% 2>&1
timeout /t 4 /nobreak >nul
echo --- jtag --- >> %LOG%
"%PROG%" -j >> %LOG% 2>&1
echo jtag=%ERRORLEVEL% >> %LOG%
if %ERRORLEVEL%==0 (
  echo --- program --- >> %LOG%
  "%PROG%" -v -f "%BIT%" >> %LOG% 2>&1
  echo prog=%ERRORLEVEL% >> %LOG%
) else (
  echo SKIP program >> %LOG%
)
echo DONE >> %LOG%
