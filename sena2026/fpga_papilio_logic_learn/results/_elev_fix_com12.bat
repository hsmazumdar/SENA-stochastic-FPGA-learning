@echo off
set LOG=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\results\fix_com12_log.txt
set PROGDIR=D:\_July2026\SENA_Papilio_FPGA\PapillioBlink\tools\papilio-prog
set PROG=%PROGDIR%\papilio-prog.exe
set BIT=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\ise\papilio_logic_learn_top.bit
set FILT=D:\_July2026\SENA_Papilio_FPGA\_tools_install\libusb140\libusb-win32-bin-1.4.0.2\bin\amd64\install-filter.exe
echo ==== %DATE% %TIME% FIX COM12 ==== > %LOG%

echo --- remove ghost Unknown FTDI nodes only --- >> %LOG%
powershell -NoProfile -Command ^
  "Get-PnpDevice | Where-Object { $_.InstanceId -like '*0403*6010*' -and $_.Status -eq 'Unknown' } | ForEach-Object { Write-Output ('REMOVE '+$_.FriendlyName+' '+$_.InstanceId); pnputil /remove-device $_.InstanceId }" >> %LOG% 2>&1

echo --- restart Error COM ports --- >> %LOG%
powershell -NoProfile -Command ^
  "Get-PnpDevice | Where-Object { $_.FriendlyName -match 'COM12|COM13' } | ForEach-Object { Write-Output ('RESTART '+$_.InstanceId); pnputil /restart-device $_.InstanceId }" >> %LOG% 2>&1

echo --- restart OK converters / composite --- >> %LOG%
powershell -NoProfile -Command ^
  "Get-PnpDevice | Where-Object { $_.InstanceId -like '*0403*6010*' -and $_.Status -eq 'OK' } | ForEach-Object { Write-Output ('RESTART_OK '+$_.FriendlyName); pnputil /restart-device $_.InstanceId }" >> %LOG% 2>&1

timeout /t 3 /nobreak >nul
pnputil /scan-devices >> %LOG% 2>&1
timeout /t 3 /nobreak >nul

echo --- install libusb filter (JTAG) --- >> %LOG%
"%FILT%" install --device=USB\VID_0403^&PID_6010 >> %LOG% 2>&1
echo filter=%ERRORLEVEL% >> %LOG%

echo --- jtag --- >> %LOG%
cd /d "%PROGDIR%"
papilio-prog.exe -j >> %LOG% 2>&1
echo jtag=%ERRORLEVEL% >> %LOG%
if %ERRORLEVEL%==0 (
  echo --- program logic_learn --- >> %LOG%
  papilio-prog.exe -v -f "%BIT%" >> %LOG% 2>&1
  echo prog=%ERRORLEVEL% >> %LOG%
)

echo --- final devices --- >> %LOG%
powershell -NoProfile -Command "Get-PnpDevice | Where-Object { $_.InstanceId -like '*0403*6010*' -or $_.FriendlyName -match 'COM1[0-3]|COM[34]' } | Format-Table Status,FriendlyName -AutoSize | Out-String -Width 120" >> %LOG%
echo DONE >> %LOG%
