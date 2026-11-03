@echo off
set LOG=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\results\restore_com_log.txt
echo ==== %DATE% %TIME% RESTORE COM ==== > %LOG%

echo --- enable all 0403:6010 nodes --- >> %LOG%
powershell -NoProfile -Command ^
  "Get-PnpDevice | Where-Object { $_.InstanceId -like '*VID_0403&PID_6010*' } | ForEach-Object { Write-Output ($_.Status + ' ' + $_.FriendlyName + ' ' + $_.InstanceId); Enable-PnpDevice -InstanceId $_.InstanceId -Confirm:$false -ErrorAction SilentlyContinue }" >> %LOG% 2>&1

echo --- pnputil scan --- >> %LOG%
pnputil /scan-devices >> %LOG% 2>&1
timeout /t 3 /nobreak >nul

echo --- enable again after scan --- >> %LOG%
powershell -NoProfile -Command ^
  "Get-PnpDevice | Where-Object { $_.InstanceId -like '*VID_0403&PID_6010*' } | ForEach-Object { Enable-PnpDevice -InstanceId $_.InstanceId -Confirm:$false -ErrorAction SilentlyContinue; Write-Output ($_.Status + ' ' + $_.FriendlyName) }" >> %LOG% 2>&1

echo --- restart USB hubs --- >> %LOG%
powershell -NoProfile -Command ^
  "Get-PnpDevice | Where-Object { $_.FriendlyName -match 'USB Root Hub|Generic USB Hub' -and $_.Status -eq 'OK' } | Select-Object -First 8 | ForEach-Object { Disable-PnpDevice -InstanceId $_.InstanceId -Confirm:$false -EA SilentlyContinue; Start-Sleep 1; Enable-PnpDevice -InstanceId $_.InstanceId -Confirm:$false -EA SilentlyContinue; Write-Output $_.FriendlyName }" >> %LOG% 2>&1

timeout /t 4 /nobreak >nul
pnputil /scan-devices >> %LOG% 2>&1
timeout /t 3 /nobreak >nul

powershell -NoProfile -Command ^
  "Get-PnpDevice | Where-Object { $_.InstanceId -like '*0403*6010*' -or $_.FriendlyName -match 'COM1[01]|COM[34]|Serial Converter|FTDI' } | Format-Table Status,Class,FriendlyName -AutoSize | Out-String -Width 200" >> %LOG%

echo DONE >> %LOG%
