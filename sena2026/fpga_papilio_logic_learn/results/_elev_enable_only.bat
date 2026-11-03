@echo off
set LOG=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\results\enable_only_log.txt
echo ==== %DATE% %TIME% ENABLE ONLY ==== > %LOG%
powershell -NoProfile -Command ^
  "Get-PnpDevice | Where-Object { $_.InstanceId -like '*VID_0403&PID_6010*' } | ForEach-Object { Enable-PnpDevice -InstanceId $_.InstanceId -Confirm:$false -EA SilentlyContinue; Start-Sleep -Milliseconds 200; $d=Get-PnpDevice -InstanceId $_.InstanceId -EA SilentlyContinue; Write-Output ($d.Status + ' | ' + $d.FriendlyName) }" >> %LOG% 2>&1
pnputil /scan-devices >> %LOG% 2>&1
timeout /t 2 /nobreak >nul
powershell -NoProfile -Command ^
  "Get-PnpDevice | Where-Object { $_.InstanceId -like '*VID_0403&PID_6010*' } | ForEach-Object { pnputil /restart-device $_.InstanceId 2>&1 | Out-String | Write-Output }" >> %LOG% 2>&1
timeout /t 3 /nobreak >nul
powershell -NoProfile -Command "Get-PnpDevice | Where-Object { $_.InstanceId -like '*0403*6010*' } | Format-Table Status,FriendlyName -AutoSize | Out-String" >> %LOG%
echo DONE >> %LOG%
