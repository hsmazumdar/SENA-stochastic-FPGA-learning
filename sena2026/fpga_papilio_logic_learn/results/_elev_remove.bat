@echo off
set LOG=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\results\retry_remove_scan.txt
echo ==== %DATE% %TIME% ==== > %LOG%
REM Remove problem FTDI nodes then rescan (forces fresh enumerate on next plug)
powershell -NoProfile -Command ^
  "Get-PnpDevice | Where-Object { $_.InstanceId -like '*VID_0403&PID_6010*' } | ForEach-Object { Write-Output ('REMOVE '+$_.Status+' '+$_.InstanceId); try { & pnputil /remove-device $_.InstanceId } catch {}; Remove-PnpDevice -InstanceId $_.InstanceId -Confirm:$false -EA SilentlyContinue }" >> %LOG% 2>&1
pnputil /scan-devices >> %LOG% 2>&1
timeout /t 3 /nobreak >nul
powershell -NoProfile -Command "Get-PnpDevice | Where-Object { $_.InstanceId -like '*0403*6010*' -or $_.FriendlyName -match 'COM1[01]|Serial Converter' } | Format-Table Status,FriendlyName -AutoSize | Out-String" >> %LOG%
echo DONE >> %LOG%
