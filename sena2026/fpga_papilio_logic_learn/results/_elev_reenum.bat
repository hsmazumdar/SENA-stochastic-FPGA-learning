@echo off
set LOG=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\results\retry_reenum_log.txt
echo ==== %DATE% %TIME% REENUM ==== > %LOG%
powershell -NoProfile -Command ^
  "$ids=@('USB\VID_0403&PID_6010\7&19CD6259&0&1','USB\VID_0403&PID_6010\5&1A60D403&0&9'); foreach($id in $ids){ Enable-PnpDevice -InstanceId $id -Confirm:$false -EA SilentlyContinue; Write-Output \"enable $id\" }; Get-PnpDevice | Where-Object { $_.InstanceId -like '*0403*6010*' } | ForEach-Object { if($_.Status -ne 'OK'){ Enable-PnpDevice -InstanceId $_.InstanceId -Confirm:$false -EA SilentlyContinue }; '{0} {1}' -f $_.Status,$_.FriendlyName }" >> %LOG% 2>&1
echo --- pnputil scan --- >> %LOG%
pnputil /scan-devices >> %LOG% 2>&1
timeout /t 5 /nobreak >nul
powershell -NoProfile -Command "Get-PnpDevice | Where-Object { $_.FriendlyName -match 'COM|Serial Converter|0403' } | Select-Object Status,FriendlyName | Format-Table -AutoSize | Out-String" >> %LOG%
echo DONE >> %LOG%
