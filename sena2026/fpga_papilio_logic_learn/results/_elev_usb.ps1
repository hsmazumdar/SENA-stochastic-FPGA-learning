$dev = Get-PnpDevice | Where-Object { $_.InstanceId -eq "USB\VID_0403&PID_6010\7&19CD6259&0&1" }
if ($dev) { Disable-PnpDevice -InstanceId $dev.InstanceId -Confirm:$false; Start-Sleep 2; Enable-PnpDevice -InstanceId $dev.InstanceId -Confirm:$false }
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
