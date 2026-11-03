@echo off
set LOG=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\results\program_log.txt
set PROG=D:\_July2026\SENA_Papilio_FPGA\PapillioBlink\tools\papilio-prog\papilio-prog.exe
set BIT=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn\ise\papilio_logic_learn_top.bit
echo ==== %DATE% %TIME% ==== > %LOG%
echo Killing zombie python... >> %LOG%
taskkill /F /PID 10508 /T >> %LOG% 2>&1
taskkill /F /PID 4048 /T >> %LOG% 2>&1
taskkill /F /IM python.exe /FI "PID eq 10508" >> %LOG% 2>&1
echo --- jtag --- >> %LOG%
"%PROG%" -j >> %LOG% 2>&1
echo jtag=%ERRORLEVEL% >> %LOG%
echo --- program --- >> %LOG%
"%PROG%" -v -f "%BIT%" >> %LOG% 2>&1
echo prog=%ERRORLEVEL% >> %LOG%
echo --- usb cycle --- >> %LOG%
powershell -NoProfile -Command "Disable-PnpDevice -InstanceId 'USB\VID_0403&PID_6010\7&19CD6259&0&1' -Confirm:$false -EA SilentlyContinue; Start-Sleep 2; Enable-PnpDevice -InstanceId 'USB\VID_0403&PID_6010\7&19CD6259&0&1' -Confirm:$false -EA SilentlyContinue" >> %LOG% 2>&1
echo DONE >> %LOG%
