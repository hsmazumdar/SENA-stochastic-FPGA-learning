@echo off
REM Program the Basys 3 over Digilent USB-JTAG (volatile). Set VIVADO_PATH if needed.
setlocal
cd /d %~dp0
if not defined VIVADO_PATH (
  for %%V in (2025.2 2025.1 2024.2 2024.1 2023.2 2023.1 2022.2) do (
    if exist "C:\Xilinx\Vivado\%%V\bin\vivado.bat" set VIVADO_PATH=C:\Xilinx\Vivado\%%V\bin\vivado.bat
  )
)
if not defined VIVADO_PATH (
  echo Vivado not found. Set VIVADO_PATH to vivado.bat and retry.
  exit /b 1
)
echo Using %VIVADO_PATH%
"%VIVADO_PATH%" -mode batch -source program_basys3.tcl
