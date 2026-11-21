@echo off
REM Non-project Vivado synth/impl/bitstream. Default H_LANES=32 (serial hidden lanes).
REM Override: set H_LANES=160  (or 0) for the full-parallel core, then rerun.
setlocal
cd /d %~dp0
if not defined H_LANES set H_LANES=32
if not defined VIVADO_PATH (
  for %%V in (2025.2 2025.1 2024.2 2024.1 2023.2 2023.1 2022.2) do (
    if exist "C:\Xilinx\Vivado\%%V\bin\vivado.bat" set VIVADO_PATH=C:\Xilinx\Vivado\%%V\bin\vivado.bat
  )
)
if not defined VIVADO_PATH (
  echo Vivado not found. Set VIVADO_PATH to vivado.bat and retry.
  exit /b 1
)
echo Using %VIVADO_PATH%  H_LANES=%H_LANES%
echo set H_LANES %H_LANES%> _run.tcl
echo source build_basys3.tcl>> _run.tcl
"%VIVADO_PATH%" -mode batch -source _run.tcl
