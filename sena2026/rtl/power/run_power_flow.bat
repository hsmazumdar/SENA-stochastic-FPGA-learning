@echo off
REM One-click Artix-7 SAIF power flow (EnergyIssueFix.docx)
setlocal
cd /d "%~dp0"

if not exist stimuli\mnist_x.bin (
  echo Preparing MNIST + weight stimuli...
  python prepare_power_stimuli.py --n-images 1000 --quick-train
  if errorlevel 1 exit /b 1
)

where vivado >nul 2>&1
if errorlevel 1 (
  echo ERROR: vivado not on PATH. Install/open Vivado settings64.bat first.
  exit /b 1
)

echo Running Vivado SAIF power flow...
vivado -mode batch -source power_analysis_full.tcl -tclargs STAGE=all CLK_MHZ=50 N_IMAGES=500 MEAN_PASSES=2.30 M=3
if errorlevel 1 exit /b 1

python ..\..\patch_table_vi_vivado.py
echo Done. Check sena_power\power.json and updated paper Table VI.
endlocal
