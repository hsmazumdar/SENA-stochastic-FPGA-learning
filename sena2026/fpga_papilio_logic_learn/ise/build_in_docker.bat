@echo off
set PKG=%pkg:\=/%
REM Windows path for -v
set PKGWIN=D:\_July2026\SENA_Project\sena2026\fpga_papilio_logic_learn
docker run --rm -v "%PKGWIN%:/work" -w /work/ise daverichmond/xilinx-ise:14.7 bash -lc "source /opt/Xilinx/14.7/ISE_DS/settings64.sh >/dev/null 2>&1 || source $(ls -d /opt/Xilinx/*/ISE_DS)/settings64.sh; xtclsh build_ise.tcl"
