#!/bin/bash
set -e
cd /work/ise
if [ -f /opt/Xilinx/14.7/ISE_DS/settings64.sh ]; then
  source /opt/Xilinx/14.7/ISE_DS/settings64.sh
else
  source "$(ls -d /opt/Xilinx/*/ISE_DS | head -1)/settings64.sh"
fi
xtclsh build_ise.tcl
echo BUILD_DONE
ls -la *.bit 2>/dev/null || true
find . -name '*.bit' -o -name '*.syr' -o -name '*.log' | head -40
