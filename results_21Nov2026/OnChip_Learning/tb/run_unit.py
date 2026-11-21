"""run_unit.py -- builds unit_w*.hex, runs tb_unit_latency.v in Icarus, checks LFSR draw vs twin."""
import os, sys, subprocess, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'twin')); import sena_hw_twin as tw
RTL = os.environ.get('RTL_DIR', os.path.join(ROOT, 'rtl'))
wd = os.path.join(ROOT, 'vectors', 'unit_latency' + os.environ.get('TAG', '')); os.makedirs(wd, exist_ok=True)
def row(vals): return ''.join('%02x' % (v & 255) for v in reversed(vals))
open(f'{wd}/unit_w1.hex', 'w').write('\n'.join(row([i*8 + j + 1 for j in range(4)]) for i in range(8)) + '\n' + row([90 + j for j in range(4)]) + '\n')
open(f'{wd}/unit_w2.hex', 'w').write('\n'.join('00' * 10 for _ in range(5)) + '\n')
env = dict(os.environ, PATH='/workspace/eda/bin:' + os.environ['PATH'])
subprocess.run(['iverilog', '-g2012', '-o', f'{wd}/u.vvp', f'{HERE}/tb_unit_latency.v', f'{RTL}/sdp_ram.v', f'{RTL}/sena_learn_core.v'], check=True, env=env)
r = subprocess.run(['vvp', '-n', 'u.vvp'], cwd=wd, capture_output=True, text=True, env=env)
out = r.stdout + r.stderr; print(out)
exp = [int(v) for v in tw.leap(tw.lane_seeds(0x1D872B41, 4, tw.C_H_MUL))]
got = None
for ln in out.splitlines():
    if ln.startswith('LFSR_FIRST_DRAW'): got = [int(v, 16) for v in ln.split()[1:]]
res = dict(latency_pass='UNIT_LATENCY PASS' in out, lfsr_draw_rtl=[hex(v) for v in got or []], lfsr_draw_twin=[hex(v) for v in exp], lfsr_match=(got == exp))
json.dump(res, open(f'{wd}/result.json', 'w'), indent=1); print(res)
