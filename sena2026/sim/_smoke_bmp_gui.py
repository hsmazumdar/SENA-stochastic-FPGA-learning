"""Headless-ish smoke test of the BMP live-learn tab (creates a real Tk window,
exercises every callback, then closes). Run: python _smoke_bmp_gui.py"""

import time
import tkinter as tk
from tkinter import ttk, messagebox

import analog_demo
from analog_demo import AnalogDemoWindow

# swallow modal dialogs so the test can run unattended
messagebox.showinfo = lambda *a, **k: None
messagebox.showerror = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError(f"showerror: {a}"))

root = tk.Tk()
root.withdraw()
st = ttk.Style(root)
for name, font in (("Title.TLabel", ("Segoe UI Semibold", 14)),
                   ("Head.TLabel", ("Segoe UI Semibold", 11)),
                   ("Big.TLabel", ("Consolas", 18, "bold")),
                   ("Tip.TLabel", ("Segoe UI", 9))):
    st.configure(name, font=font)
st.configure("Accent.TButton", font=("Segoe UI Semibold", 10))

win = AnalogDemoWindow(root)
root.update()
win.nb.select(2)
root.update()

checks = []
for shape in ("circle", "square", "triangle"):
    win._select_shape(shape)
    root.update()
    checks.append(f"load {shape}: {win.bmp_status.get()}")

for engine in ("mazumdar", "mlp"):
    win.bmp_engine.set(engine)
    win._bmp_reset()
    root.update()
    win.bmp_batch.set(128)
    for _ in range(3):
        win._bmp_step_once()
        root.update()
    checks.append(f"{engine:9s} 3 train steps -> acc={win.bmp_acc.get()} "
                  f"probes={win.bmp_seen.get()}")
    win.bmp_mode.set("test")
    win._bmp_step_once()
    root.update()
    checks.append(f"{engine:9s} frozen test step -> acc={win.bmp_acc.get()}")
    win.bmp_mode.set("train")

# continuous run through the real worker thread + UI tick
win.bmp_engine.set("mazumdar")
win.bmp_batch.set(256)          # the shipped default
win._bmp_reset()
root.update()
win._bmp_toggle_run()
t0 = time.time()
while time.time() - t0 < 8.0:
    root.update()
    time.sleep(0.05)
probes = int(win.bmp_seen.get().replace(",", ""))
assert win._bmp_error is None, f"worker raised: {win._bmp_error}"
assert win._bmp_run, "worker died during continuous run"
assert probes >= 2000, f"probe rate too low for a live demo: {probes} in 8 s"
checks.append(f"continuous 8 s -> probes={probes} ({probes/8:.0f}/s) "
              f"acc={win.bmp_acc.get()}")

win._bmp_toggle_run()
root.update()
assert not win._bmp_run
assert "Run" in win._bmp_run_btn.cget("text")
after_stop = int(win.bmp_seen.get().replace(",", ""))
time.sleep(1.0)
root.update()
assert int(win.bmp_seen.get().replace(",", "")) == after_stop, \
    "worker kept running after stop"
checks.append(f"after stop -> frozen at {after_stop} probes, "
              f"btn={win._bmp_run_btn.cget('text').strip()}")

# a reset while running must retire the old worker, not double it up
win._bmp_toggle_run()
time.sleep(0.5)
root.update()
win._bmp_reset()
root.update()
time.sleep(2.0)
root.update()
assert win._bmp_error is None, f"reset-while-running raised: {win._bmp_error}"
checks.append(f"reset while running -> ok, probes={win.bmp_seen.get()}, "
              f"running={win._bmp_run}")
win._bmp_stop()

win._on_close()
root.update()
root.destroy()

print("\n".join(checks))
print("\nSMOKE TEST PASSED")
