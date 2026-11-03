"""Regression test: closing the Analog Demo while a trainer is still running
must not raise 'bad window path name' from the completion callback."""

import faulthandler
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

from analog_demo import AnalogDemoWindow

faulthandler.enable()
messagebox.showinfo = lambda *a, **k: None
messagebox.showerror = lambda *a, **k: None

import traceback

errors = []
_orig_hook = threading.excepthook


def _thread_hook(a):
    errors.append("thread: " + "".join(traceback.format_exception(
        a.exc_type, a.exc_value, a.exc_traceback)))


threading.excepthook = _thread_hook


def report_callback_exception(exc, val, tb):
    errors.append(f"tk callback: {val!r}")


root = tk.Tk()
root.withdraw()
root.report_callback_exception = report_callback_exception
st = ttk.Style(root)
for n, f in (("Title.TLabel", ("Segoe UI Semibold", 14)),
             ("Head.TLabel", ("Segoe UI Semibold", 11)),
             ("Big.TLabel", ("Consolas", 18, "bold")),
             ("Tip.TLabel", ("Segoe UI", 9))):
    st.configure(n, font=f)
st.configure("Accent.TButton", font=("Segoe UI Semibold", 10))

win = AnalogDemoWindow(root)
win.report_callback_exception = report_callback_exception
root.update()

# --- case 1: close while the Pythagoras trainer is mid-run -----------------
win.nb.select(1)
win.engine.set("mazumdar")
win.py_epochs.set(60)
win._py_make_shape()
root.update()
win._py_start_train()
t0 = time.time()
while time.time() - t0 < 2.0:
    root.update()
    time.sleep(0.05)
win._on_close()                      # user closes the window mid-training
n_threads_before = threading.active_count()
for _ in range(int(90 / 0.05)):      # let the orphaned trainer run to the end
    try:
        root.update()
    except tk.TclError as e:
        errors.append(f"root.update: {e!r}")
        break
    if threading.active_count() < n_threads_before:
        break
    time.sleep(0.05)
print("case 1: closed during Pythagoras training; orphaned trainer ran to "
      "completion and its callbacks were dropped silently")

# --- case 2: close while the live BMP worker is streaming ------------------
win2 = AnalogDemoWindow(root)
win2.report_callback_exception = report_callback_exception
root.update()
win2.nb.select(2)
win2.bmp_batch.set(128)
win2._bmp_load_shape()
root.update()
win2._bmp_toggle_run()
t0 = time.time()
while time.time() - t0 < 3.0:
    root.update()
    time.sleep(0.05)
probes = win2.bmp_seen.get()
win2._on_close()
time.sleep(2.0)
root.update()
print(f"case 2: closed during live streaming at {probes} probes "
      f"(worker retired={not win2._bmp_run})")

root.destroy()
threading.excepthook = _orig_hook

if errors:
    print("\nFAILED — exceptions escaped:")
    for e in errors:
        print("  ", e)
    sys.exit(1)
print("\nNO EXCEPTIONS — close-during-training regression fixed")
