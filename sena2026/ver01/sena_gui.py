"""
SENA Interactive GUI (Tkinter)
==============================
Interactive I/O demo for the multiplier-free stochastic neural network
(Mazumdar 1995) and SENA temporal-vote readout.

Features
--------
* Train on Symmetry / XOR-2D / Iris / Cancer / Digits
* Interactive pattern editor (bit toggles or 8x8 digit pad)
* Live stochastic inference with vote bars + confidence
* Adjustable number of temporal votes (T) and early-exit threshold
* Train / Predict / Clear / Random-sample controls
"""

from __future__ import annotations
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_core import StochasticNet, StochasticConfig
from datasets import get_dataset, thermometer_encode, make_xor2d
from analog_demo import AnalogDemoWindow


# --------------------------------------------------------------------------- #
#  Specs per dataset (match digital experiments)
# --------------------------------------------------------------------------- #
SPECS = {
    "symmetry": dict(hid=24, ep=40, T=64, label="Mirror Symmetry (8-bit)"),
    "xor2d":    dict(hid=32, ep=40, T=128, label="Noisy 2-D XOR"),
    "iris":     dict(hid=24, ep=40, T=64, label="Iris (thermometer)"),
    "cancer":   dict(hid=48, ep=30, T=64, label="Breast Cancer"),
    "digits":   dict(hid=96, ep=35, T=64, label="Handwritten Digits 8x8"),
}

CLASS_NAMES = {
    "symmetry": ["Asymmetric", "Symmetric"],
    "xor2d":    ["Class 0", "Class 1"],
    "iris":     ["Setosa", "Versicolor", "Virginica"],
    "cancer":   ["Malignant", "Benign"],
    "digits":   [str(i) for i in range(10)],
}


# --------------------------------------------------------------------------- #
#  Operational Help (reviewer-friendly)
# --------------------------------------------------------------------------- #
HELP_PAGES = {
    "Quick Start": """\
SENA INTERACTIVE DEMO — QUICK START
====================================

Recommended path for reviewers (fastest):

  Step A  Click  Quick Test  (top-right)  — or use the left-panel button.
          • Switches to Iris automatically
          • Trains ~20 epochs (~20–40 seconds)
          • Runs 5 hold-out predictions with early-exit
          • Shows a PASS/FAIL checklist

  Step B  Press  Help (F1)  anytime to reopen this guide.

  Step C  After Quick Test PASSes, try Digits:
          1. Dataset = digits
          2. Train Network
          3. Load Random Test Sample
          4. Predict (Stochastic)
          Watch: Prediction, Confidence, Early-exit, Vote histogram.

Manual 3-click workflow (any dataset):
  1) Train Network
  2) Load Random Test Sample   (or edit Interactive Input)
  3) Predict (Stochastic)

Keyboard:  F1 = Help
""",
    "What to Look For": """\
WHAT A REVIEWER SHOULD VERIFY
=============================

1) Multiplier-free learning works
   • After Train, Test accuracy appears (Iris typically >90%).
   • Digits typically ~85–92% with default settings.
   • Network uses only ADD / SUBTRACT / COMPARE / RNG (no MACs).

2) Stochastic temporal voting (SENA readout)
   • Votes T controls how many stochastic forward passes are accumulated.
   • Vote histogram shows class tallies.
   • Confidence = leading votes / votes-used-so-far.

3) Early-exit saves compute
   • Exit thr θ (default 0.75): stop when confidence >= θ.
   • Metric "Early-exit (SENA)" shows t=exit/T and % compute saved.
   • Easy samples often exit at t=1..8 (large savings).
   • Harder / ambiguous samples use more votes.

4) Interactive I/O is live
   • Symmetry: toggle bits / Make Palindrome, then Predict.
   • Digits: draw on pad OR Load Random Test Sample.
   • Iris / XOR: sliders + presets.
   • Cancer: Load Random Test Sample (30 features).

5) Log panel
   • Timestamped operational trail for screenshots / reproducibility notes.
""",
    "Controls": """\
CONTROL REFERENCE
=================

Dataset
  symmetry  — 8-bit mirror/palindrome task (original 1995-style demo)
  xor2d     — noisy 2-D XOR (non-linear)
  iris      — 3-class flower features (thermometer-coded)  [best for Quick Test]
  cancer    — Wisconsin breast cancer (binary)
  digits    — 8x8 handwritten digits (10-class)

Epochs
  Training passes over the training set. More = slower, usually higher accuracy.

Votes T
  Number of stochastic samples for inference. Higher T = stabler confidence,
  more compute if early-exit does not fire.

Exit thr θ
  Confidence needed to stop early (SENA). Raise θ for safer/slower decisions;
  lower θ for faster/cheaper decisions.

Buttons
  Train Network           — fit the stochastic net (background thread)
  Load Random Test Sample — pull a labeled hold-out example into the I/O panel
  Predict (Stochastic)    — run SENA voting + early-exit on current input
  Clear Input             — reset interactive editors
  Quick Test (Reviewer)   — automated Iris walkthrough + checklist
  Open Operational Help   — this window (also F1)
""",
    "Datasets & Tips": """\
DATASET TIPS
============

Iris (Quick Test default)
  • Fast train, high accuracy — ideal first demo for reviewers.
  • Use class presets (Setosa / Versicolor / Virginica) after training.

Digits
  • Most impressive visual demo.
  • Prefer "Load Random Test Sample" for reliable accuracy (exact encoding).
  • Freehand drawing is approximate (thermometer encoding differs slightly).

Symmetry
  • Click bits to 0/1; "Make Palindrome" forces a symmetric pattern.
  • Hint line shows whether the current pattern is symmetric.

XOR-2D
  • Presets (--),(++),(-+),(+-) map to the four Gaussian centers.
  • Classes: same-sign quadrants vs opposite-sign (classic XOR).

Cancer
  • 30 continuous features — use Load Random Test Sample only.

Troubleshooting
  • "Train first" warning → click Train Network.
  • Digits empty pad → draw OR Load Random Test Sample.
  • Low Digits accuracy → raise Epochs to 50 and Votes T to 128.
  • Training busy → wait; status bar shows progress.
""",
    "Science / Claim Map": """\
SCIENCE ↔ GUI MAP (for paper demos)
===================================

Paper claim                          GUI evidence
-----------                          ------------
Stochastic forward activation        Vote bars change run-to-run if T small;
                                     confidence stabilizes as T grows.

Multiplier-free integer training     Train completes; accuracy vs Real-BP
                                     (see digital.json experiments).

1-bit interlayer communication       Activations are binary 0/1 per neuron
                                     (internal); votes aggregate them.

SENA early-exit / energy-proportional
  computation                        Early-exit metric + Quick Test mean
                                     compute saved %.

Analog random-threshold neuron       Analog Demo window: Vin vs noise,
                                     P(Y=1)≈sigmoid, R(X), Monte-Carlo PVT,
                                     live bitstream, pJ / µm² estimates.

Hardware-friendly RNG                Core supports LFSR (hardware-like) and
                                     numpy RNG (see digital rng_study).

Suggested screenshot set for reviews
  1) Help Quick Start tab
  2) Quick Test PASS dialog
  3) Digits: loaded sample + prediction + vote histogram
  4) Analog Demo: forward/reverse curves + Monte-Carlo PVT
  5) Log lines showing early-exit t=... and compute saved
""",
    "Analog Demo": """\
ANALOG DEMO
===========

Tab 1 — Stochastic Neuron
  Comparator + thermal noise (Eq.3 / Eq.4 dual). See previous help.

Tab 2 — Pythagoras 2-30-1  (this request)
  Network:  2 analog inputs → 30 hidden → 1 analog output (normalized)
  Target:   h = sqrt(a^2 + b^2) ,  h_norm = h/sqrt(2) ∈ [0,1]
  Data:     B/W circle image pixels + random (a,b) pairs
  Check:    reconstructed polar plot ≈ same circle

Quick path
  1) Analog Demo (F2)
  2) Tab "Pythagoras 2-30-1"
  3) Quick Demo (all steps)
  4) Confirm left image (B/W circle) ≈ right reconstructed circle
""",
}


class HelpWindow(tk.Toplevel):
    """Reviewer-friendly operational help with tabbed pages."""

    def __init__(self, master: SenaGUI):
        super().__init__(master)
        self.title("SENA — Operational Help")
        self.geometry("720x560")
        self.minsize(560, 420)
        self.configure(bg="#1a1d23")
        self.transient(master)

        hdr = ttk.Frame(self, padding=(14, 10))
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Operational Help", style="Title.TLabel").pack(side="left")
        ttk.Label(hdr, text="  reviewer guide · F1 to reopen",
                  foreground="#94a3b8").pack(side="left")
        ttk.Button(hdr, text="Run Quick Test",
                   command=lambda: (self.destroy(), master._start_quick_test())
                   ).pack(side="right", padx=4)
        ttk.Button(hdr, text="Close", command=self.destroy).pack(side="right")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        for title, body in HELP_PAGES.items():
            frame = ttk.Frame(nb, padding=4)
            nb.add(frame, text=title)
            txt = scrolledtext.ScrolledText(
                frame, wrap="word", font=("Consolas", 10),
                bg="#0f1218", fg="#e2e8f0", insertbackground="#e8eaed",
                relief="flat", padx=10, pady=10)
            txt.pack(fill="both", expand=True)
            txt.insert("1.0", body)
            txt.configure(state="disabled")

        foot = ttk.Frame(self, padding=(14, 0, 14, 10))
        foot.pack(fill="x")
        ttk.Label(
            foot,
            text="Tip: run Quick Test first, then explore Digits with Load Random Sample.",
            foreground="#fde68a").pack(anchor="w")

        self.bind("<Escape>", lambda e: self.destroy())
        self.focus_force()


class SenaGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SENA — Stochastic Ensemble Neural Architecture  |  Interactive Demo")
        self.geometry("1100x760")
        self.minsize(960, 680)
        self.configure(bg="#1a1d23")

        self.net = None
        self.data = None
        self.dataset_name = tk.StringVar(value="digits")
        self.epochs_var = tk.IntVar(value=35)
        self.votes_var = tk.IntVar(value=64)
        self.exit_thr = tk.DoubleVar(value=0.75)
        self.status_var = tk.StringVar(
            value="Ready — press Help (F1) or Quick Test for a guided review.")
        self.acc_var = tk.StringVar(value="—")
        self.pred_var = tk.StringVar(value="—")
        self.conf_var = tk.StringVar(value="—")
        self.early_var = tk.StringVar(value="—")

        # Interactive input buffers
        self.bit_vars = []          # for symmetry (8 bits)
        self.digit_grid = None      # 8x8 int array for digits
        self.digit_cells = []       # canvas rectangles
        self.xor_x = tk.DoubleVar(value=0.0)
        self.xor_y = tk.DoubleVar(value=0.0)
        self.iris_feats = [tk.DoubleVar(value=v) for v in (5.1, 3.5, 1.4, 0.2)]

        self._help_win = None
        self._analog_win = None
        self._quick_running = False
        self._train_done_cb = None

        self._build_style()
        self._build_layout()
        self._on_dataset_change()
        self.bind("<F1>", lambda e: self._open_help())
        self.bind("<F2>", lambda e: self._open_analog())
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        # First-run tip (non-blocking)
        self.after(400, self._show_startup_tip)

    # ------------------------------------------------------------------ UI --
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background="#1a1d23", foreground="#e8eaed",
                        fieldbackground="#2a2f3a", troughcolor="#2a2f3a")
        style.configure("TFrame", background="#1a1d23")
        style.configure("Card.TFrame", background="#232833")
        style.configure("TLabel", background="#1a1d23", foreground="#e8eaed",
                        font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 14),
                        foreground="#7dd3fc")
        style.configure("Head.TLabel", font=("Segoe UI Semibold", 11),
                        foreground="#fbbf24", background="#232833")
        style.configure("Big.TLabel", font=("Consolas", 18, "bold"),
                        foreground="#34d399", background="#232833")
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10))
        style.configure("Help.TButton", font=("Segoe UI Semibold", 10))
        style.configure("Quick.TButton", font=("Segoe UI Semibold", 10))
        style.configure("TCombobox", fieldbackground="#2a2f3a",
                        background="#2a2f3a", foreground="#e8eaed")
        style.configure("Horizontal.TScale", background="#1a1d23")
        style.configure("TLabelframe", background="#232833", foreground="#94a3b8")
        style.configure("TLabelframe.Label", background="#232833",
                        foreground="#7dd3fc", font=("Segoe UI Semibold", 10))
        style.configure("Tip.TLabel", background="#1e293b", foreground="#fde68a",
                        font=("Segoe UI", 9))
        style.map("TCombobox", fieldbackground=[("readonly", "#2a2f3a")],
                  selectbackground=[("readonly", "#334155")])

    def _build_layout(self):
        # Header
        hdr = ttk.Frame(self, padding=(16, 10))
        hdr.pack(fill="x")
        ttk.Label(hdr, text="SENA Interactive Demo", style="Title.TLabel").pack(side="left")
        ttk.Label(hdr, text="  multiplier-free · stochastic votes · early-exit",
                  foreground="#94a3b8").pack(side="left", padx=8)
        ttk.Button(hdr, text="Quick Test", style="Quick.TButton",
                   command=self._start_quick_test).pack(side="right", padx=(6, 0))
        ttk.Button(hdr, text="Analog Demo (F2)", style="Accent.TButton",
                   command=self._open_analog).pack(side="right", padx=(6, 0))
        ttk.Button(hdr, text="Help (F1)", style="Help.TButton",
                   command=self._open_help).pack(side="right")

        # Guidance strip
        tip = ttk.Frame(self, padding=(16, 4))
        tip.pack(fill="x")
        self.guide_var = tk.StringVar(
            value="Guide: Help (F1) · Analog Demo (F2) · Quick Test · or Train -> Sample -> Predict")
        ttk.Label(tip, textvariable=self.guide_var, style="Tip.TLabel").pack(fill="x")

        body = ttk.Frame(self, padding=(12, 4))
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 10))
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        self._build_controls(left)
        self._build_io_panel(right)
        self._build_output(right)

        # Status bar
        bar = ttk.Frame(self, padding=(12, 6))
        bar.pack(fill="x", side="bottom")
        ttk.Label(bar, textvariable=self.status_var, foreground="#94a3b8").pack(side="left")
        ttk.Label(bar, text="F1=Help   F2=Analog", foreground="#64748b").pack(side="right")

    def _build_controls(self, parent):
        box = ttk.LabelFrame(parent, text=" Training / Setup ", padding=12)
        box.pack(fill="x", pady=(0, 10))

        ttk.Label(box, text="Dataset").grid(row=0, column=0, sticky="w")
        ds = ttk.Combobox(box, textvariable=self.dataset_name,
                          values=list(SPECS.keys()), state="readonly", width=18)
        ds.grid(row=0, column=1, sticky="ew", pady=4)
        ds.bind("<<ComboboxSelected>>", lambda e: self._on_dataset_change())

        ttk.Label(box, text="Epochs").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(box, from_=5, to=200, textvariable=self.epochs_var,
                    width=10).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(box, text="Votes T").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(box, from_=1, to=512, textvariable=self.votes_var,
                    width=10).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(box, text="Exit thr θ").grid(row=3, column=0, sticky="w")
        ttk.Scale(box, from_=0.5, to=0.99, variable=self.exit_thr,
                  orient="horizontal", length=140).grid(row=3, column=1, sticky="ew")

        btn_row = ttk.Frame(box)
        btn_row.grid(row=4, column=0, columnspan=2, pady=(10, 4), sticky="ew")
        ttk.Button(btn_row, text="Train Network", style="Accent.TButton",
                   command=self._start_train).pack(fill="x", pady=2)
        ttk.Button(btn_row, text="Load Random Test Sample",
                   command=self._load_random_sample).pack(fill="x", pady=2)
        ttk.Button(btn_row, text="Predict (Stochastic)",
                   command=self._predict).pack(fill="x", pady=2)
        ttk.Button(btn_row, text="Clear Input",
                   command=self._clear_input).pack(fill="x", pady=2)
        ttk.Button(btn_row, text="Quick Test (Reviewer)",
                   command=self._start_quick_test).pack(fill="x", pady=(8, 2))
        ttk.Button(btn_row, text="Analog Demo (mixed-signal)",
                   command=self._open_analog).pack(fill="x", pady=2)
        ttk.Button(btn_row, text="Open Operational Help",
                   command=self._open_help).pack(fill="x", pady=2)

        # Metrics card
        met = ttk.LabelFrame(parent, text=" Live Metrics ", padding=12)
        met.pack(fill="x")
        ttk.Label(met, text="Test accuracy", style="Head.TLabel").pack(anchor="w")
        ttk.Label(met, textvariable=self.acc_var, style="Big.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(met, text="Prediction", style="Head.TLabel").pack(anchor="w")
        ttk.Label(met, textvariable=self.pred_var, style="Big.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(met, text="Confidence", style="Head.TLabel").pack(anchor="w")
        ttk.Label(met, textvariable=self.conf_var, style="Big.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(met, text="Early-exit (SENA)", style="Head.TLabel").pack(anchor="w")
        ttk.Label(met, textvariable=self.early_var, style="Big.TLabel").pack(anchor="w")

    def _build_io_panel(self, parent):
        self.io_frame = ttk.LabelFrame(parent, text=" Interactive Input ", padding=10)
        self.io_frame.pack(fill="both", expand=True, pady=(0, 8))
        self.io_inner = ttk.Frame(self.io_frame, style="Card.TFrame")
        self.io_inner.pack(fill="both", expand=True)

    def _build_output(self, parent):
        out = ttk.LabelFrame(parent, text=" Vote Histogram / Log ", padding=8)
        out.pack(fill="both", expand=False)

        self.vote_canvas = tk.Canvas(out, height=110, bg="#0f1218",
                                     highlightthickness=0)
        self.vote_canvas.pack(fill="x", pady=(0, 6))

        self.log = scrolledtext.ScrolledText(
            out, height=8, bg="#0f1218", fg="#cbd5e1",
            insertbackground="#e8eaed", font=("Consolas", 9),
            relief="flat", wrap="word")
        self.log.pack(fill="both", expand=True)
        self._log("SENA GUI ready. Press Help (F1) or run Quick Test for a guided walkthrough.")

    # ----------------------------------------------------------- guidance --
    def _show_startup_tip(self):
        choice = messagebox.askyesnocancel(
            "SENA Demo — Start Here",
            "Welcome.\n\n"
            "Recommended for reviewers:\n"
            "  Yes     = Open Analog Demo (F2)\n"
            "  No      = Open Help (F1)\n"
            "  Cancel  = stay on main window\n\n"
            "Also try: Quick Test (Iris, ~30s).",
            parent=self)
        if choice is True:
            self._open_analog()
        elif choice is False:
            self._open_help()

    def _open_help(self):
        if self._help_win is not None and self._help_win.winfo_exists():
            self._help_win.lift()
            self._help_win.focus_force()
            return
        self._help_win = HelpWindow(self)

    def _open_analog(self):
        if self._analog_win is not None and self._analog_win.winfo_exists():
            self._analog_win.lift()
            self._analog_win.focus_force()
            return
        try:
            self._analog_win = AnalogDemoWindow(self)
            self._log("Opened Analog Demo (comparator + thermal-noise neuron).")
            self.guide_var.set(
                "Analog Demo open: sweep Vin · Live Fire · Monte-Carlo PVT  |  F2 to reopen")
            self.status_var.set("Analog Demo open — behavioral mixed-signal neuron")
        except Exception as e:
            messagebox.showerror("Analog Demo", f"Could not open analog demo:\n{e}")
            self._log(f"Analog Demo error: {e}")

    def _start_quick_test(self):
        if self._quick_running:
            messagebox.showinfo("Busy", "Quick Test is already running.")
            return
        if hasattr(self, "_train_thread") and self._train_thread.is_alive():
            messagebox.showinfo("Busy", "Wait for current training to finish.")
            return
        if not messagebox.askokcancel(
                "Quick Test (Reviewer)",
                "This will:\n"
                "  • Switch to Iris (fast)\n"
                "  • Train ~20 epochs\n"
                "  • Load 5 hold-out samples\n"
                "  • Predict with early-exit\n"
                "  • Show a pass/fail checklist\n\n"
                "Typical runtime: 20–40 seconds.\nContinue?"):
            return
        self._quick_running = True
        self.guide_var.set("Quick Test running: train Iris -> sample -> predict -> report")
        self._log("===== QUICK TEST START (reviewer walkthrough) =====")
        threading.Thread(target=self._quick_test_worker, daemon=True).start()

    def _quick_test_worker(self):
        """Automated reviewer path: train Iris, score 5 samples, report checklist."""
        report = {
            "train_ok": False, "acc": None, "hits": 0, "n": 5,
            "early_saves": [], "errors": [], "lines": [],
        }
        try:
            def ui_set_ds():
                self.dataset_name.set("iris")
                self.epochs_var.set(20)
                self.votes_var.set(64)
                self.exit_thr.set(0.75)
                self._on_dataset_change()
            self.after(0, ui_set_ds)
            time.sleep(0.3)

            d = get_dataset("iris", seed=0, n_levels=8)
            cfg = StochasticConfig(
                layer_sizes=(d["n_in"], SPECS["iris"]["hid"], d["n_out"]),
                epochs=20, seed=0, weight_bits=16)
            net = StochasticNet(cfg)
            self.after(0, lambda: self._set_busy(True, "Quick Test: training Iris…"))
            t0 = time.time()
            net.fit(d["Xtr"], d["Ytr"], recal_every=4)
            acc = net.accuracy(d["Xte"], d["yte"], T=64)
            dt = time.time() - t0
            self.net = net
            self.data = d
            report["train_ok"] = True
            report["acc"] = acc
            report["lines"].append(f"Train OK in {dt:.1f}s  test accuracy={acc*100:.1f}%")

            def ui_train_done():
                self.acc_var.set(f"{acc * 100:.1f}%")
                self._log(f"[Quick Test] trained Iris  acc={acc*100:.1f}%  ({dt:.1f}s)")
            self.after(0, ui_train_done)

            names = CLASS_NAMES["iris"]
            rng = np.random.default_rng(7)
            idxs = rng.choice(len(d["Xte"]), size=5, replace=False)
            T, theta = 64, 0.75
            for k, i in enumerate(idxs):
                X = d["Xte"][i:i + 1]
                true = int(d["yte"][i])
                votes = np.zeros(d["n_out"], dtype=np.int64)
                exit_t, conf, pred_i = T, 0.0, 0
                for t in range(1, T + 1):
                    acts, _ = net.forward(X, stochastic=True)
                    votes += acts[-1][0]
                    pred_i = int(np.argmax(votes))
                    conf = votes[pred_i] / t
                    if conf >= theta:
                        exit_t = t
                        break
                hit = pred_i == true
                if hit:
                    report["hits"] += 1
                save = 100.0 * (1 - exit_t / T)
                report["early_saves"].append(save)
                line = (f"  sample {k+1}: true={names[true]:12s} pred={names[pred_i]:12s} "
                        f"conf={conf*100:5.1f}%  exit t={exit_t}/{T}  "
                        f"(-{save:.0f}% compute)  {'OK' if hit else 'MISS'}")
                report["lines"].append(line)

                # Reflect last sample in the GUI
                def show_one(Xi=X, yi=true, vi=votes.copy(), et=exit_t,
                             pi=pred_i, cf=conf, ii=int(i)):
                    self._sample_x = Xi
                    self._sample_y = yi
                    self.pred_var.set(names[pi])
                    self.conf_var.set(f"{cf * 100:.1f}%")
                    self.early_var.set(f"t={et}/{T}  (-{100*(1-et/T):.0f}% compute)")
                    self._draw_votes(vi, et, names)
                    self._log(f"[Quick Test] #{ii} pred={names[pi]} true={names[yi]} "
                              f"{'OK' if pi == yi else 'MISS'}")
                self.after(0, show_one)
                time.sleep(0.25)

        except Exception as e:
            report["errors"].append(str(e))
            self.after(0, lambda: self._log(f"[Quick Test] ERROR: {e}"))

        def finish():
            self._quick_running = False
            self._set_busy(False, "Quick Test finished")
            self.guide_var.set(
                "Quick Test done. Open Help for interpretation. Try Digits next for a harder demo.")
            self._log("===== QUICK TEST END =====")
            for ln in report["lines"]:
                self._log(ln)
            self._show_quick_report(report)
        self.after(0, finish)

    def _show_quick_report(self, report):
        hits, n = report["hits"], report["n"]
        acc = report["acc"]
        avg_save = float(np.mean(report["early_saves"])) if report["early_saves"] else 0.0
        checks = []
        checks.append(("Training completed without error", report["train_ok"]))
        checks.append(("Test accuracy >= 85% on Iris",
                       (acc is not None and acc >= 0.85)))
        checks.append((f"Interactive hits >= 3/5  (got {hits}/{n})", hits >= 3))
        checks.append(("Early-exit saved compute on average (>0%)", avg_save > 0))
        checks.append(("No runtime errors", len(report["errors"]) == 0))
        passed = all(ok for _, ok in checks)
        lines = ["SENA Quick Test — Reviewer Checklist", ""]
        for label, ok in checks:
            lines.append(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
        lines += [
            "",
            f"Summary: accuracy={acc*100:.1f}%" if acc is not None else "",
            f"         interactive hits={hits}/{n}",
            f"         mean compute saved by early-exit={avg_save:.0f}%",
            "",
            "PASS — demo is operational." if passed else "FAIL — see log for details.",
            "",
            "Next: open Help (F1) · try Digits with Load Random Sample · adjust Votes T / Exit thr.",
        ]
        messagebox.showinfo(
            "Quick Test Result — " + ("PASS" if passed else "FAIL"),
            "\n".join(lines),
            parent=self)

    # ----------------------------------------------------------- I/O panels --
    def _clear_io_inner(self):
        for w in self.io_inner.winfo_children():
            w.destroy()
        self.bit_vars = []
        self.digit_cells = []
        self.digit_grid = None

    def _on_dataset_change(self):
        name = self.dataset_name.get()
        self.epochs_var.set(SPECS[name]["ep"])
        self.votes_var.set(SPECS[name]["T"])
        self.net = None
        self.data = None
        self.acc_var.set("—")
        self.pred_var.set("—")
        self.conf_var.set("—")
        self.early_var.set("—")
        self._clear_io_inner()
        builders = {
            "symmetry": self._build_symmetry_input,
            "digits": self._build_digits_input,
            "xor2d": self._build_xor_input,
            "iris": self._build_iris_input,
            "cancer": self._build_cancer_input,
        }
        builders[name]()
        self._log(f"Dataset → {SPECS[name]['label']}")

    def _build_symmetry_input(self):
        ttk.Label(self.io_inner, text="Toggle 8 input bits  (palindrome ⇒ Symmetric)",
                  background="#232833").pack(anchor="w", pady=(0, 8))
        row = ttk.Frame(self.io_inner, style="Card.TFrame")
        row.pack()
        self.bit_vars = []
        for i in range(8):
            v = tk.IntVar(value=0)
            self.bit_vars.append(v)
            b = tk.Checkbutton(
                row, text=str(i), variable=v, onvalue=1, offvalue=0,
                indicatoron=False, width=4, height=2,
                bg="#334155", fg="#e8eaed", selectcolor="#0ea5e9",
                activebackground="#475569", font=("Consolas", 12, "bold"),
                command=self._preview_symmetry)
            b.pack(side="left", padx=3)
        ttk.Button(self.io_inner, text="Make Palindrome",
                   command=self._make_palindrome).pack(pady=10)
        self.sym_hint = ttk.Label(self.io_inner, text="", background="#232833")
        self.sym_hint.pack()

    def _preview_symmetry(self):
        bits = [v.get() for v in self.bit_vars]
        is_sym = bits == bits[::-1]
        self.sym_hint.configure(
            text=f"Pattern {bits}  →  {'SYMMETRIC' if is_sym else 'asymmetric'}",
            foreground="#34d399" if is_sym else "#f87171")

    def _make_palindrome(self):
        for i in range(4):
            self.bit_vars[7 - i].set(self.bit_vars[i].get())
        self._preview_symmetry()

    def _build_digits_input(self):
        ttk.Label(self.io_inner,
                  text="Click / drag on the 8×8 pad to draw a digit (0–9)",
                  background="#232833").pack(anchor="w", pady=(0, 6))
        self.digit_grid = np.zeros((8, 8), dtype=np.int64)
        self.digit_canvas = tk.Canvas(self.io_inner, width=320, height=320,
                                      bg="#0f1218", highlightthickness=1,
                                      highlightbackground="#475569")
        self.digit_canvas.pack()
        cell = 40
        self.digit_cells = []
        for r in range(8):
            row = []
            for c in range(8):
                x0, y0 = c * cell, r * cell
                rid = self.digit_canvas.create_rectangle(
                    x0 + 1, y0 + 1, x0 + cell - 1, y0 + cell - 1,
                    fill="#1e293b", outline="#334155")
                row.append(rid)
            self.digit_cells.append(row)
        self.digit_canvas.bind("<Button-1>", self._digit_paint)
        self.digit_canvas.bind("<B1-Motion>", self._digit_paint)
        self.digit_canvas.bind("<Button-3>", self._digit_erase)
        self.digit_canvas.bind("<B3-Motion>", self._digit_erase)
        hint = ttk.Frame(self.io_inner, style="Card.TFrame")
        hint.pack(pady=8)
        ttk.Label(hint, text="L-click paint · R-click erase · or Load Random Test Sample",
                  background="#232833", foreground="#94a3b8").pack()

    def _digit_paint(self, event):
        self._digit_set(event, 1)

    def _digit_erase(self, event):
        self._digit_set(event, 0)

    def _digit_set(self, event, val):
        if self.digit_grid is None:
            return
        self._sample_x = None  # freehand overrides loaded sample
        c, r = event.x // 40, event.y // 40
        if 0 <= r < 8 and 0 <= c < 8:
            self.digit_grid[r, c] = val
            color = "#38bdf8" if val else "#1e293b"
            self.digit_canvas.itemconfigure(self.digit_cells[r][c], fill=color)

    def _build_xor_input(self):
        ttk.Label(self.io_inner,
                  text="Set 2-D coordinates for noisy XOR  (centers near ±1, ±1)",
                  background="#232833").pack(anchor="w", pady=(0, 8))
        for label, var in (("X", self.xor_x), ("Y", self.xor_y)):
            row = ttk.Frame(self.io_inner, style="Card.TFrame")
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=label, width=4, background="#232833").pack(side="left")
            ttk.Scale(row, from_=-2.5, to=2.5, variable=var,
                      orient="horizontal", length=280).pack(side="left", fill="x", expand=True)
            ttk.Label(row, textvariable=var, width=6, background="#232833").pack(side="left")
        presets = ttk.Frame(self.io_inner, style="Card.TFrame")
        presets.pack(pady=10)
        for name, xv, yv in [("(--)", -1, -1), ("(++)", 1, 1),
                             ("(-+)", -1, 1), ("(+-)", 1, -1)]:
            ttk.Button(presets, text=name,
                       command=lambda a=xv, b=yv: (self.xor_x.set(a), self.xor_y.set(b))
                       ).pack(side="left", padx=4)

    def _build_iris_input(self):
        names = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
        ranges = [(4.0, 8.0), (2.0, 4.5), (1.0, 7.0), (0.1, 2.5)]
        ttk.Label(self.io_inner, text="Adjust Iris flower features",
                  background="#232833").pack(anchor="w", pady=(0, 8))
        for i, (nm, (lo, hi)) in enumerate(zip(names, ranges)):
            row = ttk.Frame(self.io_inner, style="Card.TFrame")
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=nm, width=14, background="#232833").pack(side="left")
            ttk.Scale(row, from_=lo, to=hi, variable=self.iris_feats[i],
                      orient="horizontal", length=260).pack(side="left", fill="x", expand=True)
            ttk.Label(row, textvariable=self.iris_feats[i], width=6,
                      background="#232833").pack(side="left")
        presets = ttk.Frame(self.io_inner, style="Card.TFrame")
        presets.pack(pady=10)
        for label, vals in [("Setosa", (5.1, 3.5, 1.4, 0.2)),
                            ("Versicolor", (5.9, 2.8, 4.2, 1.3)),
                            ("Virginica", (6.7, 3.0, 5.5, 2.1))]:
            ttk.Button(presets, text=label,
                       command=lambda v=vals: [self.iris_feats[i].set(v[i]) for i in range(4)]
                       ).pack(side="left", padx=4)

    def _build_cancer_input(self):
        ttk.Label(
            self.io_inner,
            text="Cancer uses 30 continuous features (thermometer-coded).\n"
                 "Use  Load Random Test Sample  to pull a real case from the hold-out set,\n"
                 "then Predict.",
            background="#232833", justify="left").pack(anchor="w", pady=20)
        self.cancer_idx = None
        self.cancer_info = ttk.Label(self.io_inner, text="No sample loaded.",
                                     background="#232833", foreground="#94a3b8")
        self.cancer_info.pack(anchor="w")

    # --------------------------------------------------------------- helpers --
    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")

    def _set_busy(self, busy: bool, msg: str = ""):
        self.status_var.set(msg if msg else ("Working…" if busy else "Ready."))
        self.config(cursor="watch" if busy else "")

    # ---------------------------------------------------------------- train --
    def _start_train(self):
        if hasattr(self, "_train_thread") and self._train_thread.is_alive():
            messagebox.showinfo("Busy", "Training already in progress.")
            return
        self._train_thread = threading.Thread(target=self._train_worker, daemon=True)
        self._train_thread.start()

    def _train_worker(self):
        name = self.dataset_name.get()
        ep = int(self.epochs_var.get())
        s = SPECS[name]
        self.after(0, lambda: self._set_busy(True, f"Training {name} for {ep} epochs…"))
        self.after(0, lambda: self._log(f"Training {SPECS[name]['label']}  epochs={ep}"))
        try:
            d = get_dataset(name, seed=0, n_levels=8)
            cfg = StochasticConfig(
                layer_sizes=(d["n_in"], s["hid"], d["n_out"]),
                epochs=ep, seed=0, weight_bits=16)
            net = StochasticNet(cfg)
            t0 = time.time()
            net.fit(d["Xtr"], d["Ytr"], recal_every=4)
            T = int(self.votes_var.get())
            acc = net.accuracy(d["Xte"], d["yte"], T=T)
            dt = time.time() - t0
            self.net = net
            self.data = d
            self._refs = getattr(d, "refs", None)

            def done():
                self.acc_var.set(f"{acc * 100:.1f}%")
                self._set_busy(False, f"Trained in {dt:.1f}s — test acc {acc * 100:.1f}%")
                self._log(f"DONE  test accuracy = {acc * 100:.2f}%  "
                          f"(n_in={d['n_in']}, hid={s['hid']}, T={T})")
            self.after(0, done)
        except Exception as e:
            self.after(0, lambda: self._set_busy(False, "Train failed"))
            self.after(0, lambda: self._log(f"ERROR: {e}"))
            self.after(0, lambda: messagebox.showerror("Training error", str(e)))

    # ------------------------------------------------------------- get input --
    def _get_input_vector(self):
        name = self.dataset_name.get()
        if self.net is None or self.data is None:
            raise RuntimeError("Train the network first.")

        if name == "symmetry":
            return np.array([[v.get() for v in self.bit_vars]], dtype=np.int64)

        if name == "digits":
            # Match thermometer encoding used in training (8 levels on pixel 0..16)
            raw = self.digit_grid.reshape(1, -1).astype(float) * 16.0  # paint → intensity
            Xb, _ = thermometer_encode(raw, n_levels=8,
                                       refs=[np.linspace(2, 14, 8)] * 64)
            # Prefer loading a real sample when grid empty
            if self.digit_grid.sum() == 0:
                raise RuntimeError("Draw a digit or Load Random Test Sample.")
            # Re-encode using same method as dataset: pixels are 0..16 typically
            # Our training used thermometer on original digits.data — so for freehand
            # we approximate; for random samples we use stored vectors.
            if hasattr(self, "_sample_x") and self._sample_x is not None:
                return self._sample_x
            return Xb.astype(np.int64)

        if name == "xor2d":
            X = np.array([[self.xor_x.get(), self.xor_y.get()]], dtype=float)
            Xb, _ = thermometer_encode(X, n_levels=12)
            # pad/truncate to training width
            n = self.data["n_in"]
            if Xb.shape[1] < n:
                Xb = np.pad(Xb, ((0, 0), (0, n - Xb.shape[1])))
            return Xb[:, :n].astype(np.int64)

        if name == "iris":
            X = np.array([[v.get() for v in self.iris_feats]], dtype=float)
            Xb, _ = thermometer_encode(X, n_levels=8)
            n = self.data["n_in"]
            if Xb.shape[1] < n:
                Xb = np.pad(Xb, ((0, 0), (0, n - Xb.shape[1])))
            return Xb[:, :n].astype(np.int64)

        if name == "cancer":
            if not hasattr(self, "_sample_x") or self._sample_x is None:
                raise RuntimeError("Load Random Test Sample for cancer.")
            return self._sample_x

        raise RuntimeError(f"Unknown dataset {name}")

    def _load_random_sample(self):
        if self.data is None:
            messagebox.showinfo("Train first", "Train the network before loading samples.")
            return
        rng = np.random.default_rng()
        i = int(rng.integers(0, len(self.data["Xte"])))
        self._sample_x = self.data["Xte"][i:i + 1]
        self._sample_y = int(self.data["yte"][i])
        name = self.dataset_name.get()
        label = CLASS_NAMES[name][self._sample_y] if self._sample_y < len(CLASS_NAMES[name]) else str(self._sample_y)
        self._log(f"Loaded test sample #{i}  true label = {label} ({self._sample_y})")

        if name == "symmetry":
            bits = self._sample_x[0]
            for j, v in enumerate(self.bit_vars):
                v.set(int(bits[j]))
            self._preview_symmetry()
        elif name == "digits":
            # Reverse-ish display: show intensity from first thermometer band
            # Reconstruct approximate 8x8 by summing thermometer groups
            x = self._sample_x[0]
            # each pixel → 8 levels; intensity ≈ sum of active bits
            pix = x.reshape(64, 8).sum(axis=1).reshape(8, 8)
            self.digit_grid = (pix > 0).astype(np.int64)
            for r in range(8):
                for c in range(8):
                    color = "#38bdf8" if self.digit_grid[r, c] else "#1e293b"
                    # brightness by count
                    if pix[r, c] > 0:
                        shade = min(255, 80 + int(pix[r, c]) * 22)
                        color = f"#{shade:02x}c8f8"
                    self.digit_canvas.itemconfigure(self.digit_cells[r][c], fill=color)
            self._sample_x = self.data["Xte"][i:i + 1]  # keep exact vector
        elif name == "cancer":
            self.cancer_info.configure(
                text=f"Sample #{i} loaded — true: {label}",
                foreground="#34d399")
        elif name == "iris":
            # Can't perfectly reverse thermometer; keep sample vector for predict
            self._log("Iris: using exact binary vector from test set for prediction.")
        self.status_var.set(f"Sample #{i} loaded (true={label})")

    def _clear_input(self):
        self._sample_x = None
        name = self.dataset_name.get()
        if name == "symmetry":
            for v in self.bit_vars:
                v.set(0)
            self._preview_symmetry()
        elif name == "digits" and self.digit_grid is not None:
            self.digit_grid[:] = 0
            for r in range(8):
                for c in range(8):
                    self.digit_canvas.itemconfigure(self.digit_cells[r][c], fill="#1e293b")
        elif name == "xor2d":
            self.xor_x.set(0.0)
            self.xor_y.set(0.0)
        elif name == "cancer" and hasattr(self, "cancer_info"):
            self.cancer_info.configure(text="No sample loaded.", foreground="#94a3b8")
        self.pred_var.set("—")
        self.conf_var.set("—")
        self.early_var.set("—")
        self.vote_canvas.delete("all")
        self._log("Input cleared.")

    # ------------------------------------------------------------- predict --
    def _predict(self):
        try:
            X = self._get_input_vector()
        except Exception as e:
            messagebox.showwarning("Input", str(e))
            return
        if self.net is None:
            messagebox.showwarning("Train first", "Train the network first.")
            return

        T = max(1, int(self.votes_var.get()))
        theta = float(self.exit_thr.get())
        name = self.dataset_name.get()
        names = CLASS_NAMES[name]

        # SENA-style sequential voting with early exit
        n_out = self.net.sizes[-1]
        votes = np.zeros(n_out, dtype=np.int64)
        exit_t = T
        for t in range(1, T + 1):
            acts, _ = self.net.forward(X, stochastic=True)
            votes += acts[-1][0]
            if n_out == 1:
                # binary: confidence = max(p, 1-p)
                p1 = votes[0] / t
                conf = max(p1, 1 - p1)
                pred_i = 1 if p1 > 0.5 else 0
            else:
                top = int(np.argmax(votes))
                conf = votes[top] / t
                pred_i = top
            if conf >= theta:
                exit_t = t
                break

        pred_name = names[pred_i] if pred_i < len(names) else str(pred_i)
        self.pred_var.set(pred_name)
        self.conf_var.set(f"{conf * 100:.1f}%")
        saved = 100.0 * (1 - exit_t / T)
        self.early_var.set(f"t={exit_t}/{T}  (−{saved:.0f}% compute)")
        self._draw_votes(votes, exit_t, names)
        true_txt = ""
        if hasattr(self, "_sample_y") and self._sample_y is not None and getattr(self, "_sample_x", None) is not None:
            # only show true if we used a loaded sample
            if np.array_equal(X, self._sample_x):
                true_name = names[self._sample_y]
                ok = "✓" if pred_i == self._sample_y else "✗"
                true_txt = f"  true={true_name} {ok}"
        self._log(f"Predict → {pred_name}  conf={conf * 100:.1f}%  "
                  f"early-exit t={exit_t}/{T}{true_txt}")
        self.status_var.set(f"Prediction: {pred_name} ({conf * 100:.1f}%)")

    def _draw_votes(self, votes, t_used, names):
        cv = self.vote_canvas
        cv.delete("all")
        w = max(cv.winfo_width(), 400)
        h = 110
        n = len(votes)
        if n == 1:
            # show 0 vs 1
            counts = [t_used - int(votes[0]), int(votes[0])]
            labels = names[:2] if len(names) >= 2 else ["0", "1"]
        else:
            counts = [int(v) for v in votes]
            labels = names[:n]
        mx = max(counts) if max(counts) > 0 else 1
        bw = max(12, (w - 40) // max(len(counts), 1) - 6)
        for i, (c, lab) in enumerate(zip(counts, labels)):
            x0 = 20 + i * (bw + 6)
            bar_h = int((h - 28) * c / mx)
            y0 = h - 18 - bar_h
            cv.create_rectangle(x0, y0, x0 + bw, h - 18, fill="#0ea5e9", outline="")
            cv.create_text(x0 + bw / 2, h - 8, text=lab[:4], fill="#94a3b8",
                           font=("Segoe UI", 8))
            cv.create_text(x0 + bw / 2, y0 - 8, text=str(c), fill="#e2e8f0",
                           font=("Consolas", 8))


def main():
    import argparse
    p = argparse.ArgumentParser(description="SENA Interactive GUI")
    p.add_argument("--no-tip", action="store_true",
                   help="Skip the startup Help prompt (useful for demos)")
    p.add_argument("--analog", action="store_true",
                   help="Open Analog Demo window on startup")
    p.add_argument("--quick-test-cli", action="store_true",
                   help="Run headless Quick Test checklist (no GUI) and exit")
    args = p.parse_args()

    if args.quick_test_cli:
        _cli_quick_test()
        return

    app = SenaGUI()
    if args.no_tip:
        app._show_startup_tip = lambda: None  # type: ignore
    if args.analog:
        app.after(200, app._open_analog)
    app.mainloop()


def _cli_quick_test():
    """Headless version of the GUI Quick Test for CI / reviewers without display."""
    print("SENA Quick Test (CLI) — Iris walkthrough")
    d = get_dataset("iris", seed=0, n_levels=8)
    cfg = StochasticConfig(
        layer_sizes=(d["n_in"], SPECS["iris"]["hid"], d["n_out"]),
        epochs=20, seed=0)
    net = StochasticNet(cfg)
    t0 = time.time()
    net.fit(d["Xtr"], d["Ytr"], recal_every=4)
    acc = net.accuracy(d["Xte"], d["yte"], T=64)
    print(f"  Train OK in {time.time()-t0:.1f}s  test accuracy={acc*100:.1f}%")
    names = CLASS_NAMES["iris"]
    rng = np.random.default_rng(7)
    idxs = rng.choice(len(d["Xte"]), size=5, replace=False)
    hits, saves = 0, []
    T, theta = 64, 0.75
    for k, i in enumerate(idxs):
        X = d["Xte"][i:i + 1]
        true = int(d["yte"][i])
        votes = np.zeros(d["n_out"], dtype=np.int64)
        exit_t, conf, pred_i = T, 0.0, 0
        for t in range(1, T + 1):
            acts, _ = net.forward(X, stochastic=True)
            votes += acts[-1][0]
            pred_i = int(np.argmax(votes))
            conf = votes[pred_i] / t
            if conf >= theta:
                exit_t = t
                break
        hit = pred_i == true
        hits += int(hit)
        save = 100.0 * (1 - exit_t / T)
        saves.append(save)
        print(f"  sample {k+1}: true={names[true]:12s} pred={names[pred_i]:12s} "
              f"conf={conf*100:5.1f}%  exit t={exit_t}/{T}  "
              f"(-{save:.0f}% compute)  {'OK' if hit else 'MISS'}")
    checks = [
        ("Training completed without error", True),
        ("Test accuracy >= 85% on Iris", acc >= 0.85),
        (f"Interactive hits >= 3/5  (got {hits}/5)", hits >= 3),
        ("Early-exit saved compute on average (>0%)", float(np.mean(saves)) > 0),
    ]
    print("\nChecklist:")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
    passed = all(ok for _, ok in checks)
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}  "
          f"(mean compute saved={float(np.mean(saves)):.0f}%)")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
