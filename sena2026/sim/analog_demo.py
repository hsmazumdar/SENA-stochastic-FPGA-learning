"""
Analog Demo window — interactive mixed-signal stochastic neuron for reviewers.
"""

from __future__ import annotations
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from analog_neuron import AnalogNeuronParams, AnalogStochasticNeuron, sweep_transfer


class AnalogDemoWindow(tk.Toplevel):
    """Interactive analog stochastic-neuron demo (behavioral SPICE-class model)."""

    BMP_GRID_EVERY = 3      # training steps between 64x64 output-map refreshes
    BMP_GRID_T = 192        # stochastic passes integrated per displayed pixel
    BMP_OVERLAP_MAX = 6     # learned edges kept in the overlap panel

    def __init__(self, master):
        super().__init__(master)
        self.title("SENA Analog Demo — Comparator + Thermal Noise Neuron")
        self.geometry("1080x720")
        self.minsize(900, 620)
        self.configure(bg="#1a1d23")
        self.transient(master)

        self.vin = tk.DoubleVar(value=0.0)
        self.temp = tk.DoubleVar(value=300.0)
        self.sigma0 = tk.DoubleVar(value=0.35)
        self.mismatch = tk.DoubleVar(value=0.05)
        self.n_samples = tk.IntVar(value=256)
        self.live_rate = tk.StringVar(value="—")
        self.metrics_var = tk.StringVar(value="")
        self._anim_job = None
        self._closed = False
        self._bitstream = []
        self.net = None
        self.circle_img = None
        self.shape_name = tk.StringVar(value="circle")
        self.shape_frac = tk.DoubleVar(value=0.55)
        self.engine = tk.StringVar(value="mazumdar")
        self.py_epochs = tk.IntVar(value=300)
        self.py_status = tk.StringVar(value="Select Cir / Sqr / Trngl, then Analog Quick Test (1000 ep)")
        self.py_mse = tk.StringVar(value="—")
        self.py_err = tk.StringVar(value="—")
        self._shape_btns = {}

        # --- tab 3: live BMP pixel learning ---
        self.bmp_engine = tk.StringVar(value="mazumdar")
        self.bmp_mode = tk.StringVar(value="train")
        self.bmp_batch = tk.IntVar(value=256)
        self.bmp_status = tk.StringVar(
            value="Press Run to stream random pixels from the 64x64 bitmap.")
        self.bmp_acc = tk.StringVar(value="—")
        self.bmp_seen = tk.StringVar(value="0")
        self.bmp_img = None
        self._bmp_lock = threading.Lock()
        self._bmp_snap = None
        self._bmp_run = False
        self._bmp_job = None
        self._bmp_learner = None
        self._bmp_stream = None
        self._bmp_hist = []
        self._bmp_overlap = []
        self._bmp_error = None
        self._bmp_gen = 0
        # plain mirrors of the Tk vars: the worker thread must never touch Tk
        self._bmp_batch_v = 256
        self._bmp_mode_v = "train"
        self._bmp_engine_v = "mazumdar"
        self._bmp_grid = None
        self._bmp_grid_id = 0
        self._bmp_overlap_id = -1

        self._build()
        self._refresh_curves()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda e: self._on_close())


    def _build(self):
        hdr = ttk.Frame(self, padding=(12, 8))
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Analog Demo",
                  style="Title.TLabel").pack(side="left")
        ttk.Label(hdr, text="  2-30-1 Pythagoras  |  shapes",
                  foreground="#94a3b8").pack(side="left", padx=8)

        shp = ttk.Frame(hdr)
        shp.pack(side="left", padx=16)
        ttk.Label(shp, text="Shape:", foreground="#fde68a").pack(side="left", padx=(0, 6))
        for key, label in (("circle", "Cir"), ("square", "Sqr"), ("triangle", "Trngl")):
            b = tk.Button(
                shp, text=label, width=6,
                font=("Segoe UI Semibold", 10),
                command=lambda k=key: self._select_shape(k),
                bg="#334155", fg="#e2e8f0", activebackground="#0ea5e9",
                relief="raised", bd=2)
            b.pack(side="left", padx=3)
            self._shape_btns[key] = b
        self._highlight_shape_btn()

        ttk.Button(hdr, text="Analog Quick Test",
                   command=self._py_quick_demo).pack(side="right", padx=(8, 0))
        ttk.Button(hdr, text="Close", command=self._on_close).pack(side="right")

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)

        tab_neuron = ttk.Frame(self.nb)
        tab_pythag = ttk.Frame(self.nb)
        tab_bmp = ttk.Frame(self.nb)
        self.nb.add(tab_neuron, text="1) Stochastic Neuron")
        self.nb.add(tab_pythag, text="2) Pythagoras 2-30-1")
        self.nb.add(tab_bmp, text="3) BMP 64x64 Live Learn")

        tip = ttk.Frame(tab_neuron, padding=(8, 4))
        tip.pack(fill="x")
        ttk.Label(
            tip,
            text="Guide: move Vin / Temperature. Live Fire = bitstream. Monte-Carlo = PVT.",
            style="Tip.TLabel").pack(fill="x")
        body = ttk.Frame(tab_neuron, padding=4)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 8))
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self._build_controls(left)
        self._build_plots(right)

        self._build_pythagoras_tab(tab_pythag)
        self._build_bmp_tab(tab_bmp)
        self.nb.select(2)



    def _build_pythagoras_tab(self, parent):
        tip = ttk.Frame(parent, padding=(8, 4))
        tip.pack(fill="x")
        ttk.Label(
            tip,
            text="Select Cir / Sqr / Trngl at top (balanced black~=white).  "
                 "Net 2->30->1 learns h=sqrt(a^2+b^2).  "
                 "Analog Quick Test = 1000 epochs + reconstruct.",
            style="Tip.TLabel").pack(fill="x")

        body = ttk.Frame(parent, padding=4)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 8))
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        box = ttk.LabelFrame(left, text=" Train 2-30-1 ", padding=10)
        box.pack(fill="x")
        ttk.Label(box, text="Architecture", style="Head.TLabel").pack(anchor="w")
        ttk.Label(box, text="2 -> 30 -> 1\nanalog in / out (normalized)",
                  background="#232833", font=("Consolas", 11),
                  foreground="#34d399").pack(anchor="w", pady=(0, 8))

        ttk.Label(box, text="Engine", style="Head.TLabel").pack(anchor="w")
        ttk.Radiobutton(
            box, text="Mazumdar 1995 (no × ÷ exp)", value="mazumdar",
            variable=self.engine, command=self._on_engine_change
        ).pack(anchor="w")
        ttk.Radiobutton(
            box, text="MLP ver01 (float backprop)", value="mlp",
            variable=self.engine, command=self._on_engine_change
        ).pack(anchor="w", pady=(0, 6))

        row = ttk.Frame(box)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Epochs", width=10).pack(side="left")
        ttk.Spinbox(row, from_=50, to=5000, textvariable=self.py_epochs,
                    width=8).pack(side="left")

        srow = ttk.Frame(box)
        srow.pack(fill="x", pady=4)
        ttk.Label(srow, text="Size", width=10).pack(side="left")
        ttk.Scale(srow, from_=0.3, to=0.9, variable=self.shape_frac,
                  orient="horizontal", length=120,
                  command=lambda _=None: self._on_size_change()
                  ).pack(side="left", fill="x", expand=True)
        self._size_lbl = ttk.Label(srow, width=5)
        self._size_lbl.pack(side="left")
        self._sync_size_lbl()

        ttk.Button(box, text="1. Generate Selected Shape",
                   command=self._py_make_shape).pack(fill="x", pady=2)
        ttk.Button(box, text="2. Train Pythagoras Net",
                   style="Accent.TButton",
                   command=self._py_start_train).pack(fill="x", pady=2)
        ttk.Button(box, text="3. Reconstruct Shape",
                   command=self._py_reconstruct).pack(fill="x", pady=2)
        ttk.Button(box, text="Analog Quick Test (1000 ep)",
                   command=self._py_quick_demo).pack(fill="x", pady=(8, 2))

        met = ttk.LabelFrame(left, text=" Status ", padding=10)
        met.pack(fill="x", pady=(10, 0))
        ttk.Label(met, textvariable=self.py_status, wraplength=220,
                  background="#232833").pack(anchor="w")
        ttk.Label(met, text="MSE", style="Head.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Label(met, textvariable=self.py_mse, style="Big.TLabel").pack(anchor="w")
        ttk.Label(met, text="Shape radius RMSE", style="Head.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Label(met, textvariable=self.py_err, style="Big.TLabel").pack(anchor="w")

        eq = ttk.LabelFrame(left, text=" Target map ", padding=10)
        eq.pack(fill="x", pady=(10, 0))
        ttk.Label(
            eq,
            text="h = sqrt(a^2 + b^2)\nh_norm = h / sqrt(2)\n\n"
                 "Shapes: Cir / Sqr / Trngl\n(balanced B/W pixels)",
            justify="left", background="#232833",
            font=("Consolas", 9), foreground="#cbd5e1").pack(anchor="w")

        self.py_fig = Figure(figsize=(7.2, 5.8), dpi=100, facecolor="#1a1d23")
        self.py_ax1 = self.py_fig.add_subplot(221)
        self.py_ax2 = self.py_fig.add_subplot(222)
        self.py_ax3 = self.py_fig.add_subplot(212)
        self._style_py_axes()
        self.py_fig.tight_layout(pad=2.0)
        self.py_canvas = FigureCanvasTkAgg(self.py_fig, master=right)
        self.py_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.after(100, self._py_make_shape)

    def _style_py_axes(self):
        for ax in (self.py_ax1, self.py_ax2, self.py_ax3):
            ax.set_facecolor("#0f1218")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.title.set_color("#e2e8f0")
            ax.xaxis.label.set_color("#94a3b8")
            ax.yaxis.label.set_color("#94a3b8")


    def _alive(self) -> bool:
        """Main thread only -- winfo_exists() is a Tk call."""
        try:
            return bool(self.winfo_exists())
        except Exception:
            return False

    def _post(self, fn):
        """Hand a callback to the Tk main loop, dropping it if the window is
        already gone. Background trainers outlive the window when the user
        closes it mid-run, so every cross-thread callback goes through here.

        `_closed` is a plain bool on purpose: a worker thread must not call
        into Tk at all, not even to ask whether the widget still exists.
        """
        if self._closed:
            return
        try:
            self.after(0, lambda: None if self._closed or not self._alive()
                       else fn())
        except Exception:
            pass

    def _highlight_shape_btn(self):
        cur = self.shape_name.get()
        for k, b in self._shape_btns.items():
            if k == cur:
                b.configure(bg="#0ea5e9", fg="#0f172a", relief="sunken")
            else:
                b.configure(bg="#334155", fg="#e2e8f0", relief="raised")

    def _select_shape(self, shape: str):
        self.shape_name.set(shape)
        self._highlight_shape_btn()
        if self.nb.index("current") == 2:
            self._bmp_load_shape()
        else:
            self.nb.select(1)
            self._py_make_shape()

    def _on_engine_change(self):
        eng = self.engine.get()
        # sensible epoch defaults per engine
        self.py_epochs.set(300 if eng == "mazumdar" else 1000)
        self.net = None
        self.py_mse.set("—")
        self.py_err.set("—")
        if eng == "mazumdar":
            self.py_status.set(
                "Engine: MAZUMDAR 1995 — ADD/SUB/COMPARE/LOGIC + RNG only "
                "(no multiply, divide, exp). Train to see it learn h=sqrt(a²+b²).")
        else:
            self.py_status.set(
                "Engine: ver01 float MLP (multiply + tanh + sigmoid) — baseline.")

    def _sync_size_lbl(self):
        try:
            self._size_lbl.configure(text=f"{self.shape_frac.get():.2f}")
        except Exception:
            pass

    def _on_size_change(self):
        self._sync_size_lbl()
        if hasattr(self, "_size_job") and self._size_job:
            self.after_cancel(self._size_job)
        self._size_job = self.after(200, self._py_make_shape)

    def _py_make_shape(self):
        from pythagoras_analog import make_shape_image, shape_balance_stats
        shape = self.shape_name.get()
        frac = float(self.shape_frac.get())
        self.circle_img = make_shape_image(shape, size=128, frac=frac)
        st = shape_balance_stats(self.circle_img)
        n_bal = min(st["n_white"], st["n_black"])
        self.py_status.set(
            f"{shape.upper()} size={frac:.2f}  canvas W={st['n_white']} B={st['n_black']}  "
            f"train balanced: {min(n_bal, 2000)} white == {min(n_bal, 2000)} black")
        self._py_show_image_only()

    def _py_make_circle(self):
        self.shape_name.set("circle")
        self._highlight_shape_btn()
        self._py_make_shape()

    def _py_show_image_only(self):
        self.py_ax1.clear(); self.py_ax2.clear(); self.py_ax3.clear()
        self._style_py_axes()
        shape = self.shape_name.get()
        if self.circle_img is not None:
            from pythagoras_analog import shape_balance_stats
            st = shape_balance_stats(self.circle_img)
            self.py_ax1.imshow(self.circle_img, cmap="gray", origin="upper",
                               vmin=0, vmax=255)
            self.py_ax1.set_title(
                f"{shape.upper()}  W={st['n_white']} B={st['n_black']}")
            self.py_ax1.set_xticks([]); self.py_ax1.set_yticks([])
        self.py_ax2.set_title("Reconstructed (after train)")
        self.py_ax2.set_aspect("equal")
        self.py_ax3.set_title("Training MSE")
        self.py_ax3.set_xlabel("epoch")
        self.py_fig.tight_layout(pad=2.0)
        self.py_canvas.draw_idle()

    def _py_start_train(self):
        import threading
        if self.circle_img is None:
            self._py_make_shape()
        if getattr(self, "_py_busy", False):
            messagebox.showinfo("Busy", "Training already running.", parent=self)
            return
        self._py_busy = True
        # every Tk variable the trainer needs is read here, on the main thread
        epochs = int(self.py_epochs.get())
        shape = self.shape_name.get()
        frac = float(self.shape_frac.get())
        engine = self.engine.get()
        self.py_status.set(f"Training 2-30-1 on {shape} for {epochs} epochs...")
        threading.Thread(target=self._py_train_worker,
                         args=(epochs, shape, frac, engine),
                         daemon=True).start()

    def _py_train_worker(self, epochs: int, shape: str, frac: float,
                         engine: str):
        try:
            if engine == "mazumdar":
                from pythagoras_mazumdar import (MazumdarPythagoras,
                                                 dataset_from_shape)
                AB, h, n_per = dataset_from_shape(self.circle_img,
                                                  n_extra=1000, seed=0)

                def cb(ep, mse):
                    self._post(lambda e=ep, m=mse: (
                        self.py_mse.set(f"{m:.5f}"),
                        self.py_status.set(
                            f"MAZUMDAR training {shape}... ep {e}/{epochs}  "
                            f"(ADD/SUB/CMP only, {n_per}W=={n_per}B)")))

                net = MazumdarPythagoras(n_levels=24, hidden=30, seed=0)
                hist_mse = net.fit(AB, h, epochs=epochs,
                                   samples_per_epoch=800, callback=cb)
                self.net = net
                *_, rmse = net.reconstruct_shape(shape, n_theta=360, size=128,
                                                 frac=frac, T=256)
                eng_line = ("Engine: Mazumdar 1995\n"
                            "Ops: ADD, SUB, COMPARE, LOGIC, RNG\n"
                            "(no multiply / divide / exp)\n")
                ok = rmse < 0.15
            else:
                from pythagoras_analog import (samples_from_shape_image,
                                               AnalogMLP2301)
                X, y, n_per = samples_from_shape_image(self.circle_img,
                                                       n_extra=1000, seed=0)

                def cb(ep, mse):
                    self._post(lambda e=ep, m=mse: (
                        self.py_mse.set(f"{m:.5f}"),
                        self.py_status.set(
                            f"MLP training {shape}... epoch {e}/{epochs}  "
                            f"(balanced {n_per}W=={n_per}B)")))

                net = AnalogMLP2301(seed=0, lr=0.08)
                hist_mse = net.train(X, y, epochs=epochs, batch=64,
                                     callback=cb)
                self.net = net
                *_, rmse = net.reconstruct_shape(shape, n_theta=360, size=128,
                                                 frac=frac)
                eng_line = "Engine: ver01 float MLP (baseline)\n"
                ok = hist_mse[-1] < 0.01 and rmse < 0.25

            def done():
                self._py_busy = False
                self.py_mse.set(f"{hist_mse[-1]:.5f}")
                self.py_err.set(f"{rmse:.4f}")
                self.py_status.set(f"Trained on {shape}. Reconstructing...")
                self._py_plot_all(hist_mse, auto_recon=True)
                msg = (
                    f"{eng_line}"
                    f"Shape: {shape.upper()}\n"
                    f"Epochs: {epochs}\n"
                    f"Final MSE: {hist_mse[-1]:.5f}\n"
                    f"Shape RMSE: {rmse:.4f}\n\n"
                    f"{'PASS' if ok else 'CHECK'}"
                )
                messagebox.showinfo("Analog Quick Test", msg, parent=self)
            self._post(done)
        except Exception as e:
            self._post(lambda: self._py_fail(e))

    def _py_fail(self, e):
        self._py_busy = False
        self.py_status.set(f"Error: {e}")
        messagebox.showerror("Train error", str(e), parent=self)

    def _py_reconstruct(self):
        if self.net is None:
            messagebox.showinfo("Train first", "Train the 2-30-1 net first.", parent=self)
            return
        self._py_plot_all(getattr(self, "_last_hist", None), auto_recon=True)

    def _py_plot_all(self, hist_mse, auto_recon=True):
        if hist_mse is not None:
            self._last_hist = hist_mse
        self.py_ax1.clear(); self.py_ax2.clear(); self.py_ax3.clear()
        self._style_py_axes()
        shape = self.shape_name.get()

        if self.circle_img is not None:
            from pythagoras_analog import shape_balance_stats
            st = shape_balance_stats(self.circle_img)
            self.py_ax1.imshow(self.circle_img, cmap="gray", origin="upper",
                               vmin=0, vmax=255)
            self.py_ax1.set_title(
                f"{shape.upper()} train  W={st['n_white']} B={st['n_black']}")
        self.py_ax1.set_xticks([]); self.py_ax1.set_yticks([])

        if self.net is not None and auto_recon:
            x, y, xt, yt, Rh, R, theta, rmse = self.net.reconstruct_shape(
                shape, n_theta=360, size=128, frac=float(self.shape_frac.get()))
            scale = 128 / 2.0
            self.py_ax2.plot(xt / scale, yt / scale, color="#64748b",
                             lw=1.5, ls="--", label="Ideal")
            self.py_ax2.plot(x / scale, y / scale, color="#38bdf8",
                             lw=2.0, label="Net 2-30-1")
            self.py_ax2.set_aspect("equal")
            self.py_ax2.set_xlim(-1.5, 1.5); self.py_ax2.set_ylim(-1.5, 1.5)
            self.py_ax2.legend(facecolor="#1e293b", edgecolor="#334155",
                               labelcolor="#e2e8f0", fontsize=8)
            self.py_ax2.set_title(f"Reconstructed {shape}")
            self.py_ax2.grid(True, color="#1e293b")
            self.py_err.set(f"{rmse:.4f}")
            self.py_status.set(f"{shape.upper()} reconstructed — RMSE={rmse:.4f}")

        if hist_mse is not None:
            self.py_ax3.plot(hist_mse, color="#a78bfa", lw=1.5)
            self.py_ax3.set_yscale("log")
            self.py_ax3.set_title("Training MSE (log)")
            self.py_ax3.set_xlabel("epoch")
            self.py_ax3.grid(True, color="#1e293b")
        self.py_fig.tight_layout(pad=2.0)
        self.py_canvas.draw_idle()

    def _py_quick_demo(self):
        """Analog Quick Test: selected shape + engine, train, reconstruct.

        Mazumdar engine: 300 epochs (~2-3 min, integer ops only).
        MLP ver01 engine: 1000 epochs (fast float baseline).
        """
        self.py_epochs.set(300 if self.engine.get() == "mazumdar" else 1000)
        self.nb.select(1)
        self._py_make_shape()
        self._py_start_train()


    # ===================================================================== #
    #  Tab 3 -- live pixel learning from a 64x64 balanced bitmap
    # ===================================================================== #
    def _build_bmp_tab(self, parent):
        tip = ttk.Frame(parent, padding=(8, 4))
        tip.pack(fill="x")
        ttk.Label(
            tip,
            text="cir.bmp / sqr.bmp / trngl.bmp are 64x64 with exactly "
                 "2048 white == 2048 black.  Random (x,y) probes are drawn in "
                 "real time, normalized x/64 y/64, label white=1 black=0.  "
                 "Press Run for continuous train/test with live overlap plots.",
            style="Tip.TLabel").pack(fill="x")

        body = ttk.Frame(parent, padding=4)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 8))
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        box = ttk.LabelFrame(left, text=" Live Pixel Learning ", padding=10)
        box.pack(fill="x")

        ttk.Label(box, text="Engine", style="Head.TLabel").pack(anchor="w")
        ttk.Radiobutton(box, text="Mazumdar 1995 (no × ÷ exp)",
                        value="mazumdar", variable=self.bmp_engine,
                        command=self._bmp_reset).pack(anchor="w")
        ttk.Radiobutton(box, text="Float MLP baseline",
                        value="mlp", variable=self.bmp_engine,
                        command=self._bmp_reset).pack(anchor="w", pady=(0, 6))

        ttk.Label(box, text="Mode", style="Head.TLabel").pack(anchor="w")
        ttk.Radiobutton(box, text="Train (weights adapt)", value="train",
                        variable=self.bmp_mode).pack(anchor="w")
        ttk.Radiobutton(box, text="Test (weights frozen)", value="test",
                        variable=self.bmp_mode).pack(anchor="w", pady=(0, 6))

        row = ttk.Frame(box)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Probes / step", width=12).pack(side="left")
        ttk.Spinbox(row, from_=32, to=2048, increment=32,
                    textvariable=self.bmp_batch, width=8).pack(side="left")

        self._bmp_run_btn = ttk.Button(box, text="▶  Run Continuous",
                                       style="Accent.TButton",
                                       command=self._bmp_toggle_run)
        self._bmp_run_btn.pack(fill="x", pady=(8, 2))
        ttk.Button(box, text="Step Once",
                   command=self._bmp_step_once).pack(fill="x", pady=2)
        ttk.Button(box, text="Reset Network",
                   command=self._bmp_reset).pack(fill="x", pady=2)
        ttk.Button(box, text="Regenerate .bmp files",
                   command=self._bmp_regenerate).pack(fill="x", pady=2)

        met = ttk.LabelFrame(left, text=" Live Metrics ", padding=10)
        met.pack(fill="x", pady=(10, 0))
        ttk.Label(met, text="Pixel accuracy", style="Head.TLabel").pack(anchor="w")
        ttk.Label(met, textvariable=self.bmp_acc, style="Big.TLabel").pack(anchor="w")
        ttk.Label(met, text="Probes drawn", style="Head.TLabel").pack(anchor="w",
                                                                     pady=(6, 0))
        ttk.Label(met, textvariable=self.bmp_seen, style="Big.TLabel").pack(anchor="w")
        ttk.Label(met, textvariable=self.bmp_status, wraplength=220,
                  background="#232833").pack(anchor="w", pady=(8, 0))

        info = ttk.LabelFrame(left, text=" Sampling protocol ", padding=10)
        info.pack(fill="x", pady=(10, 0))
        ttk.Label(
            info,
            text="x,y ~ U{0..63}\n"
                 "in  = (x/64.0, y/64.0)\n"
                 "tgt = 1 white / 0 black\n\n"
                 "net = 2 -> 30 -> 1\n"
                 "balanced 2048 / 2048",
            justify="left", background="#232833",
            font=("Consolas", 9), foreground="#cbd5e1").pack(anchor="w")

        self.bmp_fig = Figure(figsize=(7.2, 5.8), dpi=100, facecolor="#1a1d23")
        self.bmp_axA = self.bmp_fig.add_subplot(221)
        self.bmp_axB = self.bmp_fig.add_subplot(222)
        self.bmp_axC = self.bmp_fig.add_subplot(223)
        self.bmp_axD = self.bmp_fig.add_subplot(224)
        self._style_bmp_axes()
        # fixed layout: tight_layout on every live frame is far too slow
        self.bmp_fig.subplots_adjust(left=0.07, right=0.98, top=0.93,
                                     bottom=0.09, wspace=0.18, hspace=0.28)
        self.bmp_canvas = FigureCanvasTkAgg(self.bmp_fig, master=right)
        self.bmp_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.after(120, self._bmp_load_shape)

    def _style_bmp_axes(self):
        for ax in (self.bmp_axA, self.bmp_axB, self.bmp_axC, self.bmp_axD):
            ax.set_facecolor("#0f1218")
            ax.tick_params(colors="#94a3b8", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.title.set_color("#e2e8f0")
            ax.title.set_size(9)
            ax.xaxis.label.set_color("#94a3b8")
            ax.yaxis.label.set_color("#94a3b8")

    def _bmp_load_shape(self):
        from shape_bmp import get_shape_image, balance_stats, BMP_NAMES
        shape = self.shape_name.get()
        self.bmp_img = get_shape_image(shape, size=64)
        st = balance_stats(self.bmp_img)
        self.bmp_status.set(
            f"{BMP_NAMES[shape]}  64x64  white={st['n_white']} "
            f"black={st['n_black']}  balanced={st['balanced']}")
        self._bmp_reset()

    def _bmp_regenerate(self):
        from shape_bmp import write_all_bmps
        paths = write_all_bmps(size=64)
        self._bmp_load_shape()
        messagebox.showinfo(
            "Bitmaps written",
            "Regenerated 64x64 bitmaps, each exactly 2048 white / 2048 black:\n\n"
            + "\n".join(paths[k] for k in paths), parent=self)

    def _bmp_reset(self):
        from shape_bmp import PixelStream
        from pixel_learners import make_learner
        was_running = self._bmp_run
        self._bmp_stop()
        if self.bmp_img is None:
            return
        self._bmp_sync_vars()
        self._bmp_learner = make_learner(self._bmp_engine_v, seed=0)
        self._bmp_stream = PixelStream(self.bmp_img, seed=1)
        self._bmp_hist = []
        self._bmp_overlap = []
        self._bmp_grid = None
        self._bmp_grid_id = 0
        self._bmp_overlap_id = -1
        with self._bmp_lock:
            self._bmp_snap = None
        self.bmp_acc.set("—")
        self.bmp_seen.set("0")
        self._bmp_redraw(force_blank=True)
        if was_running:
            self._bmp_start()

    def _bmp_toggle_run(self):
        self._bmp_stop() if self._bmp_run else self._bmp_start()

    def _bmp_sync_vars(self):
        """Copy Tk vars into plain attributes -- main thread only."""
        self._bmp_batch_v = max(32, int(self.bmp_batch.get()))
        self._bmp_mode_v = self.bmp_mode.get()
        self._bmp_engine_v = self.bmp_engine.get()

    def _bmp_start(self):
        if self._bmp_learner is None:
            self._bmp_load_shape()
        self._bmp_sync_vars()
        self._bmp_error = None
        self._bmp_gen += 1
        self._bmp_run = True
        self._bmp_run_btn.configure(text="■  Stop")
        threading.Thread(target=self._bmp_worker, args=(self._bmp_gen,),
                         daemon=True).start()
        self._bmp_job = self.after(200, self._bmp_tick)

    def _bmp_stop(self):
        self._bmp_run = False
        self._bmp_gen += 1              # retires any worker still in flight
        if self._bmp_job is not None:
            self.after_cancel(self._bmp_job)
            self._bmp_job = None
        if hasattr(self, "_bmp_run_btn"):
            self._bmp_run_btn.configure(text="▶  Run Continuous")

    def _bmp_step_once(self):
        if self._bmp_run:
            return
        if self._bmp_learner is None:
            self._bmp_load_shape()
        self._bmp_sync_vars()
        self._bmp_one_step(self._bmp_batch_v, self._bmp_mode_v)
        self._bmp_tick(reschedule=False)

    def _bmp_one_step(self, n: int, mode: str, want_grid: bool = True):
        """One probe batch: sample -> (train) -> score -> publish snapshot.

        Runs on the worker thread, so it takes plain arguments and touches no
        Tk variable; results are handed to the UI through _bmp_snap.

        The 64x64 output map costs far more than a training step, so it is
        refreshed only every BMP_GRID_EVERY steps and otherwise reused. That
        keeps the probe rate high while the picture still updates smoothly.
        """
        X, y, px, py = self._bmp_stream.next_batch(n)
        if mode == "train":
            self._bmp_learner.train_chunk(X, y)
        acc = self._bmp_learner.accuracy(X, y)
        if want_grid or self._bmp_grid is None:
            self._bmp_grid = self._bmp_learner.predict_grid(
                64, **({"T": self.BMP_GRID_T}
                       if self._bmp_engine_v == "mazumdar" else {}))
            self._bmp_grid_id += 1
        seen = self._bmp_stream.n_drawn
        self._bmp_hist.append((seen, acc))
        with self._bmp_lock:
            self._bmp_snap = dict(grid=self._bmp_grid, acc=acc, seen=seen,
                                  px=px, py=py, lab=y,
                                  grid_id=self._bmp_grid_id,
                                  hist=list(self._bmp_hist))

    def _bmp_worker(self, gen: int):
        i = 0
        while self._bmp_run and gen == self._bmp_gen:
            try:
                self._bmp_one_step(self._bmp_batch_v, self._bmp_mode_v,
                                   want_grid=(i % self.BMP_GRID_EVERY == 0))
                i += 1
            except Exception as e:
                self._bmp_error = e
                self._bmp_run = False
                return

    def _bmp_tick(self, reschedule: bool = True):
        if self._bmp_error is not None:
            err, self._bmp_error = self._bmp_error, None
            self._bmp_stop()
            self.bmp_status.set(f"Error: {err}")
            messagebox.showerror("Live learning error", str(err), parent=self)
            return
        self._bmp_sync_vars()
        with self._bmp_lock:
            snap = self._bmp_snap
        if snap is not None:
            self.bmp_acc.set(f"{snap['acc'] * 100:.1f}%")
            self.bmp_seen.set(f"{snap['seen']:,}")
            self._bmp_redraw(snap)
        if reschedule and self._bmp_run:
            self._bmp_job = self.after(400, self._bmp_tick)

    def _bmp_redraw(self, snap=None, force_blank: bool = False):
        for ax in (self.bmp_axA, self.bmp_axB, self.bmp_axC, self.bmp_axD):
            ax.clear()
        self._style_bmp_axes()
        shape = self.shape_name.get()
        eng = "Mazumdar" if self.bmp_engine.get() == "mazumdar" else "float MLP"
        mode = self.bmp_mode.get()

        # A -- source bitmap with the probes of the current step
        if self.bmp_img is not None:
            self.bmp_axA.imshow(self.bmp_img, cmap="gray", origin="upper",
                                vmin=0, vmax=255, interpolation="nearest")
            if snap is not None:
                lab = snap["lab"].astype(bool)
                self.bmp_axA.scatter(snap["px"][lab], snap["py"][lab], s=5,
                                     c="#22c55e", linewidths=0, label="white=1")
                self.bmp_axA.scatter(snap["px"][~lab], snap["py"][~lab], s=5,
                                     c="#f43f5e", linewidths=0, label="black=0")
            self.bmp_axA.set_title(f"{shape} 64x64 — random probes")
        self.bmp_axA.set_xticks([]); self.bmp_axA.set_yticks([])

        # B -- live analog output map
        if snap is not None:
            self.bmp_axB.imshow(snap["grid"], cmap="viridis", origin="upper",
                                vmin=0, vmax=1, interpolation="nearest")
            self.bmp_axB.set_title(f"Net output ({eng}, {mode})")
        else:
            self.bmp_axB.set_title("Net output — press Run")
        self.bmp_axB.set_xticks([]); self.bmp_axB.set_yticks([])

        # C -- OVERLAP: ideal edge + fading history of learned edges
        if self.bmp_img is not None:
            self.bmp_axC.contour(self.bmp_img.astype(float), levels=[127.5],
                                 colors="#facc15", linewidths=2.0)
        if snap is not None:
            if snap["grid_id"] != self._bmp_overlap_id:
                self._bmp_overlap_id = snap["grid_id"]
                self._bmp_overlap.append(snap["grid"])
                self._bmp_overlap = self._bmp_overlap[-self.BMP_OVERLAP_MAX:]
            k = max(len(self._bmp_overlap), 1)
            for i, g in enumerate(self._bmp_overlap):
                self.bmp_axC.contour(g, levels=[0.5], colors="#38bdf8",
                                     linewidths=0.8 + 1.4 * (i + 1) / k,
                                     alpha=0.12 + 0.88 * ((i + 1) / k) ** 2)
        self.bmp_axC.set_xlim(0, 63); self.bmp_axC.set_ylim(0, 63)
        self.bmp_axC.invert_yaxis()
        self.bmp_axC.set_aspect("equal")
        self.bmp_axC.set_xticks([]); self.bmp_axC.set_yticks([])
        self.bmp_axC.set_title("Overlap: ideal (yellow) vs learned edge")

        # D -- continuous accuracy trace
        hist = snap["hist"] if snap else []
        if hist:
            xs = [h[0] for h in hist]
            ys = [h[1] * 100 for h in hist]
            col = "#34d399" if mode == "train" else "#f472b6"
            self.bmp_axD.plot(xs, ys, color=col, lw=1.4)
            self.bmp_axD.axhline(50, color="#475569", ls=":", lw=1)
            self.bmp_axD.set_ylim(40, 102)
        self.bmp_axD.set_title(f"Live {mode} accuracy (%)")
        self.bmp_axD.set_xlabel("probes drawn")
        self.bmp_axD.grid(True, color="#1e293b")

        self.bmp_canvas.draw_idle()

    def _build_controls(self, parent):
        box = ttk.LabelFrame(parent, text=" Analog Knobs ", padding=10)
        box.pack(fill="x")

        def row(label, var, lo, hi, fmt="{:.2f}"):
            f = ttk.Frame(box)
            f.pack(fill="x", pady=3)
            ttk.Label(f, text=label, width=14).pack(side="left")
            ttk.Scale(f, from_=lo, to=hi, variable=var, orient="horizontal",
                      length=160, command=lambda _=None: self._on_knob()
                      ).pack(side="left", fill="x", expand=True)
            lab = ttk.Label(f, width=7)
            lab.pack(side="left")

            def sync(*_):
                try:
                    lab.configure(text=fmt.format(var.get()))
                except Exception:
                    pass
            var.trace_add("write", sync)
            sync()

        row("Vin (V)", self.vin, -2.0, 2.0)
        row("Temp T (K)", self.temp, 200.0, 400.0, "{:.0f}")
        row("Noise σ0 (V)", self.sigma0, 0.05, 1.0)
        row("Mismatch (V)", self.mismatch, 0.0, 0.3)

        f = ttk.Frame(box)
        f.pack(fill="x", pady=3)
        ttk.Label(f, text="Samples / pt", width=14).pack(side="left")
        ttk.Spinbox(f, from_=64, to=2048, textvariable=self.n_samples,
                    width=8).pack(side="left")

        btns = ttk.Frame(box)
        btns.pack(fill="x", pady=(10, 4))
        ttk.Button(btns, text="Refresh Curves",
                   command=self._refresh_curves).pack(fill="x", pady=2)
        ttk.Button(btns, text="Live Fire (bitstream)",
                   command=self._toggle_live).pack(fill="x", pady=2)
        ttk.Button(btns, text="Monte-Carlo PVT",
                   command=self._run_monte_carlo).pack(fill="x", pady=2)
        ttk.Button(btns, text="Single-Shot Decision",
                   command=self._single_shot).pack(fill="x", pady=2)

        met = ttk.LabelFrame(parent, text=" Circuit Metrics ", padding=10)
        met.pack(fill="x", pady=(10, 0))
        ttk.Label(met, textvariable=self.metrics_var, justify="left",
                  background="#232833", font=("Consolas", 9)).pack(anchor="w")

        live = ttk.LabelFrame(parent, text=" Live Output ", padding=10)
        live.pack(fill="x", pady=(10, 0))
        ttk.Label(live, text="Emp. P(Y=1) at Vin", style="Head.TLabel").pack(anchor="w")
        ttk.Label(live, textvariable=self.live_rate, style="Big.TLabel").pack(anchor="w")
        self.bit_canvas = tk.Canvas(live, height=48, bg="#0f1218",
                                    highlightthickness=0, width=220)
        self.bit_canvas.pack(fill="x", pady=6)
        ttk.Label(live, text="Bitstream (last 40 decisions)",
                  foreground="#94a3b8", background="#232833").pack(anchor="w")

        info = ttk.LabelFrame(parent, text=" Circuit Idea ", padding=10)
        info.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(
            info,
            text=("Vin ──► +\\ comparator\n"
                  "         }──► Y∈{0,1}\n"
                  "Vth+n ─► -/\n\n"
                  "n(t) ~ thermal noise\n"
                  "P(Y=1) ≈ sigmoid(Vin/σ)\n"
                  "= Mazumdar Eq.3 analog"),
            justify="left", background="#232833",
            font=("Consolas", 9), foreground="#cbd5e1").pack(anchor="w")

    def _build_plots(self, parent):
        self.fig = Figure(figsize=(7.2, 5.8), dpi=100, facecolor="#1a1d23")
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor("#0f1218")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.title.set_color("#e2e8f0")
            ax.xaxis.label.set_color("#94a3b8")
            ax.yaxis.label.set_color("#94a3b8")
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _params(self) -> AnalogNeuronParams:
        return AnalogNeuronParams(
            v0=0.0,
            sigma0=float(self.sigma0.get()),
            T=float(self.temp.get()),
            mismatch=float(self.mismatch.get()),
            seed=0,
        )

    def _on_knob(self):
        # light debounce via after
        if hasattr(self, "_knob_job") and self._knob_job:
            self.after_cancel(self._knob_job)
        self._knob_job = self.after(180, self._refresh_curves)

    def _refresh_curves(self):
        p = self._params()
        ns = int(self.n_samples.get())
        data = sweep_transfer(p, n_pts=81, n_samples=max(64, ns))
        vin, pf, pr = data["vin"], data["p_fire"], data["p_rev"]
        ideal, ideal_d = data["ideal"], data["ideal_d"]
        m = data["metrics"]
        self.metrics_var.set(
            f"σ(T) = {m['sigma_V']:.3f} V\n"
            f"Vth  = {m['vth_static']:+.3f} V\n"
            f"Area ≈ {m['area_um2']:.1f} µm²\n"
            f"E/dec≈ {m['energy_pJ']:.4f} pJ\n"
            f"Vdd   = {m['vdd']:.1f} V"
        )

        # mark current Vin
        vnow = float(self.vin.get())
        n = AnalogStochasticNeuron(p)
        p_now = float(n.fire_prob(np.array([vnow]), n_samples=max(128, ns))[0])
        self.live_rate.set(f"{p_now * 100:.1f}%")

        self.ax1.clear()
        self.ax2.clear()
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor("#0f1218")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.title.set_color("#e2e8f0")
            ax.xaxis.label.set_color("#94a3b8")
            ax.yaxis.label.set_color("#94a3b8")
            ax.grid(True, color="#1e293b", linewidth=0.6)

        self.ax1.plot(vin, ideal, color="#64748b", lw=2, label="Ideal sigmoid")
        self.ax1.plot(vin, pf, color="#38bdf8", lw=2, label="Analog P(Y=1)")
        self.ax1.axvline(vnow, color="#fbbf24", ls="--", lw=1, label=f"Vin={vnow:.2f}")
        self.ax1.scatter([vnow], [p_now], color="#fbbf24", zorder=5)
        self.ax1.set_ylim(-0.05, 1.05)
        self.ax1.set_ylabel("P(fire)")
        self.ax1.set_title("Forward characteristic (analog ≈ Eq.3)")
        self.ax1.legend(facecolor="#1e293b", edgecolor="#334155",
                        labelcolor="#e2e8f0", fontsize=8)

        self.ax2.plot(vin, ideal_d, color="#64748b", lw=2, label="Ideal σ'(norm)")
        self.ax2.plot(vin, pr, color="#a78bfa", lw=2, label="Analog R(X)")
        self.ax2.axvline(vnow, color="#fbbf24", ls="--", lw=1)
        self.ax2.set_ylim(-0.05, 1.05)
        self.ax2.set_xlabel("Vin (V)")
        self.ax2.set_ylabel("P(learn gate)")
        self.ax2.set_title("Reverse characteristic (analog ≈ Eq.4)")
        self.ax2.legend(facecolor="#1e293b", edgecolor="#334155",
                        labelcolor="#e2e8f0", fontsize=8)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def _single_shot(self):
        n = AnalogStochasticNeuron(self._params())
        y = int(n.fire(float(self.vin.get()), n_samples=1)[0])
        thr = n.sample_threshold(1)[0]
        self._bitstream.append(y)
        self._bitstream = self._bitstream[-40:]
        self._draw_bits()
        messagebox.showinfo(
            "Single-Shot",
            f"Vin = {self.vin.get():.3f} V\n"
            f"Sampled Vth = {thr:.3f} V\n"
            f"Comparator Y = {y}\n\n"
            f"(Y=1 if Vin > Vth)",
            parent=self)

    def _toggle_live(self):
        if self._anim_job is not None:
            self.after_cancel(self._anim_job)
            self._anim_job = None
            return
        self._bitstream = []
        self._live_tick()

    def _live_tick(self):
        n = AnalogStochasticNeuron(self._params())
        y = int(n.fire(float(self.vin.get()), n_samples=1)[0])
        self._bitstream.append(y)
        self._bitstream = self._bitstream[-40:]
        rate = float(np.mean(self._bitstream)) if self._bitstream else 0.0
        self.live_rate.set(f"{rate * 100:.1f}%  ({len(self._bitstream)} samples)")
        self._draw_bits()
        self._anim_job = self.after(40, self._live_tick)

    def _draw_bits(self):
        cv = self.bit_canvas
        cv.delete("all")
        w = max(cv.winfo_width(), 220)
        h = 48
        n = len(self._bitstream)
        if n == 0:
            return
        bw = max(2, w // 40)
        for i, b in enumerate(self._bitstream):
            x0 = i * bw
            color = "#22c55e" if b else "#334155"
            cv.create_rectangle(x0, 8, x0 + bw - 1, h - 8, fill=color, outline="")

    def _run_monte_carlo(self):
        p = self._params()
        vin = np.linspace(-2, 2, 61)
        neuron = AnalogStochasticNeuron(p)
        curves = neuron.monte_carlo_curves(vin, n_devices=35, n_samples=200)
        mean = curves.mean(axis=0)
        lo = np.percentile(curves, 10, axis=0)
        hi = np.percentile(curves, 90, axis=0)

        self.ax1.clear()
        self.ax1.set_facecolor("#0f1218")
        self.ax1.tick_params(colors="#94a3b8")
        for spine in self.ax1.spines.values():
            spine.set_color("#334155")
        self.ax1.grid(True, color="#1e293b", linewidth=0.6)
        for c in curves[::3]:
            self.ax1.plot(vin, c, color="#334155", lw=0.7, alpha=0.7)
        self.ax1.fill_between(vin, lo, hi, color="#0ea5e9", alpha=0.25,
                              label="10–90% band")
        self.ax1.plot(vin, mean, color="#38bdf8", lw=2.5, label="Mean P(Y=1)")
        ideal = AnalogStochasticNeuron.ideal_sigmoid(vin, scale=neuron.sigma)
        self.ax1.plot(vin, ideal, color="#fbbf24", lw=1.5, ls="--", label="Ideal")
        self.ax1.set_ylim(-0.05, 1.05)
        self.ax1.set_title(f"Monte-Carlo PVT (N=35, mismatch σ={p.mismatch:.2f} V)")
        self.ax1.set_ylabel("P(fire)")
        self.ax1.legend(facecolor="#1e293b", edgecolor="#334155",
                        labelcolor="#e2e8f0", fontsize=8)
        self.ax1.title.set_color("#e2e8f0")
        self.canvas.draw_idle()
        messagebox.showinfo(
            "Monte-Carlo PVT",
            f"35 mismatched analog neurons simulated.\n"
            f"Mismatch σ = {p.mismatch:.3f} V\n"
            f"Noise σ(T) = {neuron.sigma:.3f} V\n\n"
            "Shaded band = 10th–90th percentile of P(Y=1).",
            parent=self)

    def _on_close(self):
        self._closed = True
        if self._anim_job is not None:
            self.after_cancel(self._anim_job)
            self._anim_job = None
        self._bmp_stop()
        self.destroy()
