from pathlib import Path
import re

p = Path(r"d:\_July2026\SENA_Project\sena2026\sim\analog_demo.py")
text = p.read_text(encoding="utf-8")

build_new = '''
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
        self.nb.add(tab_neuron, text="1) Stochastic Neuron")
        self.nb.add(tab_pythag, text="2) Pythagoras 2-30-1")

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
        self.nb.select(1)

'''

text, n = re.subn(
    r"    def _build\(self\):.*?    def _build_pythagoras_tab\(self, parent\):",
    build_new + "\n    def _build_pythagoras_tab(self, parent):",
    text, count=1, flags=re.S)
print("build replaced", n)

pythag_tab_new = '''
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
        ttk.Label(box, text="2 -> 30 -> 1\\nanalog in / out (normalized)",
                  background="#232833", font=("Consolas", 11),
                  foreground="#34d399").pack(anchor="w", pady=(0, 8))

        row = ttk.Frame(box)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Epochs", width=10).pack(side="left")
        ttk.Spinbox(row, from_=50, to=5000, textvariable=self.py_epochs,
                    width=8).pack(side="left")

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
            text="h = sqrt(a^2 + b^2)\\n"
                 "h_norm = h / sqrt(2)\\n\\n"
                 "Shapes: Cir / Sqr / Trngl\\n"
                 "(balanced B/W pixels)",
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

'''
# fix double-escaped newlines in label texts
pythag_tab_new = pythag_tab_new.replace("\\n", "\n")

text, n = re.subn(
    r"    def _build_pythagoras_tab\(self, parent\):.*?    def _style_py_axes\(self\):",
    pythag_tab_new + "    def _style_py_axes(self):",
    text, count=1, flags=re.S)
print("pythag tab replaced", n)

methods_new = r'''
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
        self.nb.select(1)
        self._py_make_shape()

    def _py_make_shape(self):
        from pythagoras_analog import make_shape_image, shape_balance_stats
        shape = self.shape_name.get()
        self.circle_img = make_shape_image(shape, size=128)
        st = shape_balance_stats(self.circle_img)
        self.py_status.set(
            f"{shape.upper()}  white={st['n_white']} black={st['n_black']}  "
            f"|d|={st['balance_err']}  (target equal)")
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
        epochs = int(self.py_epochs.get())
        shape = self.shape_name.get()
        self.py_status.set(f"Training 2-30-1 on {shape} for {epochs} epochs...")
        threading.Thread(target=self._py_train_worker, args=(epochs,),
                         daemon=True).start()

    def _py_train_worker(self, epochs: int):
        from pythagoras_analog import samples_from_shape_image, AnalogMLP2301
        shape = self.shape_name.get()
        try:
            X, y = samples_from_shape_image(self.circle_img, n_extra=1000, seed=0)
            net = AnalogMLP2301(seed=0, lr=0.08)

            def cb(ep, mse):
                self.after(0, lambda e=ep, m=mse: (
                    self.py_mse.set(f"{m:.5f}"),
                    self.py_status.set(f"Training {shape}... epoch {e}/{epochs}")))

            hist_mse = net.train(X, y, epochs=epochs, batch=64, callback=cb)
            self.net = net
            *_, rmse = net.reconstruct_shape(shape, n_theta=360, size=128)

            def done():
                self._py_busy = False
                self.py_mse.set(f"{hist_mse[-1]:.5f}")
                self.py_err.set(f"{rmse:.4f}")
                self.py_status.set(f"Trained on {shape}. Reconstructing...")
                self._py_plot_all(hist_mse, auto_recon=True)
                messagebox.showinfo(
                    "Analog Quick Test",
                    f"Shape: {shape.upper()}\n"
                    f"Epochs: {epochs}\n"
                    f"Final MSE: {hist_mse[-1]:.5f}\n"
                    f"Shape RMSE: {rmse:.4f}\n\n"
                    f"{'PASS' if hist_mse[-1] < 0.01 and rmse < 0.25 else 'CHECK'}",
                    parent=self)
            self.after(0, done)
        except Exception as e:
            self.after(0, lambda: self._py_fail(e))

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
                shape, n_theta=360, size=128)
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
        """Analog Quick Test: selected shape, 1000 epochs, reconstruct."""
        self.py_epochs.set(1000)
        self.nb.select(1)
        self._py_make_shape()
        self._py_start_train()

'''

text, n = re.subn(
    r"    def _py_make_circle\(self\):.*?    def _build_controls\(self, parent\):",
    methods_new + "\n    def _build_controls(self, parent):",
    text, count=1, flags=re.S)
print("methods replaced", n)
print("highlight present", "_highlight_shape_btn" in text)

p.write_text(text, encoding="utf-8", newline="\n")
print("wrote", p)
