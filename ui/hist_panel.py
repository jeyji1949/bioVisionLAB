import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from core.histogram import (compute_histogram, stretch_contrast,
                             equalize_histogram, threshold_binary, image_stats)

WHITE   = "#FFFFFF"
BG      = "#F4F6F9"
BORDER  = "#DDE2EC"
COLOR   = "#6C47CC"
COLOR_L = "#F0EDFB"
ACCENT  = "#1A6FBF"
TEXT    = "#1C2333"
TEXT2   = "#5A6478"
TEXT3   = "#9BA5B8"

BIO_INTERP = {
    "Aucune":    "Distribution brute — visualiser la répartition des niveaux de gris",
    "stretch":   "Étirement → améliore le contraste des images sous/surexposées",
    "equalize":  "Égalisation → révèle les détails cachés dans les zones sombres",
    "threshold": "Seuillage → isoler les cellules du fond",
}


class HistPanel(tk.Frame):

    def __init__(self, parent, state):
        super().__init__(parent, bg=BG)
        self.state = state
        self._photos = {}
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._sidebar()
        self._main_area()

    def _sidebar(self):
        sb = tk.Frame(self, bg=WHITE, width=240)
        sb.grid(row=0, column=0, sticky="ns")
        sb.grid_propagate(False)
        tk.Frame(sb, bg=BORDER, width=1).place(relx=1, rely=0, relheight=1)

        title_bar = tk.Frame(sb, bg=COLOR_L, height=56)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)
        tk.Frame(title_bar, bg=COLOR, width=3).pack(side="left", fill="y")
        col = tk.Frame(title_bar, bg=COLOR_L)
        col.pack(side="left", padx=12, pady=10)
        tk.Label(col, text="Étape 2 — Histogramme", bg=COLOR_L, fg=COLOR,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Label(col, text="Analyse & amélioration du contraste", bg=COLOR_L, fg=COLOR,
                 font=("Helvetica", 8)).pack(anchor="w")

        body = tk.Frame(sb, bg=WHITE)
        body.pack(fill="both", expand=True)

        self.interp_frame = tk.Frame(body, bg=COLOR_L)
        self.interp_frame.pack(fill="x", padx=12, pady=(14, 0))
        tk.Label(self.interp_frame, text="Interprétation biologique", bg=COLOR_L,
                 fg=COLOR, font=("Helvetica", 8, "bold")).pack(anchor="w", padx=8, pady=(5, 0))
        self.interp_lbl = tk.Label(self.interp_frame, text="—", bg=COLOR_L, fg=COLOR,
                                    font=("Helvetica", 8), wraplength=200, justify="left")
        self.interp_lbl.pack(anchor="w", padx=8, pady=(2, 6))

        self._section(body, "OPÉRATION")
        self.op_var = tk.StringVar(value="Aucune")
        for label, val in [("Affichage seul", "Aucune"),
                            ("Étirement contraste", "stretch"),
                            ("Égalisation", "equalize"),
                            ("Seuillage", "threshold")]:
            tk.Radiobutton(body, text=label, variable=self.op_var, value=val,
                           bg=WHITE, fg=TEXT, selectcolor=COLOR_L,
                           activebackground=WHITE, font=("Helvetica", 10),
                           command=self._on_op_change).pack(anchor="w", padx=20, pady=2)

        self._section(body, "SEUIL")
        self.thresh_var = tk.IntVar(value=127)
        self._slider(body, "Valeur", self.thresh_var, 0, 255, 1,
                     fmt=lambda v: str(int(v)))

        self._section(body, "BINS HISTOGRAMME")
        self.bins_var = tk.IntVar(value=256)
        self._slider(body, "Bins", self.bins_var, 16, 256, 16,
                     fmt=lambda v: str(int(v)))

        self._sep(body)
        self._section(body, "STATISTIQUES")
        self.stat_vars = {}
        for key, lbl in [("min", "Min"), ("max", "Max"), ("mean", "Moyenne"),
                          ("std", "Écart-type"), ("entropy", "Entropie")]:
            row = tk.Frame(body, bg=WHITE)
            row.pack(fill="x", padx=14, pady=1)
            tk.Label(row, text=lbl, bg=WHITE, fg=TEXT2,
                     font=("Helvetica", 9), width=11, anchor="w").pack(side="left")
            v = tk.Label(row, text="—", bg=WHITE, fg=COLOR,
                         font=("Helvetica", 9, "bold"))
            v.pack(side="right")
            self.stat_vars[key] = v

        tk.Button(body, text="Calculer", bg=COLOR, fg=WHITE,
                  font=("Helvetica", 10, "bold"), relief="flat", cursor="hand2",
                  pady=7, command=self.run).pack(fill="x", padx=12, pady=(14, 4))
        tk.Button(body, text="Envoyer vers Fourier →", bg=BG, fg=ACCENT,
                  font=("Helvetica", 9), relief="flat", cursor="hand2",
                  pady=5, command=self._send_next).pack(fill="x", padx=12, pady=(0, 12))

    def _main_area(self):
        main = tk.Frame(self, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=3)
        main.rowconfigure(1, weight=2)

        for col, (title, tag, key) in enumerate([
            ("Image source (M1 ou originale)", "INPUT",        "orig"),
            ("Résultat",                        "RÉSULTAT M2", "result"),
        ]):
            p = tk.Frame(main, bg=WHITE)
            p.grid(row=0, column=col, sticky="nsew",
                   padx=(10 if col == 0 else 4, 4 if col == 0 else 10), pady=(10, 4))
            p.columnconfigure(0, weight=1)
            p.rowconfigure(1, weight=1)
            hdr = tk.Frame(p, bg=COLOR_L, height=36)
            hdr.grid(row=0, column=0, sticky="ew")
            hdr.grid_propagate(False)
            tk.Label(hdr, text=title, bg=COLOR_L, fg=COLOR,
                     font=("Helvetica", 10, "bold")).pack(side="left", padx=12, pady=8)
            tk.Label(hdr, text=tag, bg=COLOR_L, fg=COLOR,
                     font=("Helvetica", 8)).pack(side="right", padx=12)
            cv = tk.Canvas(p, bg="#E8EDF4", highlightthickness=0)
            cv.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
            setattr(self, f"cv_{key}", cv)
            ph = tk.Label(cv, bg="#E8EDF4", fg=TEXT3, font=("Helvetica", 9),
                          text="En attente…")
            ph.place(relx=0.5, rely=0.5, anchor="center")
            setattr(self, f"ph_{key}", ph)

        hp = tk.Frame(main, bg=WHITE)
        hp.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(4, 10))
        hp.columnconfigure(0, weight=1)
        hp.rowconfigure(1, weight=1)
        hdr2 = tk.Frame(hp, bg=COLOR_L, height=34)
        hdr2.grid(row=0, column=0, sticky="ew")
        hdr2.grid_propagate(False)
        tk.Label(hdr2, text="Histogramme des niveaux de gris", bg=COLOR_L, fg=COLOR,
                 font=("Helvetica", 10, "bold")).pack(side="left", padx=12, pady=7)
        self.hist_title = tk.Label(hdr2, text="—", bg=COLOR_L, fg=COLOR,
                                    font=("Helvetica", 8))
        self.hist_title.pack(side="left", pady=7)

        fig_frame = tk.Frame(hp, bg=WHITE)
        fig_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.fig, self.ax = plt.subplots(figsize=(8, 2.0), facecolor=WHITE)
        self.ax.set_facecolor(BG)
        self.ax.tick_params(colors=TEXT3, labelsize=7)
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)
        self.fig.tight_layout(pad=1.0)
        self.hist_cv = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.hist_cv.get_tk_widget().pack(fill="both", expand=True)

    def _sep(self, p):
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=10, pady=8)

    def _section(self, p, text):
        tk.Label(p, text=text, bg=WHITE, fg=TEXT3,
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=14, pady=(8, 3))

    def _slider(self, p, label, var, lo, hi, res, fmt=None):
        row = tk.Frame(p, bg=WHITE)
        row.pack(fill="x", padx=14, pady=1)
        tk.Label(row, text=label, bg=WHITE, fg=TEXT2, font=("Helvetica", 9),
                 width=10, anchor="w").pack(side="left")
        vl = tk.Label(row, text=fmt(var.get()) if fmt else str(var.get()),
                      bg=WHITE, fg=ACCENT, font=("Helvetica", 9, "bold"), width=6)
        vl.pack(side="right")
        tk.Scale(p, variable=var, from_=lo, to=hi, resolution=res,
                 orient="horizontal", bg=WHITE, troughcolor=BORDER,
                 highlightthickness=0, showvalue=False,
                 command=lambda v: vl.config(
                     text=fmt(float(v)) if fmt else str(int(float(v))))
                 ).pack(fill="x", padx=14)

    def _display(self, pixels, key):
        cv = getattr(self, f"cv_{key}")
        cv.update_idletasks()
        cw, ch = cv.winfo_width(), cv.winfo_height()
        if cw < 10:
            cw, ch = 300, 240
        img   = Image.fromarray(pixels.astype(np.uint8), mode="L")
        sc    = min(cw / img.width, ch / img.height, 1.0)
        img   = img.resize((max(1, int(img.width * sc)),
                            max(1, int(img.height * sc))), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        cv.delete("all")
        cv.create_image(cw // 2, ch // 2, anchor="center", image=photo)
        getattr(self, f"ph_{key}").place_forget()
        self._photos[key] = photo

    def _draw_hist(self, pixels, bins, title):
        hist = compute_histogram(pixels, bins)
        self.ax.clear()
        self.ax.set_facecolor(BG)
        x = np.linspace(0, 255, bins)
        self.ax.bar(x, hist, width=256 / bins, color=COLOR, alpha=0.72, linewidth=0)
        self.ax.set_xlim(0, 255)
        self.ax.tick_params(colors=TEXT3, labelsize=7)
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)
        self.ax.set_ylabel("Fréquence", fontsize=7, color=TEXT2)
        self.hist_cv.draw()
        self.hist_title.config(text=title)

    def _on_op_change(self, *_):
         self.interp_lbl.config(text=BIO_INTERP.get(self.op_var.get(), "—"))

    def run(self):
        if not self.state.has_image():
            return
        pix  = self.state.pipeline_input(1)
        op   = self.op_var.get()
        bins = self.bins_var.get()
        if op == "stretch":
            result, title = stretch_contrast(pix), "Après étirement de contraste"
        elif op == "equalize":
            result, title = equalize_histogram(pix), "Après égalisation"
        elif op == "threshold":
            t = self.thresh_var.get()
            result, title = threshold_binary(pix, t), f"Seuillage t={t}"
        else:
            result, title = pix, "Distribution originale"
        self.state.m2_result = result
        self._display(result, "result")
        self._draw_hist(result, bins, title)
        stats = image_stats(result)
        for k, lbl in self.stat_vars.items():
            v = stats.get(k, "—")
            lbl.config(text=str(v) + (" bits" if k == "entropy" else ""))
        self.interp_lbl.config(text=BIO_INTERP.get(op, "—"))
        self.state.m2_log = {
            "operation": op,
            "bins":      bins,
            "seuil":     self.thresh_var.get() if op == "threshold" else "—",
            "min":       stats.get("min", "—"),
            "max":       stats.get("max", "—"),
            "mean":      stats.get("mean", "—"),
            "std":       stats.get("std", "—"),
            "entropy":   stats.get("entropy", "—"),
        }
        self.state.mark_step(1)

    def _send_next(self):
        if self.state.m2_result is not None:
            self.state.mark_step(1)

    def refresh_image(self):
        if self.state.has_image():
            pix = self.state.pipeline_input(1)
            self._display(pix, "orig")
            self._draw_hist(pix, 256, "Distribution — entrée M2")
            stats = image_stats(pix)
            for k, lbl in self.stat_vars.items():
                v = stats.get(k, "—")
                lbl.config(text=str(v) + (" bits" if k == "entropy" else ""))