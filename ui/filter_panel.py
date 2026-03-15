import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk

from core.filters import (mean_filter, gaussian_filter, median_filter,
                           laplacian_filter, add_salt_pepper,
                           add_gaussian_noise, NOISE_FILTER_MAP)
from core.histogram import to_grayscale

BG      = "#F4F6F9"
WHITE   = "#FFFFFF"
BORDER  = "#DDE2EC"
COLOR   = "#0F9D8E"
COLOR_L = "#E6F7F5"
ACCENT  = "#1A6FBF"
TEXT    = "#1C2333"
TEXT2   = "#5A6478"
TEXT3   = "#9BA5B8"
OK      = "#1A8754"
OK_L    = "#E8F6EF"
WARN    = "#B45309"
WARN_L  = "#FEF9EE"

FILTER_FNS = {
    "Médian":    lambda p, k, s: median_filter(p, k),
    "Gaussien":  lambda p, k, s: gaussian_filter(p, k, s),
    "Moyenneur": lambda p, k, s: mean_filter(p, k),
    "Laplacien": lambda p, k, s: laplacian_filter(p),
}

BIO_CONTEXT = {
    "Poivre & Sel": "Artefact typique des capteurs CCD à faible luminosité",
    "Gaussien":     "Bruit thermique du capteur ou vibrations mécaniques",
    "Aucun bruit":  "Image microscopique brute — appliquer un filtre doux",
}


class FilterPanel(tk.Frame):

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
        tk.Label(col, text="Étape 1 — Prétraitement", bg=COLOR_L, fg=COLOR,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Label(col, text="Débruitage de l'échantillon", bg=COLOR_L, fg=COLOR,
                 font=("Helvetica", 8)).pack(anchor="w")

        body = tk.Frame(sb, bg=WHITE)
        body.pack(fill="both", expand=True)

        self.ctx_frame = tk.Frame(body, bg=WARN_L)
        self.ctx_frame.pack(fill="x", padx=12, pady=(14, 0))
        tk.Label(self.ctx_frame, text="Contexte microscopique", bg=WARN_L, fg=WARN,
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=8, pady=(6, 0))
        self.ctx_lbl = tk.Label(self.ctx_frame, text="—", bg=WARN_L, fg=WARN,
                                 font=("Helvetica", 8), wraplength=200, justify="left")
        self.ctx_lbl.pack(anchor="w", padx=8, pady=(2, 6))

        self._section(body, "TYPE DE BRUIT")
        self.noise_var = tk.StringVar(value="Poivre & Sel")
        for n in NOISE_FILTER_MAP:
            tk.Radiobutton(body, text=n, variable=self.noise_var, value=n,
                           bg=WHITE, fg=TEXT, selectcolor=COLOR_L,
                           activebackground=WHITE, font=("Helvetica", 10),
                           command=self._on_noise_change).pack(anchor="w", padx=20, pady=2)

        self._section(body, "INTENSITÉ")
        self.amount_var = tk.DoubleVar(value=0.05)
        self._slider(body, "Quantité", self.amount_var, 0.01, 0.20, 0.01,
                     fmt=lambda v: f"{v*100:.0f}%")

        tk.Button(body, text="Ajouter le bruit", bg=COLOR, fg=WHITE,
                  font=("Helvetica", 10, "bold"), relief="flat", cursor="hand2",
                  pady=7, command=self._apply_noise).pack(fill="x", padx=12, pady=(12, 4))

        self._sep(body)
        self._section(body, "FILTRE RECOMMANDÉ")

        self.rec_badge = tk.Frame(body, bg=OK_L)
        self.rec_badge.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(self.rec_badge, text="✓ Recommandé :", bg=OK_L, fg=OK,
                 font=("Helvetica", 8, "bold")).pack(side="left", padx=8, pady=5)
        self.rec_lbl = tk.Label(self.rec_badge, text="Médian", bg=OK_L, fg=OK,
                                 font=("Helvetica", 8))
        self.rec_lbl.pack(side="left", pady=5)

        self.filter_var = tk.StringVar(value="Médian")
        self.filter_rbs = {}
        for fname in FILTER_FNS:
            rb = tk.Radiobutton(body, text=fname, variable=self.filter_var, value=fname,
                                bg=WHITE, fg=TEXT, selectcolor=COLOR_L,
                                activebackground=WHITE, font=("Helvetica", 10))
            rb.pack(anchor="w", padx=20, pady=2)
            self.filter_rbs[fname] = rb

        self._section(body, "PARAMÈTRES")
        self.ksize_var = tk.IntVar(value=3)
        self._slider(body, "Noyau", self.ksize_var, 3, 11, 2,
                     fmt=lambda v: f"{int(v)}×{int(v)}")
        self.sigma_var = tk.DoubleVar(value=1.0)
        self._slider(body, "Sigma", self.sigma_var, 0.5, 5.0, 0.5,
                     fmt=lambda v: f"{v:.1f}")

        tk.Button(body, text="Appliquer le filtre", bg=ACCENT, fg=WHITE,
                  font=("Helvetica", 10, "bold"), relief="flat", cursor="hand2",
                  pady=7, command=self._apply_filter).pack(fill="x", padx=12, pady=(12, 4))
        tk.Button(body, text="Envoyer vers Histogramme →", bg=BG, fg=ACCENT,
                  font=("Helvetica", 9), relief="flat", cursor="hand2",
                  pady=5, command=self._send_next).pack(fill="x", padx=12, pady=(0, 12))

        self._on_noise_change()

    def _main_area(self):
        main = tk.Frame(self, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure((0, 1, 2), weight=1, uniform="col")
        main.rowconfigure(1, weight=1)

        sbar = tk.Frame(main, bg=WHITE, height=42)
        sbar.grid(row=0, column=0, columnspan=3, sticky="ew")
        sbar.pack_propagate(False)
        tk.Frame(sbar, bg=BORDER, height=1).pack(fill="x", side="bottom")
        self.stats = {}
        for key, label in [("size", "Dimensions"), ("snr", "Bruit ajouté"),
                            ("filter", "Filtre appliqué"), ("mse", "MSE")]:
            f = tk.Frame(sbar, bg=WHITE)
            f.pack(side="left", padx=18, pady=6)
            tk.Label(f, text=label, bg=WHITE, fg=TEXT3,
                     font=("Helvetica", 7, "bold")).pack(anchor="w")
            lbl = tk.Label(f, text="—", bg=WHITE, fg=TEXT,
                           font=("Helvetica", 9, "bold"))
            lbl.pack(anchor="w")
            self.stats[key] = lbl

        specs = [
            ("Image microscopique", "SOURCE",       COLOR,  COLOR_L,   "orig"),
            ("Image bruitée",       "BRUIT AJOUTÉ", WARN,   WARN_L,    "noisy"),
            ("Image débruitée",     "RÉSULTAT M1",  ACCENT, "#EBF3FB", "result"),
        ]
        self.canvases = {}
        self.ph_lbls  = {}
        for col, (title, tag, c, bg_c, key) in enumerate(specs):
            panel = tk.Frame(main, bg=WHITE)
            panel.grid(row=1, column=col, sticky="nsew",
                       padx=(10 if col == 0 else 4, 4 if col < 2 else 10), pady=10)
            panel.columnconfigure(0, weight=1)
            panel.rowconfigure(1, weight=1)

            hdr = tk.Frame(panel, bg=bg_c, height=36)
            hdr.grid(row=0, column=0, sticky="ew")
            hdr.grid_propagate(False)
            tk.Label(hdr, text=title, bg=bg_c, fg=c,
                     font=("Helvetica", 10, "bold")).pack(side="left", padx=12, pady=8)
            tk.Label(hdr, text=tag, bg=bg_c, fg=c,
                     font=("Helvetica", 8)).pack(side="right", padx=12)

            cv = tk.Canvas(panel, bg="#E8EDF4", highlightthickness=0)
            cv.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
            self.canvases[key] = cv

            ph = tk.Label(cv, bg="#E8EDF4", fg=TEXT3, font=("Helvetica", 9),
                          text="Charger un échantillon" if key == "orig" else "En attente…")
            ph.place(relx=0.5, rely=0.5, anchor="center")
            self.ph_lbls[key] = ph

            info = tk.Label(panel, text="—", bg=WHITE, fg=TEXT3, font=("Helvetica", 8))
            info.grid(row=2, column=0, pady=(0, 6))
            self.ph_lbls[key + "_info"] = info

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
        cv = self.canvases[key]
        cv.update_idletasks()
        cw, ch = cv.winfo_width(), cv.winfo_height()
        if cw < 10:
            cw, ch = 320, 320
        img   = Image.fromarray(pixels.astype(np.uint8), mode="L")
        sc    = min(cw / img.width, ch / img.height, 1.0)
        img   = img.resize((max(1, int(img.width * sc)),
                            max(1, int(img.height * sc))), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        cv.delete("all")
        cv.create_image(cw // 2, ch // 2, anchor="center", image=photo)
        self.ph_lbls[key].place_forget()
        self._photos[key] = photo

    def _on_noise_change(self, *_):
        noise = self.noise_var.get()
        info  = NOISE_FILTER_MAP.get(noise, {})
        rec   = info.get("recommended")
        self.ctx_lbl.config(text=BIO_CONTEXT.get(noise, "—"))
        if rec:
            self.rec_badge.config(bg=OK_L)
            self.rec_lbl.config(text=rec, bg=OK_L, fg=OK)
            self.filter_var.set(rec)
        else:
            self.rec_badge.config(bg=WARN_L)
            self.rec_lbl.config(text="Libre — tous disponibles", bg=WARN_L, fg=WARN)
        for fname, rb in self.filter_rbs.items():
            rb.config(fg=ACCENT if fname == rec else TEXT,
                      font=("Helvetica", 10, "bold" if fname == rec else "normal"))

    def _apply_noise(self):
        if not self.state.has_image():
            return
        pix    = self.state.original_pixels.copy()
        noise  = self.noise_var.get()
        amount = self.amount_var.get()
        if noise == "Poivre & Sel":
            noisy = add_salt_pepper(pix, amount)
            label = f"S&P {amount*100:.0f}%"
        elif noise == "Gaussien":
            noisy = add_gaussian_noise(pix, sigma=amount * 100)
            label = f"Gauss σ={amount*100:.0f}"
        else:
            noisy = pix.copy()
            label = "Aucun"
        self.state.m1_result = noisy
        self._display(noisy, "noisy")
        mse = float(np.mean((pix.astype(float) - noisy.astype(float)) ** 2))
        self.stats["snr"].config(text=label)
        self.stats["mse"].config(text=f"{mse:.1f}")
        self.ph_lbls["noisy_info"].config(
            text=f"min={noisy.min()}  max={noisy.max()}  moy={noisy.mean():.1f}")

    def _apply_filter(self):
        src = self.state.m1_result if self.state.m1_result is not None \
              else self.state.original_pixels
        if src is None:
            return
        fname  = self.filter_var.get()
        result = FILTER_FNS[fname](src, self.ksize_var.get(), self.sigma_var.get())
        self.state.m1_result = result
        self._display(result, "result")
        self.state.m1_log = {
            "noise":  self.noise_var.get(),
            "amount": f"{self.amount_var.get()*100:.0f}%",
            "filter": fname,
            "ksize":  f"{self.ksize_var.get()}×{self.ksize_var.get()}",
            "sigma":  f"{self.sigma_var.get():.1f}",
            "mse":    self.stats["mse"].cget("text"),
            "min": int(result.min()), "max": int(result.max()),
            "mean": round(float(result.mean()), 1),
        }
        self.stats["filter"].config(text=fname)
        self.ph_lbls["result_info"].config(
            text=f"min={result.min()}  max={result.max()}  moy={result.mean():.1f}")
        self.state.mark_step(0)

    def _send_next(self):
        if self.state.m1_result is not None:
            self.state.mark_step(0)

    def refresh_image(self):
        if self.state.has_image():
            self._display(self.state.pipeline_input(0), "orig")
            self.stats["size"].config(
                text=f"{self.state.img_w}×{self.state.img_h}")
            self.ph_lbls["orig_info"].config(
                text=f"Niveaux de gris · {self.state.img_w}×{self.state.img_h} px")