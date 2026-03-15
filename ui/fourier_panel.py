import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from core.fft_manual import (fft2d, ifft2d, fft_shift,
                               apply_frequency_mask, spectrum_image)

WHITE   = "#FFFFFF"
BG      = "#F4F6F9"
BORDER  = "#DDE2EC"
COLOR   = "#C47A00"
COLOR_L = "#FDF4E3"
ACCENT  = "#1A6FBF"
TEXT    = "#1C2333"
TEXT2   = "#5A6478"
TEXT3   = "#9BA5B8"

BIO_CONTEXT = {
    "spectrum":   "Visualiser les fréquences dominantes — taches lumineuses = structures répétitives",
    "passe-bas":  "Éliminer le bruit haute fréquence — adoucir l'image microscopique",
    "passe-haut": "Renforcer les contours cellulaires — détecter les membranes",
    "passe-bande":"Isoler une plage de textures — séparer noyau du cytoplasme",
}


class FourierPanel(tk.Frame):

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
        tk.Label(col, text="Étape 3 — Fourier", bg=COLOR_L, fg=COLOR,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Label(col, text="Filtrage fréquentiel microscopique", bg=COLOR_L, fg=COLOR,
                 font=("Helvetica", 8)).pack(anchor="w")

        body = tk.Frame(sb, bg=WHITE)
        body.pack(fill="both", expand=True)

        badge = tk.Frame(body, bg=COLOR_L)
        badge.pack(fill="x", padx=12, pady=(14, 0))
        tk.Label(badge, text="Cooley-Tukey radix-2 — from scratch",
                 bg=COLOR_L, fg=COLOR, font=("Helvetica", 8, "bold")).pack(
                     anchor="w", padx=8, pady=(5, 0))
        tk.Label(badge, text="Zéro bibliothèque de traitement d'image",
                 bg=COLOR_L, fg=COLOR, font=("Helvetica", 8)).pack(
                     anchor="w", padx=8, pady=(1, 6))

        self._section(body, "APPLICATION BIOLOGIQUE")
        self.ctx_lbl = tk.Label(body, text="—", bg=WHITE, fg=TEXT2,
                                 font=("Helvetica", 8), wraplength=210, justify="left")
        self.ctx_lbl.pack(anchor="w", padx=14, pady=(0, 8))

        self._sep(body)
        self._section(body, "TYPE DE FILTRE")
        self.ftype_var = tk.StringVar(value="spectrum")
        for label, val in [("Spectre de magnitude",   "spectrum"),
                            ("Passe-bas — lissage",    "passe-bas"),
                            ("Passe-haut — contours",  "passe-haut"),
                            ("Passe-bande — textures", "passe-bande")]:
            tk.Radiobutton(body, text=label, variable=self.ftype_var, value=val,
                           bg=WHITE, fg=TEXT, selectcolor=COLOR_L,
                           activebackground=WHITE, font=("Helvetica", 10),
                           command=self._on_type_change).pack(anchor="w", padx=20, pady=2)

        self._section(body, "RAYON DE COUPURE")
        self.radius_var = tk.IntVar(value=30)
        self._slider(body, "Rayon", self.radius_var, 5, 150, 1,
                     fmt=lambda v: f"{int(v)} px")

        self._sep(body)
        self.status_lbl = tk.Label(body, text="En attente…", bg=WHITE, fg=TEXT3,
                                    font=("Helvetica", 8), wraplength=210)
        self.status_lbl.pack(padx=14, anchor="w", pady=(0, 10))

        tk.Button(body, text="Calculer FFT", bg=COLOR, fg=WHITE,
                  font=("Helvetica", 10, "bold"), relief="flat", cursor="hand2",
                  pady=7, command=self.run).pack(fill="x", padx=12, pady=(4, 4))
        tk.Button(body, text="Envoyer vers Morphologie →", bg=BG, fg=ACCENT,
                  font=("Helvetica", 9), relief="flat", cursor="hand2",
                  pady=5, command=self._send_next).pack(fill="x", padx=12, pady=(0, 12))

        self._on_type_change()

    def _main_area(self):
        main = tk.Frame(self, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure((0, 1, 2), weight=1, uniform="col")
        main.rowconfigure(0, weight=1)

        for col, (title, tag, c, bg_c, key) in enumerate([
            ("Entrée (M2 ou originale)", "SOURCE",    ACCENT,    "#EBF3FB", "orig"),
            ("Spectre FFT (log scale)",  "FRÉQUENCE", COLOR,     COLOR_L,   "spectrum"),
            ("Image reconstruite (IFFT)","RÉSULTAT M3","#1A8754","#E8F6EF", "result"),
        ]):
            p = tk.Frame(main, bg=WHITE)
            p.grid(row=0, column=col, sticky="nsew",
                   padx=(10 if col == 0 else 4, 4 if col < 2 else 10), pady=10)
            p.columnconfigure(0, weight=1)
            p.rowconfigure(1, weight=1)
            hdr = tk.Frame(p, bg=bg_c, height=36)
            hdr.grid(row=0, column=0, sticky="ew")
            hdr.grid_propagate(False)
            tk.Label(hdr, text=title, bg=bg_c, fg=c,
                     font=("Helvetica", 10, "bold")).pack(side="left", padx=12, pady=8)
            tk.Label(hdr, text=tag, bg=bg_c, fg=c,
                     font=("Helvetica", 8)).pack(side="right", padx=12)
            cv = tk.Canvas(p, bg="#E8EDF4", highlightthickness=0)
            cv.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
            setattr(self, f"cv_{key}", cv)
            ph = tk.Label(cv, bg="#E8EDF4", fg=TEXT3, font=("Helvetica", 9),
                          text="En attente…")
            ph.place(relx=0.5, rely=0.5, anchor="center")
            setattr(self, f"ph_{key}", ph)

    def _sep(self, p):
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=10, pady=8)

    def _section(self, p, t):
        tk.Label(p, text=t, bg=WHITE, fg=TEXT3,
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=14, pady=(8, 3))

    def _slider(self, p, label, var, lo, hi, res, fmt=None):
        row = tk.Frame(p, bg=WHITE)
        row.pack(fill="x", padx=14, pady=1)
        tk.Label(row, text=label, bg=WHITE, fg=TEXT2, font=("Helvetica", 9),
                 width=8, anchor="w").pack(side="left")
        vl = tk.Label(row, text=fmt(var.get()) if fmt else str(var.get()),
                      bg=WHITE, fg=ACCENT, font=("Helvetica", 9, "bold"), width=8)
        vl.pack(side="right")
        tk.Scale(p, variable=var, from_=lo, to=hi, resolution=res,
                 orient="horizontal", bg=WHITE, troughcolor=BORDER,
                 highlightthickness=0, showvalue=False,
                 command=lambda v: vl.config(
                     text=fmt(float(v)) if fmt else str(int(float(v))))
                 ).pack(fill="x", padx=14)

    def _on_type_change(self, *_):
        self.ctx_lbl.config(text=BIO_CONTEXT.get(self.ftype_var.get(), "—"))

    def _display(self, pixels, key):
        cv = getattr(self, f"cv_{key}")
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
        getattr(self, f"ph_{key}").place_forget()
        self._photos[key] = photo

    def run(self):
        if not self.state.has_image():
            return
        self.status_lbl.config(text="Calcul FFT en cours…")
        self.update()
        pix    = self.state.pipeline_input(2)
        ftype  = self.ftype_var.get()
        radius = self.radius_var.get()
        re, im, W, H = fft2d(pix)
        sr, si = fft_shift(re, im, W, H)
        spec   = spectrum_image(sr, si, W, H, pix.shape[0], pix.shape[1])
        self._display(spec, "spectrum")
        if ftype != "spectrum":
            apply_frequency_mask(sr, si, W, H, ftype, radius)
            br, bi = fft_shift(sr, si, W, H)
            ifft2d(br, bi, W, H)
            raw = br[:pix.shape[0], :pix.shape[1]]
            mn, mx = raw.min(), raw.max()
            result = ((raw - mn) / (mx - mn) * 255).astype(np.uint8) \
                     if mx > mn else np.zeros_like(pix)
        else:
            result = spec
        self.state.m3_result = result
        self._display(result, "result")
        self.state.m3_log = {
            "filter": ftype,
            "radius": radius,
            "algo":   "Cooley-Tukey radix-2 from scratch",
        }
        self.status_lbl.config(text=f"FFT terminée — {ftype}")
        self.state.mark_step(2)

    def _send_next(self):
        if self.state.m3_result is not None:
            self.state.mark_step(2)

    def refresh_image(self):
        if self.state.has_image():
            self._display(self.state.pipeline_input(2), "orig")