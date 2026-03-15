import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from core.morpho import run_morpho

WHITE   = "#FFFFFF"
BG      = "#F4F6F9"
BORDER  = "#DDE2EC"
COLOR   = "#C0393B"
COLOR_L = "#FBE9E9"
ACCENT  = "#1A6FBF"
TEXT    = "#1C2333"
TEXT2   = "#5A6478"
TEXT3   = "#9BA5B8"
OK      = "#1A8754"
OK_L    = "#E8F6EF"

BIO_OPS = {
    "Érosion":    "Réduire et séparer les objets collés — séparer cellules jointives",
    "Dilatation": "Élargir les régions — fermer les micro-lacunes membranaires",
    "Opening":    "Supprimer le bruit binaire sans altérer la forme des cellules",
    "Closing":    "Boucher les trous internes — reconstruire les noyaux fragmentés",
    "Top Hat":    "Extraire les structures brillantes fines — granules cytoplasmiques",
    "Black Hat":  "Extraire les structures sombres — vacuoles, inclusions",
    "Gradient":   "Détecter les contours — tracer les membranes cellulaires",
}


class MorphoPanel(tk.Frame):

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
        tk.Label(col, text="Étape 4 — Morphologie", bg=COLOR_L, fg=COLOR,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Label(col, text="Segmentation & comptage cellulaire", bg=COLOR_L, fg=COLOR,
                 font=("Helvetica", 8)).pack(anchor="w")

        body = tk.Frame(sb, bg=WHITE)
        body.pack(fill="both", expand=True)

        self._section(body, "APPLICATION BIOLOGIQUE")
        self.ctx_lbl = tk.Label(body, text="—", bg=WHITE, fg=TEXT2,
                                 font=("Helvetica", 8), wraplength=210, justify="left")
        self.ctx_lbl.pack(anchor="w", padx=14, pady=(0, 6))

        self._sep(body)
        self._section(body, "OPÉRATION")
        self.op_var = tk.StringVar(value="Érosion")
        for op in BIO_OPS:
            tk.Radiobutton(body, text=op, variable=self.op_var, value=op,
                           bg=WHITE, fg=TEXT, selectcolor=COLOR_L,
                           activebackground=WHITE, font=("Helvetica", 10),
                           command=self._on_op_change).pack(anchor="w", padx=20, pady=1)

        self._sep(body)
        self._section(body, "SEUIL DE BINARISATION")
        self.thresh_var = tk.IntVar(value=127)
        self._slider(body, "Seuil", self.thresh_var, 0, 255, 1,
                     fmt=lambda v: str(int(v)))

        self._section(body, "ITÉRATIONS")
        self.iter_var = tk.IntVar(value=1)
        self._slider(body, "N", self.iter_var, 1, 10, 1,
                     fmt=lambda v: str(int(v)))

        self._sep(body)
        rpt = tk.Frame(body, bg=OK_L)
        rpt.pack(fill="x", padx=12, pady=(0, 10))
        tk.Label(rpt, text="Rapport d'analyse", bg=OK_L, fg=OK,
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=8, pady=(6, 2))
        self.rpt_cells = tk.Label(rpt, text="Objets détectés : —",
                                   bg=OK_L, fg=OK, font=("Helvetica", 9, "bold"))
        self.rpt_cells.pack(anchor="w", padx=8)
        self.rpt_area = tk.Label(rpt, text="Aire moyenne : —",
                                  bg=OK_L, fg=OK, font=("Helvetica", 8))
        self.rpt_area.pack(anchor="w", padx=8, pady=(0, 6))

        tk.Button(body, text="Appliquer", bg=COLOR, fg=WHITE,
                  font=("Helvetica", 10, "bold"), relief="flat", cursor="hand2",
                  pady=7, command=self.run).pack(fill="x", padx=12, pady=(4, 4))
        tk.Button(body, text="Générer rapport complet", bg=BG, fg=ACCENT,
                  font=("Helvetica", 9), relief="flat", cursor="hand2",
                  pady=5, command=self._full_report).pack(fill="x", padx=12, pady=(0, 12))

        self._on_op_change()

    def _main_area(self):
        main = tk.Frame(self, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=0)

        for col, (title, tag, c, bg_c, key) in enumerate([
            ("Entrée binarisée (M3 ou originale)", "SOURCE",      ACCENT, "#EBF3FB", "orig"),
            ("Résultat morphologique",              "RÉSULTAT M4", COLOR,  COLOR_L,   "result"),
        ]):
            p = tk.Frame(main, bg=WHITE)
            p.grid(row=0, column=col, sticky="nsew",
                   padx=(10 if col == 0 else 4, 4 if col == 0 else 10), pady=(10, 4))
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

        sum_bar = tk.Frame(main, bg=WHITE, height=48)
        sum_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(4, 10))
        sum_bar.pack_propagate(False)
        tk.Frame(sum_bar, bg=BORDER, height=1).pack(fill="x", side="top")
        self.summary_lbl = tk.Label(
            sum_bar,
            text="Pipeline : charger un échantillon pour commencer l'analyse",
            bg=WHITE, fg=TEXT2, font=("Helvetica", 9))
        self.summary_lbl.pack(side="left", padx=14, pady=10)

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
                      bg=WHITE, fg=ACCENT, font=("Helvetica", 9, "bold"), width=5)
        vl.pack(side="right")
        tk.Scale(p, variable=var, from_=lo, to=hi, resolution=res,
                 orient="horizontal", bg=WHITE, troughcolor=BORDER,
                 highlightthickness=0, showvalue=False,
                 command=lambda v: vl.config(
                     text=fmt(float(v)) if fmt else str(int(float(v))))
                 ).pack(fill="x", padx=14)

    def _on_op_change(self, *_):
        self.ctx_lbl.config(text=BIO_OPS.get(self.op_var.get(), "—"))

    def _display(self, pixels, key):
        cv = getattr(self, f"cv_{key}")
        cv.update_idletasks()
        cw, ch = cv.winfo_width(), cv.winfo_height()
        if cw < 10:
            cw, ch = 400, 400
        img   = Image.fromarray(pixels.astype(np.uint8), mode="L")
        sc    = min(cw / img.width, ch / img.height, 1.0)
        img   = img.resize((max(1, int(img.width * sc)),
                            max(1, int(img.height * sc))), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        cv.delete("all")
        cv.create_image(cw // 2, ch // 2, anchor="center", image=photo)
        getattr(self, f"ph_{key}").place_forget()
        self._photos[key] = photo

    def _count_objects(self, result):
        binary  = (result > 127).astype(np.uint8)
        visited = np.zeros_like(binary)
        count, areas = 0, []
        for y in range(binary.shape[0]):
            for x in range(binary.shape[1]):
                if binary[y, x] == 1 and visited[y, x] == 0:
                    area = self._flood(binary, visited, y, x)
                    count += 1
                    areas.append(area)
        return count, (float(np.mean(areas)) if areas else 0.0)

    def _flood(self, binary, visited, y0, x0):
        stack = [(y0, x0)]
        h, w  = binary.shape
        area  = 0
        while stack:
            y, x = stack.pop()
            if y < 0 or y >= h or x < 0 or x >= w:
                continue
            if visited[y, x] or binary[y, x] == 0:
                continue
            visited[y, x] = 1
            area += 1
            stack += [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]
        return area

    def run(self):
        if not self.state.has_image():
            return
        pix    = self.state.pipeline_input(3)
        result = run_morpho(pix, self.op_var.get(),
                            self.thresh_var.get(), self.iter_var.get())
        self.state.m4_result = result
        self._display(result, "result")
        count, mean_area = self._count_objects(result)
        self.rpt_cells.config(text=f"Objets détectés : {count}")
        self.rpt_area.config(text=f"Aire moyenne : {mean_area:.1f} px²")
        self.state.m4_log = {
            "operation":  self.op_var.get(),
            "seuil":      self.thresh_var.get(),
            "iterations": self.iter_var.get(),
            "count":      count,
            "mean_area":  round(mean_area, 1),
        }
        self.state.mark_step(3)
        steps = ["M1", "M2", "M3", "M4"]
        done  = [s for s, d in zip(steps, self.state.steps_done) if d]
        self.summary_lbl.config(
            text=f"Pipeline : {' → '.join(done)}  |  {count} objets · aire moy. {mean_area:.0f} px²")

    def _full_report(self):
        if not self.state.has_image():
            return
        steps_done = [s for s, d in zip(
            ["Prétraitement", "Histogramme", "Fourier", "Morphologie"],
            self.state.steps_done) if d]
        count, mean_area = 0, 0.0
        if self.state.m4_result is not None:
            count, mean_area = self._count_objects(self.state.m4_result)
        msg = (
            f"=== RAPPORT D'ANALYSE MICROSCOPIQUE ===\n\n"
            f"Fichier     : {self.state.filename}\n"
            f"Dimensions  : {self.state.img_w} × {self.state.img_h} px\n\n"
            f"Étapes complétées :\n" +
            "\n".join(f"  ✓ {s}" for s in steps_done) +
            f"\n\nRésultats M4 — Morphologie :\n"
            f"  Objets détectés : {count}\n"
            f"  Aire moyenne    : {mean_area:.1f} px²\n"
        )
        from tkinter import messagebox
        messagebox.showinfo("Rapport d'analyse", msg)

    def refresh_image(self):
        if self.state.has_image():
            pix  = self.state.pipeline_input(3)
            self._display(pix, "orig")
            done = [s for s, d in zip(["M1", "M2", "M3"], self.state.steps_done[:3]) if d]
            if done:
                self.summary_lbl.config(
                    text=f"Entrée : résultat de {done[-1]} — prêt pour la segmentation")