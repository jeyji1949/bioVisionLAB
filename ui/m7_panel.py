"""
ui/m7_panel.py
==============
Étape 7 — Classification ML de cellules sanguines (BloodMNIST)
Panneau Tkinter intégré dans BioVision Lab.

Layout
------
  Sidebar (gauche, 260px) :
    - Paramètres RF (n_estimators, n_train)
    - Paramètres CNN (epochs, batch_size)
    - Boutons Charger / Lancer RF / Lancer CNN
    - Log de progression

  Main area (droite) :
    - Onglet RF  : accuracy, matrice de confusion, accuracy par classe
    - Onglet CNN : accuracy, courbes loss/acc, accuracy par classe
"""

import tkinter as tk
from tkinter import ttk
import threading
import numpy as np

WHITE   = "#FFFFFF"
BG      = "#F4F6F9"
BORDER  = "#DDE2EC"
ACCENT  = "#1A6FBF"
TEXT    = "#1C2333"
TEXT2   = "#5A6478"
TEXT3   = "#9BA5B8"

C_M7    = "#0F6B3A"   # dark green — ML module
C_M7L   = "#E6F4ED"
C_RF    = "#1A6FBF"   # blue — Random Forest
C_RFL   = "#EBF3FB"
C_CNN   = "#6C47CC"   # purple — CNN
C_CNNL  = "#F0EDFB"
C_OK    = "#1A8754"
C_OKL   = "#E8F6EF"
C_ERR   = "#C0393B"
C_ERRL  = "#FBE9E9"
C_WARN  = "#C47A00"
C_WARNL = "#FDF4E3"

BLOOD_CLASSES = [
    "Basophile", "Éosinophile", "Érythroblaste", "Granulocyte imm.",
    "Lymphocyte", "Monocyte", "Neutrophile", "Plaquette"
]


class M7Panel(tk.Frame):

    def __init__(self, parent, state):
        super().__init__(parent, bg=BG)
        self.state   = state
        self._runner = None
        self._thread = None
        self._build()

    # ── Build ──────────────────────────────────────────────────────────────────

    def _build(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._sidebar()
        self._main_area()

    # ── Sidebar ────────────────────────────────────────────────────────────────

    def _sidebar(self):
        sb = tk.Frame(self, bg=WHITE, width=260)
        sb.grid(row=0, column=0, sticky="ns")
        sb.grid_propagate(False)
        tk.Frame(sb, bg=BORDER, width=1).place(relx=1, rely=0, relheight=1)

        # Title
        tb = tk.Frame(sb, bg=C_M7L, height=56)
        tb.pack(fill="x"); tb.pack_propagate(False)
        tk.Frame(tb, bg=C_M7, width=3).pack(side="left", fill="y")
        col = tk.Frame(tb, bg=C_M7L)
        col.pack(side="left", padx=12, pady=10)
        tk.Label(col, text="Étape 7 — ML Classification",
                 bg=C_M7L, fg=C_M7, font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Label(col, text="BloodMNIST · Random Forest · CNN",
                 bg=C_M7L, fg=C_M7, font=("Helvetica", 8)).pack(anchor="w")

        body = tk.Frame(sb, bg=WHITE)
        body.pack(fill="both", expand=True)

        # Dataset info badge
        badge = tk.Frame(body, bg=C_M7L)
        badge.pack(fill="x", padx=12, pady=(14, 0))
        tk.Label(badge, text="BloodMNIST — 17 092 images",
                 bg=C_M7L, fg=C_M7, font=("Helvetica", 8, "bold")).pack(
                     anchor="w", padx=8, pady=(5, 0))
        tk.Label(badge, text="28×28 RGB · 8 types de cellules sanguines",
                 bg=C_M7L, fg=C_M7, font=("Helvetica", 8)).pack(
                     anchor="w", padx=8, pady=(1, 6))

        self._sep(body)
        self._section(body, "PARAMÈTRES DONNÉES")

        self.n_train_var = tk.IntVar(value=2000)
        self._slider(body, "Train samples", self.n_train_var,
                     500, 5000, 500, fmt=lambda v: str(int(v)))
        self.n_test_var = tk.IntVar(value=400)
        self._slider(body, "Test samples", self.n_test_var,
                     100, 1000, 100, fmt=lambda v: str(int(v)))

        tk.Button(body, text="Charger BloodMNIST",
                  bg=C_M7, fg=WHITE,
                  font=("Helvetica", 10, "bold"), relief="flat",
                  cursor="hand2", pady=7,
                  command=self._load_data).pack(fill="x", padx=12, pady=(10, 4))

        self._sep(body)
        self._section(body, "RANDOM FOREST")

        self.n_est_var = tk.IntVar(value=10)
        self._slider(body, "Arbres", self.n_est_var, 5, 30, 5,
                     fmt=lambda v: str(int(v)))
        self.depth_var = tk.IntVar(value=8)
        self._slider(body, "Profondeur max", self.depth_var, 4, 16, 2,
                     fmt=lambda v: str(int(v)))

        self.rf_btn = tk.Button(body, text="Entraîner Random Forest",
                                bg=C_RF, fg=WHITE,
                                font=("Helvetica", 10, "bold"), relief="flat",
                                cursor="hand2", pady=7,
                                state="disabled",
                                command=self._train_rf)
        self.rf_btn.pack(fill="x", padx=12, pady=(8, 4))

        self._sep(body)
        self._section(body, "CNN PYTORCH")

        self.epochs_var = tk.IntVar(value=5)
        self._slider(body, "Epochs", self.epochs_var, 1, 20, 1,
                     fmt=lambda v: str(int(v)))
        self.batch_var = tk.IntVar(value=64)
        self._slider(body, "Batch size", self.batch_var, 32, 256, 32,
                     fmt=lambda v: str(int(v)))

        self.cnn_btn = tk.Button(body, text="Entraîner CNN",
                                 bg=C_CNN, fg=WHITE,
                                 font=("Helvetica", 10, "bold"), relief="flat",
                                 cursor="hand2", pady=7,
                                 state="disabled",
                                 command=self._train_cnn)
        self.cnn_btn.pack(fill="x", padx=12, pady=(8, 4))

        self._sep(body)
        self._section(body, "PROGRESSION")

        log_frame = tk.Frame(body, bg=BG, height=120)
        log_frame.pack(fill="x", padx=12, pady=(0, 8))
        log_frame.pack_propagate(False)

        self._log_text = tk.Text(log_frame, bg=BG, fg=TEXT2,
                                  font=("Courier", 8), relief="flat",
                                  state="disabled", wrap="word",
                                  height=7)
        self._log_text.pack(fill="both", expand=True)

        self.status_lbl = tk.Label(body, text="Prêt — charger BloodMNIST",
                                    bg=WHITE, fg=TEXT3, font=("Helvetica", 8))
        self.status_lbl.pack(padx=14, anchor="w", pady=(0, 8))

    # ── Main area (tabs) ───────────────────────────────────────────────────────

    def _main_area(self):
        main = tk.Frame(self, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        # Tab bar
        tab_bar = tk.Frame(main, bg=WHITE, height=40)
        tab_bar.grid(row=0, column=0, sticky="ew")
        tab_bar.pack_propagate(False)
        tk.Frame(main, bg=BORDER, height=1).grid(row=0, column=0,
                                                   sticky="ew", pady=(39, 0))

        self._active_tab = tk.StringVar(value="rf")
        self._tab_btns   = {}
        for label, key, c in [
            ("Random Forest", "rf",  C_RF),
            ("CNN PyTorch",   "cnn", C_CNN),
        ]:
            btn = tk.Button(tab_bar, text=label,
                            bg=C_RFL if key == "rf" else WHITE,
                            fg=c if key == "rf" else TEXT3,
                            font=("Helvetica", 10,
                                  "bold" if key == "rf" else "normal"),
                            relief="flat", cursor="hand2",
                            padx=16, pady=8,
                            command=lambda k=key: self._switch_tab(k))
            btn.pack(side="left")
            self._tab_btns[key] = (btn, c)

        # Content frames
        self._content = tk.Frame(main, bg=BG)
        self._content.grid(row=1, column=0, sticky="nsew")
        self._content.columnconfigure(0, weight=1)
        self._content.rowconfigure(0, weight=1)

        self._rf_frame  = tk.Frame(self._content, bg=BG)
        self._cnn_frame = tk.Frame(self._content, bg=BG)
        for f in [self._rf_frame, self._cnn_frame]:
            f.place(relx=0, rely=0, relwidth=1, relheight=1)
            f.columnconfigure(0, weight=1)
            f.rowconfigure(0, weight=1)

        self._build_rf_frame()
        self._build_cnn_frame()
        self._switch_tab("rf")

    def _switch_tab(self, key):
        self._active_tab.set(key)
        if key == "rf":
            self._rf_frame.lift()
        else:
            self._cnn_frame.lift()
        for k, (btn, c) in self._tab_btns.items():
            active = (k == key)
            light  = C_RFL if k == "rf" else C_CNNL
            btn.config(
                bg=light if active else WHITE,
                fg=c if active else TEXT3,
                font=("Helvetica", 10, "bold" if active else "normal"),
            )

    # ── RF frame ───────────────────────────────────────────────────────────────

    def _build_rf_frame(self):
        f = self._rf_frame
        f.rowconfigure(0, weight=0)
        f.rowconfigure(1, weight=1)

        # Header card
        hdr = tk.Frame(f, bg=WHITE)
        hdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        hdr.columnconfigure((0, 1, 2, 3), weight=1, uniform="rh")
        tk.Frame(hdr, bg=C_RF, height=3).grid(
            row=0, column=0, columnspan=4, sticky="ew")

        self._rf_stats = {}
        for col, (label, key) in enumerate([
            ("Accuracy",    "accuracy"),
            ("N arbres",    "n_estimators"),
            ("Train",       "n_train"),
            ("Test",        "n_test"),
        ]):
            cell = tk.Frame(hdr, bg=WHITE)
            cell.grid(row=1, column=col, sticky="ew",
                      padx=(14 if col == 0 else 8, 8), pady=10)
            tk.Label(cell, text=label, bg=WHITE, fg=TEXT3,
                     font=("Helvetica", 8)).pack(anchor="w")
            lbl = tk.Label(cell, text="—", bg=WHITE, fg=C_RF,
                           font=("Helvetica", 16, "bold"))
            lbl.pack(anchor="w")
            self._rf_stats[key] = lbl

        # Scrollable results
        host = tk.Frame(f, bg=BG)
        host.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)

        canvas = tk.Canvas(host, bg=BG, highlightthickness=0)
        sb     = tk.Scrollbar(host, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")
        canvas.grid(row=0, column=0, sticky="nsew")

        self._rf_scroll = tk.Frame(canvas, bg=BG)
        self._rf_scroll.columnconfigure(0, weight=1)
        win = canvas.create_window((0, 0), window=self._rf_scroll, anchor="nw")
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win, width=e.width))
        self._rf_scroll.bind("<Configure>",
                             lambda e: canvas.configure(
                                 scrollregion=canvas.bbox("all")))

        self._rf_placeholder()

    def _rf_placeholder(self):
        for w in self._rf_scroll.winfo_children():
            w.destroy()
        tk.Label(self._rf_scroll,
                 text="Charger BloodMNIST puis lancer l'entraînement Random Forest",
                 bg=BG, fg=TEXT3, font=("Helvetica", 10)).pack(pady=60)

    def _render_rf_results(self, results):
        for w in self._rf_scroll.winfo_children():
            w.destroy()

        self._rf_stats["accuracy"].config(
            text=f"{results['accuracy']}%")
        self._rf_stats["n_estimators"].config(
            text=str(results["n_estimators"]))
        self._rf_stats["n_train"].config(text=str(results["n_train"]))
        self._rf_stats["n_test"].config(text=str(results["n_test"]))

        sf = self._rf_scroll

        # Per-class accuracy bar chart
        self._section_lbl(sf, "Accuracy par classe de cellule")
        bar_frame = tk.Frame(sf, bg=WHITE)
        bar_frame.pack(fill="x", padx=10, pady=(0, 10))

        per_class = results["per_class_acc"]
        max_acc   = max(per_class) if per_class else 100
        for i, (cls, acc) in enumerate(zip(BLOOD_CLASSES, per_class)):
            row = tk.Frame(bar_frame, bg=WHITE)
            row.pack(fill="x", padx=10, pady=2)
            row.columnconfigure(1, weight=1)
            tk.Label(row, text=cls, bg=WHITE, fg=TEXT2,
                     font=("Helvetica", 9), width=18, anchor="w").grid(
                         row=0, column=0, padx=(8, 6))
            bar_bg = tk.Frame(row, bg=BG, height=18)
            bar_bg.grid(row=0, column=1, sticky="ew", padx=(0, 8))
            bar_bg.update_idletasks()
            bar_bg.columnconfigure(0, weight=1)
            w_pct = acc / 100.0
            color = C_OK if acc >= 70 else (C_WARN if acc >= 50 else C_ERR)
            tk.Frame(bar_bg, bg=color, height=18,
                     width=max(1, int(200 * w_pct))).pack(side="left")
            tk.Label(row, text=f"{acc}%", bg=WHITE, fg=color,
                     font=("Helvetica", 9, "bold"), width=6).grid(
                         row=0, column=2, padx=(0, 8))

        # Confusion matrix
        self._section_lbl(sf, "Matrice de confusion (prédiction × réel)")
        cm_frame = tk.Frame(sf, bg=WHITE)
        cm_frame.pack(fill="x", padx=10, pady=(0, 10))
        self._draw_confusion_matrix(cm_frame, results["confusion_matrix"],
                                     color=C_RF)

    # ── CNN frame ──────────────────────────────────────────────────────────────

    def _build_cnn_frame(self):
        f = self._cnn_frame
        f.rowconfigure(0, weight=0)
        f.rowconfigure(1, weight=1)

        hdr = tk.Frame(f, bg=WHITE)
        hdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        hdr.columnconfigure((0, 1, 2, 3), weight=1, uniform="ch")
        tk.Frame(hdr, bg=C_CNN, height=3).grid(
            row=0, column=0, columnspan=4, sticky="ew")

        self._cnn_stats = {}
        for col, (label, key) in enumerate([
            ("Accuracy",  "accuracy"),
            ("Epochs",    "epochs"),
            ("Train",     "n_train"),
            ("Test",      "n_test"),
        ]):
            cell = tk.Frame(hdr, bg=WHITE)
            cell.grid(row=1, column=col, sticky="ew",
                      padx=(14 if col == 0 else 8, 8), pady=10)
            tk.Label(cell, text=label, bg=WHITE, fg=TEXT3,
                     font=("Helvetica", 8)).pack(anchor="w")
            lbl = tk.Label(cell, text="—", bg=WHITE, fg=C_CNN,
                           font=("Helvetica", 16, "bold"))
            lbl.pack(anchor="w")
            self._cnn_stats[key] = lbl

        host = tk.Frame(f, bg=BG)
        host.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)

        canvas = tk.Canvas(host, bg=BG, highlightthickness=0)
        sb     = tk.Scrollbar(host, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")
        canvas.grid(row=0, column=0, sticky="nsew")

        self._cnn_scroll = tk.Frame(canvas, bg=BG)
        self._cnn_scroll.columnconfigure(0, weight=1)
        win = canvas.create_window((0, 0), window=self._cnn_scroll, anchor="nw")
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win, width=e.width))
        self._cnn_scroll.bind("<Configure>",
                              lambda e: canvas.configure(
                                  scrollregion=canvas.bbox("all")))

        self._cnn_placeholder()

    def _cnn_placeholder(self):
        for w in self._cnn_scroll.winfo_children():
            w.destroy()
        tk.Label(self._cnn_scroll,
                 text="Charger BloodMNIST puis lancer l'entraînement CNN",
                 bg=BG, fg=TEXT3, font=("Helvetica", 10)).pack(pady=60)

    def _render_cnn_results(self, results):
        for w in self._cnn_scroll.winfo_children():
            w.destroy()

        self._cnn_stats["accuracy"].config(
            text=f"{results['accuracy']}%")
        self._cnn_stats["epochs"].config(text=str(results["epochs"]))
        self._cnn_stats["n_train"].config(text=str(results["n_train"]))
        self._cnn_stats["n_test"].config(text=str(results["n_test"]))

        sf = self._cnn_scroll

        # Training curves (text-based sparkline)
        self._section_lbl(sf, "Courbes d'entraînement")
        curves = tk.Frame(sf, bg=WHITE)
        curves.pack(fill="x", padx=10, pady=(0, 10))
        curves.columnconfigure((0, 1), weight=1, uniform="cv")

        for col, (title, values, color) in enumerate([
            ("Val Accuracy (%)",  results["val_accs"],    C_CNN),
            ("Train Loss",        results["train_losses"], C_ERR),
        ]):
            panel = tk.Frame(curves, bg=BG)
            panel.grid(row=0, column=col, sticky="ew",
                       padx=(0 if col == 0 else 4, 4 if col == 0 else 0),
                       pady=8)
            tk.Label(panel, text=title, bg=BG, fg=TEXT3,
                     font=("Helvetica", 8, "bold")).pack(anchor="w", padx=8)
            self._sparkline(panel, values, color)

        # Per-class accuracy
        self._section_lbl(sf, "Accuracy par classe de cellule")
        bar_frame = tk.Frame(sf, bg=WHITE)
        bar_frame.pack(fill="x", padx=10, pady=(0, 10))
        per_class = results["per_class_acc"]
        for i, (cls, acc) in enumerate(zip(BLOOD_CLASSES, per_class)):
            row = tk.Frame(bar_frame, bg=WHITE)
            row.pack(fill="x", padx=10, pady=2)
            row.columnconfigure(1, weight=1)
            tk.Label(row, text=cls, bg=WHITE, fg=TEXT2,
                     font=("Helvetica", 9), width=18, anchor="w").grid(
                         row=0, column=0, padx=(8, 6))
            bar_bg = tk.Frame(row, bg=BG, height=18)
            bar_bg.grid(row=0, column=1, sticky="ew", padx=(0, 8))
            w_pct = acc / 100.0
            color = C_OK if acc >= 70 else (C_WARN if acc >= 50 else C_ERR)
            tk.Frame(bar_bg, bg=color, height=18,
                     width=max(1, int(200 * w_pct))).pack(side="left")
            tk.Label(row, text=f"{acc}%", bg=WHITE, fg=color,
                     font=("Helvetica", 9, "bold"), width=6).grid(
                         row=0, column=2, padx=(0, 8))

        # Confusion matrix
        self._section_lbl(sf, "Matrice de confusion (prédiction × réel)")
        cm_frame = tk.Frame(sf, bg=WHITE)
        cm_frame.pack(fill="x", padx=10, pady=(0, 10))
        self._draw_confusion_matrix(cm_frame, results["confusion_matrix"],
                                     color=C_CNN)

    # ── Widgets helpers ────────────────────────────────────────────────────────

    def _draw_confusion_matrix(self, parent, cm_list, color=C_RF):
        """Affiche la matrice de confusion sous forme de grille colorée."""
        cm    = np.array(cm_list)
        n     = cm.shape[0]
        max_v = cm.max() if cm.max() > 0 else 1

        grid = tk.Frame(parent, bg=WHITE)
        grid.pack(padx=10, pady=8)

        # Short class names
        short = ["Baso", "Éosi", "Éryt", "Gran",
                 "Lymp", "Mono", "Neut", "Plaq"]

        # Header row
        tk.Label(grid, text="", bg=WHITE, width=5).grid(row=0, column=0)
        for j in range(n):
            tk.Label(grid, text=short[j], bg=WHITE, fg=TEXT3,
                     font=("Helvetica", 7), width=5).grid(row=0, column=j+1)

        for i in range(n):
            tk.Label(grid, text=short[i], bg=WHITE, fg=TEXT3,
                     font=("Helvetica", 7), width=5).grid(row=i+1, column=0)
            for j in range(n):
                v = int(cm[i, j])
                intensity = v / max_v
                if i == j:
                    # Diagonal: green
                    r = int(26  + (232 - 26)  * (1 - intensity))
                    g = int(135 + (246 - 135) * (1 - intensity))
                    b = int(84  + (239 - 84)  * (1 - intensity))
                else:
                    # Off-diagonal: red tones
                    r = int(255)
                    g = int(255 - 120 * intensity)
                    b = int(255 - 120 * intensity)
                bg_hex = f"#{r:02x}{g:02x}{b:02x}"
                fg_hex = TEXT if intensity < 0.5 else WHITE
                tk.Label(grid, text=str(v), bg=bg_hex, fg=fg_hex,
                         font=("Helvetica", 7, "bold" if i == j else "normal"),
                         width=5, height=1, relief="flat").grid(
                             row=i+1, column=j+1, padx=1, pady=1)

        # Legend
        tk.Label(parent,
                 text="Lignes = classe réelle · Colonnes = classe prédite · "
                      "Diagonale = bonnes prédictions",
                 bg=WHITE, fg=TEXT3, font=("Helvetica", 7)).pack(
                     padx=10, pady=(0, 8))

    def _sparkline(self, parent, values, color):
        """Mini graphe en barres verticales pour les courbes d'entraînement."""
        if not values:
            tk.Label(parent, text="—", bg=BG, fg=TEXT3).pack()
            return
        max_v = max(values) if max(values) > 0 else 1
        height = 50
        canvas = tk.Canvas(parent, bg=BG, height=height,
                           highlightthickness=0)
        canvas.pack(fill="x", padx=8, pady=4)
        canvas.update_idletasks()
        cw = canvas.winfo_width() or 300
        bar_w = max(4, cw // len(values) - 2)
        for i, v in enumerate(values):
            bh = int(v / max_v * (height - 10))
            x0 = i * (bar_w + 2) + 4
            y0 = height - bh
            canvas.create_rectangle(x0, y0, x0 + bar_w, height,
                                     fill=color, outline="")
            canvas.create_text(x0 + bar_w // 2, y0 - 2,
                                text=str(v), font=("Helvetica", 6),
                                fill=TEXT3, anchor="s")

    def _section_lbl(self, parent, title):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", padx=10, pady=(12, 4))
        tk.Label(f, text=title, bg=BG, fg=TEXT,
                 font=("Helvetica", 10, "bold")).pack(side="left")
        tk.Frame(f, bg=BORDER, height=1).pack(
            side="left", fill="x", expand=True, padx=(8, 0), pady=6)

    def _sep(self, p):
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=10, pady=8)

    def _section(self, p, text):
        tk.Label(p, text=text, bg=WHITE, fg=TEXT3,
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=14, pady=(8, 3))

    def _slider(self, p, label, var, lo, hi, res, fmt=None):
        row = tk.Frame(p, bg=WHITE)
        row.pack(fill="x", padx=14, pady=1)
        tk.Label(row, text=label, bg=WHITE, fg=TEXT2, font=("Helvetica", 9),
                 width=14, anchor="w").pack(side="left")
        vl = tk.Label(row, text=fmt(var.get()) if fmt else str(var.get()),
                      bg=WHITE, fg=ACCENT, font=("Helvetica", 9, "bold"), width=6)
        vl.pack(side="right")
        tk.Scale(p, variable=var, from_=lo, to=hi, resolution=res,
                 orient="horizontal", bg=WHITE, troughcolor=BORDER,
                 highlightthickness=0, showvalue=False,
                 command=lambda v: vl.config(
                     text=fmt(float(v)) if fmt else str(int(float(v))))
                 ).pack(fill="x", padx=14)

    # ── Log helpers ────────────────────────────────────────────────────────────

    def _log(self, msg):
        """Append a line to the progress log (thread-safe via after())."""
        def _do():
            self._log_text.config(state="normal")
            self._log_text.insert("end", msg + "\n")
            self._log_text.see("end")
            self._log_text.config(state="disabled")
        self.after(0, _do)

    def _set_status(self, msg, color=TEXT3):
        self.after(0, lambda: self.status_lbl.config(text=msg, fg=color))

    def _set_buttons(self, load=True, rf=True, cnn=True):
        def _do():
            s_rf  = "normal" if rf  else "disabled"
            s_cnn = "normal" if cnn else "disabled"
            self.rf_btn.config(state=s_rf,
                                bg=C_RF  if rf  else TEXT3)
            self.cnn_btn.config(state=s_cnn,
                                 bg=C_CNN if cnn else TEXT3)
        self.after(0, _do)

    # ── Actions ────────────────────────────────────────────────────────────────

    def _load_data(self):
        if self._thread and self._thread.is_alive():
            return
        self._set_buttons(rf=False, cnn=False)
        self._set_status("Chargement…", C_WARN)
        self._thread = threading.Thread(
            target=self._load_data_thread, daemon=True)
        self._thread.start()

    def _load_data_thread(self):
        try:
            from core.m7_model import M7Runner
            self._runner = M7Runner()
            self._runner.load_data(
                n_train=self.n_train_var.get(),
                n_test=self.n_test_var.get(),
                progress_cb=self._log,
            )
            self._set_status("BloodMNIST chargé ✓", C_OK)
            self._set_buttons(rf=True, cnn=True)
        except ImportError as e:
            self._log(f"⚠ {e}")
            self._set_status("Installer medmnist et torch", C_ERR)
            self._set_buttons(rf=False, cnn=False)
        except Exception as e:
            self._log(f"Erreur : {e}")
            self._set_status("Erreur de chargement", C_ERR)
            self._set_buttons(rf=False, cnn=False)

    def _train_rf(self):
        if not self._runner or self._thread and self._thread.is_alive():
            return
        self._set_buttons(rf=False, cnn=False)
        self._set_status("Entraînement RF…", C_WARN)
        self._thread = threading.Thread(
            target=self._train_rf_thread, daemon=True)
        self._thread.start()

    def _train_rf_thread(self):
        try:
            results = self._runner.train_random_forest(
                n_estimators=self.n_est_var.get(),
                max_depth=self.depth_var.get(),
                progress_cb=self._log,
            )
            self.after(0, self._render_rf_results, results)
            self.after(0, self._switch_tab, "rf")
            self._set_status(
                f"RF terminé — Accuracy : {results['accuracy']}%", C_OK)
        except Exception as e:
            self._log(f"Erreur RF : {e}")
            self._set_status("Erreur RF", C_ERR)
        finally:
            self._set_buttons(rf=True, cnn=True)

    def _train_cnn(self):
        if not self._runner or self._thread and self._thread.is_alive():
            return
        self._set_buttons(rf=False, cnn=False)
        self._set_status("Entraînement CNN…", C_WARN)
        self._thread = threading.Thread(
            target=self._train_cnn_thread, daemon=True)
        self._thread.start()

    def _train_cnn_thread(self):
        try:
            results = self._runner.train_cnn(
                epochs=self.epochs_var.get(),
                batch_size=self.batch_var.get(),
                progress_cb=self._log,
            )
            self.after(0, self._render_cnn_results, results)
            self.after(0, self._switch_tab, "cnn")
            self._set_status(
                f"CNN terminé — Accuracy : {results['accuracy']}%", C_OK)
        except Exception as e:
            self._log(f"Erreur CNN : {e}")
            self._set_status("Erreur CNN", C_ERR)
        finally:
            self._set_buttons(rf=True, cnn=True)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def refresh_image(self):
        """Called by main_window when switching to this panel."""
        pass