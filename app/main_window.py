import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image

from app.state import AppState
from core.histogram import to_grayscale
from ui.filter_panel  import FilterPanel
from ui.hist_panel    import HistPanel
from ui.fourier_panel import FourierPanel
from ui.morpho_panel  import MorphoPanel
from ui.report_panel  import ReportPanel
from ui.ai_panel      import AiPanel

BG        = "#F4F6F9"
WHITE     = "#FFFFFF"
BORDER    = "#DDE2EC"
ACCENT    = "#1A6FBF"
TEXT      = "#1C2333"
TEXT2     = "#5A6478"
TEXT3     = "#9BA5B8"

STEP_COLORS = [
    ("#0F9D8E", "#E6F7F5"),
    ("#6C47CC", "#F0EDFB"),
    ("#C47A00", "#FDF4E3"),
    ("#C0393B", "#FBE9E9"),
    ("#1A6FBF", "#EBF3FB"),
    ("#6C47CC", "#F0EDFB"),  # IA — violet
]
STEP_LABELS = [
    ("01", "Prétraitement",  "Débruitage"),
    ("02", "Histogramme",    "Contraste"),
    ("03", "Fourier",        "Fréquentiel"),
    ("04", "Morphologie",    "Segmentation"),
    ("05", "Rapport",        "Interprétation"),
    ("06", "IA",             "Prédiction"),
]


class MainWindow:

    def __init__(self, root):
        self.root   = root
        self.state  = AppState()
        self.state.on_step_complete = self._on_step_done
        self._active_step = 0
        self._step_frames = []
        self._panels = []
        self._setup_root()
        self._setup_styles()
        self._build()

    def _setup_root(self):
        self.root.title("BioVision Lab — Pipeline d'Analyse Microscopique")
        self.root.geometry("1360x860")
        self.root.minsize(1000, 660)
        self.root.configure(bg=BG)

    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TFrame",       background=BG)
        s.configure("White.TFrame", background=WHITE)

    def _build(self):
        # ── Header ─────────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=WHITE, height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Frame(hdr, bg=ACCENT, width=4).pack(side="left", fill="y")
        tk.Label(hdr, text="BioVision", bg=WHITE, fg=ACCENT,
                 font=("Helvetica", 17, "bold")).pack(side="left", padx=(14,0), pady=14)
        tk.Label(hdr, text=" Lab", bg=WHITE, fg=TEXT,
                 font=("Helvetica", 17, "bold")).pack(side="left", pady=14)
        tk.Label(hdr, text="  ·  Analyse d'image microscopique  ·  Master BIAM",
                 bg=WHITE, fg=TEXT3, font=("Helvetica", 9)).pack(side="left", pady=14)

        self.img_lbl = tk.Label(hdr, text="Aucun échantillon chargé",
                                 bg=WHITE, fg=TEXT3, font=("Helvetica", 9))
        self.img_lbl.pack(side="right", padx=16)

        tk.Button(hdr, text="  Charger un échantillon  ",
                  bg=ACCENT, fg=WHITE, font=("Helvetica", 9, "bold"),
                  relief="flat", cursor="hand2", padx=4, pady=6,
                  command=self._load_image).pack(side="right", padx=12, pady=10)

        tk.Button(hdr, text="Guide d'utilisation",
                  bg=WHITE, fg=ACCENT, font=("Helvetica", 9),
                  relief="flat", cursor="hand2", padx=4, pady=6,
                  highlightbackground=BORDER, highlightthickness=1,
                  command=self._open_guide).pack(side="right", padx=4, pady=10)

        # ── Pipeline bar ───────────────────────────────────────────────────────
        pipe_bar = tk.Frame(self.root, bg=WHITE, height=70)
        pipe_bar.pack(fill="x")
        pipe_bar.pack_propagate(False)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        inner = tk.Frame(pipe_bar, bg=WHITE)
        inner.pack(expand=True, pady=10)

        for i, ((num, title, sub), (color, bg_lt)) in enumerate(
                zip(STEP_LABELS, STEP_COLORS)):
            if i > 0:
                tk.Frame(inner, bg=BORDER, width=26, height=1).pack(
                    side="left", pady=8)

            sf = tk.Frame(inner, bg=WHITE, cursor="hand2")
            sf.pack(side="left")
            sf.bind("<Button-1>", lambda e, idx=i: self._switch_step(idx))
            self._step_frames.append((sf, color, bg_lt))

            circle = tk.Label(sf, text=num, bg=bg_lt, fg=color,
                               font=("Helvetica", 10, "bold"), width=3, pady=2)
            circle.pack(side="left", padx=(0, 6))
            circle.bind("<Button-1>", lambda e, idx=i: self._switch_step(idx))

            col = tk.Frame(sf, bg=WHITE)
            col.pack(side="left")
            col.bind("<Button-1>", lambda e, idx=i: self._switch_step(idx))

            t_lbl = tk.Label(col, text=title, bg=WHITE, fg=TEXT,
                              font=("Helvetica", 10, "bold"), anchor="w")
            t_lbl.pack(anchor="w")
            t_lbl.bind("<Button-1>", lambda e, idx=i: self._switch_step(idx))

            s_lbl = tk.Label(col, text=sub, bg=WHITE, fg=TEXT3,
                              font=("Helvetica", 8), anchor="w")
            s_lbl.pack(anchor="w")
            s_lbl.bind("<Button-1>", lambda e, idx=i: self._switch_step(idx))

        # ── Step containers ────────────────────────────────────────────────────
        self.content = tk.Frame(self.root, bg=BG)
        self.content.pack(fill="both", expand=True)

        for PanelCls in [FilterPanel, HistPanel, FourierPanel,
                         MorphoPanel, ReportPanel, AiPanel]:
            p = PanelCls(self.content, self.state)
            p.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._panels.append(p)

        # ── Status bar ─────────────────────────────────────────────────────────
        sb = tk.Frame(self.root, bg=WHITE, height=26)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", side="top")
        self.status_lbl = tk.Label(
            sb, text="Prêt — charger un échantillon microscopique",
            bg=WHITE, fg=TEXT3, font=("Helvetica", 8))
        self.status_lbl.pack(side="left", padx=16)
        tk.Label(sb,
                 text="Algorithmes 100% manuels · numpy · pillow · matplotlib",
                 bg=WHITE, fg=TEXT3,
                 font=("Helvetica", 8)).pack(side="right", padx=16)

        self._switch_step(0)

    # ── Navigation ─────────────────────────────────────────────────────────────

    def _switch_step(self, idx):
        self._active_step = idx
        self._panels[idx].lift()
        if idx == 4:
            self._panels[4].refresh_image()
        if idx == 5:
            self._panels[5].refresh_image()
        for i, (sf, color, bg_lt) in enumerate(self._step_frames):
            active = (i == idx)
            sf.config(bg=bg_lt if active else WHITE)
            for w in sf.winfo_children():
                w.config(bg=bg_lt if active else WHITE)
                for ww in w.winfo_children():
                    ww.config(bg=bg_lt if active else WHITE)

    def _on_step_done(self, idx):
        sf, color, bg_lt = self._step_frames[idx]
        circle = sf.winfo_children()[0]
        circle.config(text="✓")
        self.status_lbl.config(
            text=f"Étape {idx+1} complète — {STEP_LABELS[idx][1]}")
        if idx + 1 < len(self._panels):
            self._panels[idx + 1].refresh_image()

    # ── Guide ──────────────────────────────────────────────────────────────────

    def _open_guide(self):
        import os, sys, subprocess
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "assets", "guide_utilisation.docx")
        if not os.path.exists(path):
            messagebox.showinfo("Guide", f"Fichier introuvable :\n{path}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            messagebox.showinfo("Guide", f"Ouvrir manuellement :\n{path}")

    # ── Image loading ──────────────────────────────────────────────────────────

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Charger un échantillon microscopique",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("Tous fichiers", "*.*")])
        if not path:
            return
        try:
            img  = Image.open(path).convert("RGB")
            arr  = np.array(img, dtype=np.uint8)
            gray = to_grayscale(arr)

            self.state.filepath        = path
            self.state.filename        = path.split("/")[-1]
            self.state.original_pixels = gray
            self.state.img_w           = gray.shape[1]
            self.state.img_h           = gray.shape[0]
            self.state.reset()

            self.img_lbl.config(
                text=f"{self.state.filename}  ·  {self.state.img_w}×{self.state.img_h} px")
            self.status_lbl.config(
                text=f"Échantillon chargé : {self.state.filename} "
                     f"({self.state.img_w}×{self.state.img_h})")

            for p in self._panels:
                p.refresh_image()

            for i, (sf, color, bg_lt) in enumerate(self._step_frames):
                sf.winfo_children()[0].config(text=STEP_LABELS[i][0])

        except Exception as e:
            messagebox.showerror("Erreur de chargement", str(e))