import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

from core.histogram import image_stats
from core.cell_classifier import CellClassifier, classifier_context_block
from core.glcm import GLCMAnalyzer



WHITE   = "#FFFFFF"
BG      = "#F4F6F9"
BORDER  = "#DDE2EC"
ACCENT  = "#1A6FBF"
TEXT    = "#1C2333"
TEXT2   = "#5A6478"
TEXT3   = "#9BA5B8"

C_M1 = "#0F9D8E"; C_M1L = "#E6F7F5"
C_M2 = "#6C47CC"; C_M2L = "#F0EDFB"
C_M3 = "#C47A00"; C_M3L = "#FDF4E3"
C_M4 = "#C0393B"; C_M4L = "#FBE9E9"
C_OK = "#1A8754"; C_OKL = "#E8F6EF"

INTERP_FILTER = {
    "Médian":    "Filtre non-linéaire — élimine le bruit impulsionnel sans flouter les contours",
    "Gaussien":  "Lissage gaussien — réduit le bruit continu (vibrations, thermique)",
    "Moyenneur": "Lissage uniforme — rapide mais atténue les bords",
    "Laplacien": "Détection de contours — rehausse les structures fines",
    "—":         "Aucun filtre appliqué",
}
INTERP_NOISE = {
    "Poivre & Sel": "Bruit impulsionnel (pixels blancs/noirs aléatoires)",
    "Gaussien":     "Bruit thermique continu (distribution normale)",
    "Aucun bruit":  "Pas de bruit ajouté",
}
INTERP_HIST = {
    "Aucune":    "Histogramme affiché sans transformation",
    "stretch":   "Étirement linéaire — plage dynamique étendue à [0, 255]",
    "equalize":  "Égalisation CDF — redistribution uniforme des niveaux",
    "threshold": "Seuillage binaire — séparation fond / objets",
    "—":         "Non exécuté",
}
INTERP_FOURIER = {
    "spectrum":    "Spectre de magnitude (log) — diagnostic fréquentiel",
    "passe-bas":   "Passe-bas circulaire — atténuation du bruit haute fréquence",
    "passe-haut":  "Passe-haut circulaire — renforcement des contours cellulaires",
    "passe-bande": "Passe-bande — isolation d'une plage de textures biologiques",
    "—":           "Non exécuté",
}
INTERP_MORPHO = {
    "Érosion":    "Séparation des cellules jointives — réduction des objets",
    "Dilatation": "Fermeture des lacunes membranaires",
    "Opening":    "Suppression du bruit binaire — forme préservée",
    "Closing":    "Reconstruction des noyaux fragmentés",
    "Top Hat":    "Extraction des structures brillantes fines (granules)",
    "Black Hat":  "Extraction des structures sombres (vacuoles)",
    "Gradient":   "Contours morphologiques — tracé des membranes",
    "—":          "Non exécuté",
}


class ReportPanel(tk.Frame):

    def __init__(self, parent, state):
        super().__init__(parent, bg=BG)
        self.state   = state
        self._photos = {}
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._top_bar()
        self._body()

    def _top_bar(self):
        bar = tk.Frame(self, bg=WHITE, height=48)
        bar.grid(row=0, column=0, sticky="ew")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=BORDER, height=1).pack(fill="x", side="bottom")
        tk.Frame(bar, bg=ACCENT, width=4).pack(side="left", fill="y")
        tk.Label(bar, text="Étape 5 — Rapport d'analyse",
                 bg=WHITE, fg=ACCENT,
                 font=("Helvetica", 13, "bold")).pack(side="left", padx=14, pady=10)
        tk.Label(bar,
                 text="Résumé des opérations · Paramètres · Interprétations biologiques",
                 bg=WHITE, fg=TEXT3,
                 font=("Helvetica", 9)).pack(side="left")
        tk.Button(bar, text="Actualiser",
                  bg=ACCENT, fg=WHITE,
                  font=("Helvetica", 9, "bold"), relief="flat",
                  cursor="hand2", padx=10, pady=4,
                  command=self.refresh_image).pack(side="right", padx=12, pady=8)

    def _body(self):
        host = tk.Frame(self, bg=BG)
        host.grid(row=1, column=0, sticky="nsew")
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(host, bg=BG, highlightthickness=0)
        sb = tk.Scrollbar(host, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")
        self._canvas.grid(row=0, column=0, sticky="nsew")

        self._sf = tk.Frame(self._canvas, bg=BG)
        self._sf.columnconfigure(0, weight=1)
        self._win = self._canvas.create_window((0, 0), window=self._sf, anchor="nw")

        self._canvas.bind("<Configure>",
                          lambda e: self._canvas.itemconfig(self._win, width=e.width))
        self._sf.bind("<Configure>",
                      lambda e: self._canvas.configure(
                          scrollregion=self._canvas.bbox("all")))
        self._canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        self._placeholder()

    def _placeholder(self):
        for w in self._sf.winfo_children():
            w.destroy()
        tk.Label(self._sf,
                 text="Exécuter au moins un module puis cliquer Actualiser",
                 bg=BG, fg=TEXT3,
                 font=("Helvetica", 10)).pack(pady=80)

    def refresh_image(self):
        if not self.state.has_image():
            return
        for w in self._sf.winfo_children():
            w.destroy()

        # ── Module C — run cell classifier ────────────────────────────────────
        if self.state.m4_result is not None:
            clf = CellClassifier(n_clusters=3)
            self.state.mc_results = clf.run(
                self.state.m4_result,
                self.state.original_pixels
            )
        # ── Module D — run GLCM texture analysis ─────────────────────────────────
        src = self.state.m1_result if self.state.m1_result is not None \
            else self.state.original_pixels
        ana = GLCMAnalyzer(levels=8, distances=[1, 2])
        self.state.md_results = ana.run(src)

        sf = self._sf

        # ── Comparaison visuelle ───────────────────────────────────────────────
        self._section(sf, "Comparaison visuelle")
        vis = tk.Frame(sf, bg=BG)
        vis.pack(fill="x", padx=16, pady=(0, 8))
        vis.columnconfigure((0, 1), weight=1, uniform="v")

        for col, (title, tag, c, bgc, pix) in enumerate([
            ("Image originale", "ENTRÉE", ACCENT, "#EBF3FB",
             self.state.original_pixels),
            ("Résultat final",  "SORTIE", C_M4,   C_M4L,
             self._best()),
        ]):
            p = tk.Frame(vis, bg=WHITE)
            p.grid(row=0, column=col, sticky="nsew",
                   padx=(0 if col == 0 else 6, 6 if col == 0 else 0))
            p.columnconfigure(0, weight=1)
            p.rowconfigure(1, weight=1)

            hdr = tk.Frame(p, bg=bgc, height=32)
            hdr.grid(row=0, column=0, sticky="ew")
            hdr.grid_propagate(False)
            tk.Label(hdr, text=title, bg=bgc, fg=c,
                     font=("Helvetica", 9, "bold")).pack(
                         side="left", padx=10, pady=6)
            tk.Label(hdr, text=tag, bg=bgc, fg=c,
                     font=("Helvetica", 8)).pack(side="right", padx=10)

            cv = tk.Canvas(p, bg="#E8EDF4", height=180, highlightthickness=0)
            cv.grid(row=1, column=0, sticky="ew", padx=6, pady=6)

            if pix is not None:
                cv.update_idletasks()
                cw = cv.winfo_width() or 400
                img = Image.fromarray(pix.astype(np.uint8), mode="L")
                sc  = min(cw / img.width, 180 / img.height, 1.0)
                img = img.resize((max(1, int(img.width * sc)),
                                  max(1, int(img.height * sc))),
                                 Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                cv.after(60, lambda cv=cv, ph=photo: (
                    cv.delete("all"),
                    cv.create_image(cv.winfo_width() // 2, 90,
                                    anchor="center", image=ph)
                ))
                self._photos[f"vis_{col}"] = photo
            else:
                tk.Label(cv, text="Non calculé", bg="#E8EDF4",
                         fg=TEXT3, font=("Helvetica", 9)).place(
                             relx=.5, rely=.5, anchor="center")

        # ── Résumé des 4 étapes ───────────────────────────────────────────────
        self._section(sf, "Résumé des 4 étapes du pipeline")

        for num, name, c, bgc, log, rows in [
            ("01", "Prétraitement",       C_M1, C_M1L,
             self.state.m1_log, self._m1_rows()),
            ("02", "Histogramme",         C_M2, C_M2L,
             self.state.m2_log, self._m2_rows()),
            ("03", "Transformée Fourier", C_M3, C_M3L,
             self.state.m3_log, self._m3_rows()),
            ("04", "Morphologie",         C_M4, C_M4L,
             self.state.m4_log, self._m4_rows()),
        ]:
            self._step_card(sf, num, name, c, bgc, bool(log), rows)

        # ── Module C — Analyse cellulaire ─────────────────────────────────────
        if hasattr(self.state, "mc_results") and self.state.mc_results:
            self._section(sf, "Analyse cellulaire — Module C (k-means)")
            self._mc_card(sf, self.state.mc_results)
        
          # ── Module D — GLCM texture card ─────────────────────────────────────────
        if hasattr(self.state, "md_results") and self.state.md_results:
            self._section(sf, "Analyse de texture — Module D (GLCM Haralick)")
            self._md_card(sf, self.state.md_results)

        # ── Bilan global ──────────────────────────────────────────────────────
        self._section(sf, "Bilan global")
        self._global_card(sf)

        tk.Frame(sf, bg=BG, height=24).pack()

    # ── Cards ─────────────────────────────────────────────────────────────────

    def _step_card(self, parent, num, name, c, bgc, done, rows):
        card = tk.Frame(parent, bg=WHITE,
                        highlightbackground=c if done else BORDER,
                        highlightthickness=2 if done else 1)
        card.pack(fill="x", padx=16, pady=(0, 10))

        hdr = tk.Frame(card, bg=bgc if done else BG, height=38)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text=num, bg=bgc if done else BG, fg=c,
                 font=("Helvetica", 11, "bold"), width=3).pack(
                     side="left", padx=10, pady=6)
        tk.Label(hdr, text=name, bg=bgc if done else BG, fg=c,
                 font=("Helvetica", 11, "bold")).pack(side="left")
        tk.Label(hdr,
                 text="✓ Exécuté" if done else "— Non exécuté",
                 bg=bgc if done else BG,
                 fg=C_OK if done else TEXT3,
                 font=("Helvetica", 9, "bold")).pack(side="right", padx=12)

        if not done:
            tk.Label(card, text="Ce module n'a pas été exécuté.",
                     bg=WHITE, fg=TEXT3,
                     font=("Helvetica", 9, "italic")).pack(
                         anchor="w", padx=14, pady=10)
            return

        body = tk.Frame(card, bg=WHITE)
        body.pack(fill="x", padx=14, pady=10)

        for i, (label, value, interp) in enumerate(rows):
            row_bg = BG if i % 2 == 0 else WHITE
            row = tk.Frame(body, bg=row_bg)
            row.pack(fill="x", pady=1)
            row.columnconfigure(1, weight=0)
            row.columnconfigure(2, weight=1)

            tk.Label(row, text=label, bg=row_bg, fg=TEXT3,
                     font=("Helvetica", 8, "bold"),
                     width=18, anchor="w").grid(
                         row=0, column=0, padx=(8, 4), pady=5, sticky="w")
            tk.Label(row, text=value, bg=row_bg, fg=c,
                     font=("Helvetica", 9, "bold"),
                     anchor="w").grid(
                         row=0, column=1, padx=4, pady=5, sticky="w")
            if interp:
                tk.Label(row, text=interp, bg=row_bg, fg=TEXT2,
                         font=("Helvetica", 8),
                         anchor="w", wraplength=500).grid(
                             row=0, column=2, padx=(8, 8), pady=5, sticky="w")

    def _mc_card(self, parent, results):
        C_MC  = "#6C47CC"
        C_MCL = "#F0EDFB"

        card = tk.Frame(parent, bg=WHITE,
                        highlightbackground=C_MC, highlightthickness=2)
        card.pack(fill="x", padx=16, pady=(0, 10))

        hdr = tk.Frame(card, bg=C_MCL, height=38)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Classification cellulaire (k-means from scratch)",
                 bg=C_MCL, fg=C_MC,
                 font=("Helvetica", 10, "bold")).pack(side="left", padx=14, pady=6)
        tk.Label(hdr,
                 text=f"{results['n_objects']} objets · {results['n_clusters']} clusters",
                 bg=C_MCL, fg=C_MC,
                 font=("Helvetica", 9)).pack(side="right", padx=12)

        body = tk.Frame(card, bg=WHITE)
        body.pack(fill="x", padx=14, pady=10)
        body.columnconfigure((0, 1, 2, 3), weight=1, uniform="mc")

        labels_fr = {
            "nucleus":   "Noyaux",
            "cytoplasm": "Cytoplasme",
            "debris":    "Débris",
            "uncertain": "Incertain",
        }
        colors = {
            "nucleus":   "#1A6FBF",
            "cytoplasm": "#1A8754",
            "debris":    "#C0393B",
            "uncertain": "#9BA5B8",
        }

        for col, ctype in enumerate(["nucleus", "cytoplasm", "debris", "uncertain"]):
            count = results["summary"][ctype]
            pct   = results["summary_pct"][ctype]
            c     = colors[ctype]
            f = tk.Frame(body, bg=BG)
            f.grid(row=0, column=col, sticky="ew",
                   padx=(0 if col == 0 else 4, 4 if col < 3 else 0))
            tk.Label(f, text=labels_fr[ctype], bg=BG, fg=TEXT3,
                     font=("Helvetica", 8)).pack(pady=(8, 2))
            tk.Label(f, text=str(count), bg=BG, fg=c,
                     font=("Helvetica", 16, "bold")).pack()
            tk.Label(f, text=f"{pct}%", bg=BG, fg=c,
                     font=("Helvetica", 9)).pack(pady=(0, 8))

        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=14)
        tk.Label(card,
                 text=f"Inertie k-means : {results['inertia']}  ·  "
                      f"Variance PCA : PC1={results['pca_variance'][0]:.1f}%  "
                      f"PC2={results['pca_variance'][1]:.1f}%",
                 bg=WHITE, fg=TEXT3,
                 font=("Helvetica", 8)).pack(anchor="w", padx=14, pady=6)

    def _md_card(self, parent, results):
        """Display Module D (GLCM Haralick) results in the report."""
        C_MD  = "#C47A00"
        C_MDL = "#FDF4E3"
 
        f = results["features"]
        card = tk.Frame(parent, bg=WHITE,
                        highlightbackground=C_MD, highlightthickness=2)
        card.pack(fill="x", padx=16, pady=(0, 10))
 
        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(card, bg=C_MDL, height=38)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr,
                 text="Texture GLCM — Haralick (from scratch)",
                 bg=C_MDL, fg=C_MD,
                 font=("Helvetica", 10, "bold")).pack(
                     side="left", padx=14, pady=6)
        tk.Label(hdr,
                 text=(f"{results['levels']} niveaux · "
                       f"distances {results['distances']} · "
                       f"4 directions"),
                 bg=C_MDL, fg=C_MD,
                 font=("Helvetica", 9)).pack(side="right", padx=12)
 
        # ── Profile badge ─────────────────────────────────────────────────────
        badge = tk.Frame(card, bg=C_MDL)
        badge.pack(fill="x", padx=14, pady=(6, 2))
        tk.Label(badge,
                 text=f"Profil : {results['profile']}",
                 bg=C_MDL, fg=C_MD,
                 font=("Helvetica", 10, "bold")).pack(side="left")
 
        # ── Feature grid — 2 columns ─────────────────────────────────────────
        body = tk.Frame(card, bg=WHITE)
        body.pack(fill="x", padx=14, pady=(6, 4))
        body.columnconfigure((0, 1), weight=1, uniform="md")
 
        feature_rows = [
            ("Contraste",     f['contrast'],
             "Variation locale — élevé = texture rugueuse"),
            ("Énergie",       f['energy'],
             "Uniformité — élevé = texture homogène"),
            ("Homogénéité",   f['homogeneity'],
             "Similarité voisins — élevé = transitions douces"),
            ("Entropie",      f['entropy'],
             "Complexité — élevé = texture riche"),
            ("Corrélation",   f['correlation'],
             "Régularité directionnelle — élevé = motif répété"),
            ("Dissimilarité", f['dissimilarity'],
             "Différence linéaire — moins sensible que contraste"),
        ]
 
        for i, (label, value, tip) in enumerate(feature_rows):
            col = i % 2
            row = i // 2
            row_bg = BG if col == 0 else WHITE
            cell = tk.Frame(body, bg=row_bg)
            cell.grid(row=row, column=col, sticky="ew",
                      padx=(0 if col == 0 else 6, 6 if col == 0 else 0),
                      pady=2)
            cell.columnconfigure(1, weight=1)
 
            tk.Label(cell, text=label, bg=row_bg, fg=TEXT3,
                     font=("Helvetica", 8, "bold"),
                     width=14, anchor="w").grid(
                         row=0, column=0, padx=(8, 4), pady=4, sticky="w")
            tk.Label(cell, text=str(value), bg=row_bg, fg=C_MD,
                     font=("Helvetica", 9, "bold"),
                     anchor="w").grid(
                         row=0, column=1, padx=4, pady=4, sticky="w")
            tk.Label(cell, text=tip, bg=row_bg, fg=TEXT2,
                     font=("Helvetica", 7),
                     anchor="w", wraplength=260).grid(
                         row=1, column=0, columnspan=2,
                         padx=(8, 8), pady=(0, 4), sticky="w")
 
        # ── Interpretation ────────────────────────────────────────────────────
        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=14)
        tk.Label(card,
                 text=results["profile_detail"],
                 bg=WHITE, fg=TEXT2,
                 font=("Helvetica", 8),
                 anchor="w", justify="left",
                 wraplength=900).pack(anchor="w", padx=14, pady=8)
    
    def _global_card(self, parent):
        pix = self._best()
        if pix is None:
            tk.Label(parent, text="Aucun résultat disponible.",
                     bg=BG, fg=TEXT3,
                     font=("Helvetica", 9)).pack(padx=16)
            return

        stats      = image_stats(pix)
        done_count = sum(self.state.steps_done[:4])
        m4         = self.state.m4_log

        card = tk.Frame(parent, bg=WHITE,
                        highlightbackground=ACCENT, highlightthickness=2)
        card.pack(fill="x", padx=16, pady=(0, 8))

        hdr = tk.Frame(card, bg="#EBF3FB", height=38)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Bilan de l'analyse microscopique",
                 bg="#EBF3FB", fg=ACCENT,
                 font=("Helvetica", 11, "bold")).pack(
                     side="left", padx=14, pady=6)
        tk.Label(hdr,
                 text=f"{done_count}/4 modules complétés",
                 bg="#EBF3FB",
                 fg=C_OK if done_count == 4 else C_M3,
                 font=("Helvetica", 9, "bold")).pack(side="right", padx=12)

        body = tk.Frame(card, bg=WHITE)
        body.pack(fill="x", padx=14, pady=10)
        body.columnconfigure((0, 1, 2, 3), weight=1, uniform="gc")

        for col, (label, val, mc) in enumerate([
            ("Entropie finale",
             f"{stats.get('entropy', '—')} bits", ACCENT),
            ("Contraste final",
             f"{int(pix.max()) - int(pix.min())} niveaux", C_M2),
            ("Moyenne pixel",
             f"{stats.get('mean', '—')}", C_M3),
            ("Objets détectés",
             f"{m4.get('count', '—')}" if m4 else "M4 non exec.", C_M4),
        ]):
            f = tk.Frame(body, bg=BG)
            f.grid(row=0, column=col, sticky="ew",
                   padx=(0 if col == 0 else 6, 6 if col < 3 else 0))
            tk.Label(f, text=label, bg=BG, fg=TEXT3,
                     font=("Helvetica", 8)).pack(pady=(8, 2))
            tk.Label(f, text=val, bg=BG, fg=mc,
                     font=("Helvetica", 14, "bold")).pack(pady=(0, 8))

        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=14)

        interp_body = tk.Frame(card, bg=WHITE)
        interp_body.pack(fill="x", padx=14, pady=(8, 12))
        for line in self._interpretation(stats, m4, done_count, pix):
            tk.Label(interp_body, text=line,
                     bg=WHITE, fg=TEXT2,
                     font=("Helvetica", 9),
                     anchor="w", justify="left",
                     wraplength=1000).pack(fill="x", anchor="w", pady=1)

    # ── Row data ──────────────────────────────────────────────────────────────

    def _m1_rows(self):
        log = self.state.m1_log
        if not log:
            return []
        noise = log.get("noise", "—")
        fil   = log.get("filter", "—")
        return [
            ("Bruit ajouté",   noise,
             INTERP_NOISE.get(noise, "")),
            ("Intensité",      log.get("amount", "—"), ""),
            ("Filtre",         fil,
             INTERP_FILTER.get(fil, "")),
            ("Taille noyau",   log.get("ksize", "—"), ""),
            ("Sigma",          log.get("sigma", "—"),
             "Largeur du noyau gaussien"),
            ("MSE",            log.get("mse", "—"),
             "Mean Squared Error — distorsion introduite par le bruit"),
            ("Min / Max",
             f"{log.get('min','—')} / {log.get('max','—')}", ""),
            ("Moyenne pixel",  str(log.get("mean", "—")), ""),
        ]

    def _m2_rows(self):
        log = self.state.m2_log
        if not log:
            return []
        op   = log.get("operation", "—")
        rows = [
            ("Opération", op, INTERP_HIST.get(op, "")),
            ("Bins", str(log.get("bins", "—")),
             "Résolution de l'histogramme"),
        ]
        if op == "threshold":
            rows.append(("Seuil", str(log.get("seuil", "—")),
                         "Valeur de coupure fond / objet"))
        rows += [
            ("Min",        str(log.get("min", "—")),     ""),
            ("Max",        str(log.get("max", "—")),     ""),
            ("Moyenne",    str(log.get("mean", "—")),    ""),
            ("Écart-type", str(log.get("std", "—")),     ""),
            ("Entropie",
             f"{log.get('entropy','—')} bits",
             "> 6 bits = bonne richesse informationnelle"),
        ]
        return rows

    def _m3_rows(self):
        log = self.state.m3_log
        if not log:
            return []
        ft = log.get("filter", "—")
        return [
            ("Algorithme",
             log.get("algo", "Cooley-Tukey radix-2"),
             "FFT implémentée from scratch — sans numpy.fft"),
            ("Type de filtre", ft, INTERP_FOURIER.get(ft, "")),
            ("Rayon coupure",
             f"{log.get('radius','—')} px",
             "Fréquence de coupure en pixels"),
        ]

    def _m4_rows(self):
        log = self.state.m4_log
        if not log:
            return []
        op = log.get("operation", "—")
        return [
            ("Opération",          op,
             INTERP_MORPHO.get(op, "")),
            ("Seuil binarisation", str(log.get("seuil", "—")),
             "Sépare fond (noir) des objets (blanc)"),
            ("Itérations",         str(log.get("iterations", "—")),
             "Nombre d'applications de l'opérateur"),
            ("Objets détectés",    str(log.get("count", "—")),
             "Comptage par flood fill 4-connexe"),
            ("Aire moyenne",
             f"{log.get('mean_area','—')} px²",
             "Surface moyenne des objets segmentés"),
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", padx=16, pady=(16, 6))
        tk.Label(f, text=title, bg=BG, fg=TEXT,
                 font=("Helvetica", 11, "bold")).pack(side="left")
        tk.Frame(f, bg=BORDER, height=1).pack(
            side="left", fill="x", expand=True, padx=(10, 0), pady=6)

    def _best(self):
        for r in [self.state.m4_result, self.state.m3_result,
                  self.state.m2_result, self.state.m1_result]:
            if r is not None:
                return r
        return self.state.original_pixels

    def _interpretation(self, stats, m4log, done_count, pix):
        lines = []
        done_names = [n for n, d in zip(
            ["M1 Prétraitement", "M2 Histogramme",
             "M3 Fourier", "M4 Morphologie"],
            self.state.steps_done[:4]) if d]
        lines.append(
            f"Pipeline exécuté : "
            f"{' → '.join(done_names) if done_names else 'aucun module'}.")

        entropy  = float(stats.get("entropy", 0))
        contrast = int(pix.max()) - int(pix.min())
        mean     = float(stats.get("mean", 128))

        if entropy >= 6.0:
            lines.append(
                f"Entropie {entropy} bits — image riche, "
                f"distribution équilibrée.")
        elif entropy >= 4.0:
            lines.append(
                f"Entropie {entropy} bits — contraste modéré. "
                f"Appliquer une égalisation (M2) pour enrichir.")
        else:
            lines.append(
                f"Entropie {entropy} bits — image peu contrastée. "
                f"Étirement ou égalisation fortement recommandé.")

        if contrast >= 180:
            lines.append(
                f"Contraste {contrast} niveaux — dynamique étendue, "
                f"structures bien différenciées.")
        elif contrast >= 100:
            lines.append(
                f"Contraste {contrast} niveaux — acceptable, "
                f"amélioration possible par M2.")
        else:
            lines.append(
                f"Contraste {contrast} niveaux — insuffisant "
                f"pour une segmentation fiable.")

        if mean < 80:
            lines.append(
                f"Luminosité moyenne {mean} — image sombre "
                f"(sous-exposition ou fluorescence).")
        elif mean > 180:
            lines.append(
                f"Luminosité moyenne {mean} — image claire "
                f"(surexposition possible).")
        else:
            lines.append(
                f"Luminosité moyenne {mean} — exposition correcte.")

        if m4log:
            count = m4log.get("count", 0)
            area  = m4log.get("mean_area", 0)
            op    = m4log.get("operation", "—")
            lines.append(
                f"Segmentation {op} : {count} objet(s) détecté(s), "
                f"aire moyenne {area} px².")
            if count == 0:
                lines.append(
                    "→ Aucun objet : ajuster le seuil ou améliorer "
                    "le contraste (M2) avant de relancer M4.")
            elif area and float(area) < 20:
                lines.append(
                    "→ Objets très petits : probable bruit résiduel — "
                    "appliquer Opening avant comptage.")
        else:
            lines.append(
                "Segmentation (M4) non exécutée — lancer la morphologie "
                "pour le comptage cellulaire.")
        return lines