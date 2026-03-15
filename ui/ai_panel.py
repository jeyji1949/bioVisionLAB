import tkinter as tk
from tkinter import scrolledtext
import threading
import json
import urllib.request
import urllib.error
import base64
import numpy as np
from PIL import Image
import io

from core.histogram import image_stats

WHITE   = "#FFFFFF"
BG      = "#F4F6F9"
BORDER  = "#DDE2EC"
ACCENT  = "#1A6FBF"
TEXT    = "#1C2333"
TEXT2   = "#5A6478"
TEXT3   = "#9BA5B8"
C_AI    = "#6C47CC"
C_AIL   = "#F0EDFB"
C_USER  = "#1A6FBF"
C_USERL = "#EBF3FB"
C_OK    = "#1A8754"
C_OKL   = "#E8F6EF"
C_ERR   = "#C0393B"
C_ERRL  = "#FBE9E9"

#API_URL = "https://api.anthropic.com/v1/messages"
#MODEL   = "claude-haiku-4-5-20251001"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
MODEL   = "gemini-2.0-flash"

SYSTEM_PROMPT = """Tu es un expert en bioinformatique et en traitement d'image médicale, \
spécialisé dans l'analyse d'images microscopiques. Tu travailles avec BioVision Lab, \
une application de traitement d'image développée dans le cadre du Master BIAM \
(Bioinformatique et Intelligence Artificielle pour la Médecine de Précision).

Tu reçois les données complètes d'un pipeline d'analyse d'image microscopique \
comprenant 4 modules : Prétraitement (M1), Histogramme (M2), Fourier (M3), \
Morphologie (M4). Tu dois :

1. Identifier le type probable d'image biologique selon les statistiques
2. Interpréter les résultats de chaque module de façon biologique et clinique
3. Évaluer la qualité du traitement appliqué
4. Suggérer des améliorations concrètes si nécessaire
5. Générer un rapport médical structuré et professionnel en français

Réponds toujours en français. Sois précis, professionnel, et adapte ton langage \
au niveau Master en bioinformatique. Structure tes réponses avec des sections claires."""


def _build_context(state) -> str:
    """Construire le contexte complet du pipeline pour l'IA."""
    lines = []
    lines.append("=== DONNÉES DU PIPELINE BIOVISION LAB ===\n")

    # Image info
    lines.append(f"Fichier : {state.filename or 'inconnu'}")
    lines.append(f"Dimensions : {state.img_w} × {state.img_h} px\n")

    # Stats image originale
    if state.original_pixels is not None:
        s = image_stats(state.original_pixels)
        lines.append("--- Image originale ---")
        lines.append(f"Min={s['min']}  Max={s['max']}  Moyenne={s['mean']}  "
                     f"Écart-type={s['std']}  Entropie={s['entropy']} bits")
        contrast = int(state.original_pixels.max()) - int(state.original_pixels.min())
        lines.append(f"Contraste={contrast} niveaux\n")

    # M1 log
    if state.m1_log:
        l = state.m1_log
        lines.append("--- Module 1 — Prétraitement ---")
        lines.append(f"Bruit ajouté : {l.get('noise','—')} (intensité {l.get('amount','—')})")
        lines.append(f"Filtre appliqué : {l.get('filter','—')} (noyau {l.get('ksize','—')}, sigma {l.get('sigma','—')})")
        lines.append(f"MSE (distorsion bruit) : {l.get('mse','—')}")
        lines.append(f"Après filtrage — Min={l.get('min','—')}  Max={l.get('max','—')}  Moyenne={l.get('mean','—')}\n")

    # M2 log
    if state.m2_log:
        l = state.m2_log
        lines.append("--- Module 2 — Histogramme ---")
        lines.append(f"Opération : {l.get('operation','—')} (bins={l.get('bins','—')})")
        if l.get('operation') == 'threshold':
            lines.append(f"Seuil de binarisation : {l.get('seuil','—')}")
        lines.append(f"Min={l.get('min','—')}  Max={l.get('max','—')}  "
                     f"Moyenne={l.get('mean','—')}  Écart-type={l.get('std','—')}")
        lines.append(f"Entropie après traitement : {l.get('entropy','—')} bits\n")

    # M3 log
    if state.m3_log:
        l = state.m3_log
        lines.append("--- Module 3 — Transformée de Fourier ---")
        lines.append(f"Algorithme : {l.get('algo','Cooley-Tukey radix-2 from scratch')}")
        lines.append(f"Type de filtre : {l.get('filter','—')} (rayon={l.get('radius','—')} px)\n")

    # M4 log
    if state.m4_log:
        l = state.m4_log
        lines.append("--- Module 4 — Morphologie ---")
        lines.append(f"Opération : {l.get('operation','—')} "
                     f"(seuil={l.get('seuil','—')}, itérations={l.get('iterations','—')})")
        lines.append(f"Objets détectés : {l.get('count','—')}")
        lines.append(f"Aire moyenne des objets : {l.get('mean_area','—')} px²\n")

    # Stats résultat final
    best = None
    for r in [state.m4_result, state.m3_result, state.m2_result, state.m1_result]:
        if r is not None:
            best = r
            break
    if best is None:
        best = state.original_pixels

    if best is not None:
        s = image_stats(best)
        lines.append("--- Résultat final du pipeline ---")
        lines.append(f"Min={s['min']}  Max={s['max']}  Moyenne={s['mean']}  "
                     f"Écart-type={s['std']}  Entropie={s['entropy']} bits")
        contrast = int(best.max()) - int(best.min())
        lines.append(f"Contraste={contrast} niveaux")

    done = sum(state.steps_done[:4])
    lines.append(f"\nModules complétés : {done}/4")

    return "\n".join(lines)


def _pixels_to_base64(pixels: np.ndarray, max_size: int = 256) -> str:
    """Convertir un array numpy en base64 PNG pour l'API vision."""
    img = Image.fromarray(pixels.astype(np.uint8), mode="L").convert("RGB")
    # Resize si trop grande
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class AiPanel(tk.Frame):

    def __init__(self, parent, state):
        super().__init__(parent, bg=BG)
        self.state    = state
        self._history = []   # [{"role": "user"|"assistant", "content": str}]
        self._loading = False
        self._build()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._sidebar()
        self._chat_area()

    def _sidebar(self):
        sb = tk.Frame(self, bg=WHITE, width=240)
        sb.grid(row=0, column=0, sticky="ns")
        sb.grid_propagate(False)
        tk.Frame(sb, bg=BORDER, width=1).place(relx=1, rely=0, relheight=1)

        # Title
        title_bar = tk.Frame(sb, bg=C_AIL, height=56)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)
        tk.Frame(title_bar, bg=C_AI, width=3).pack(side="left", fill="y")
        col = tk.Frame(title_bar, bg=C_AIL)
        col.pack(side="left", padx=12, pady=10)
        tk.Label(col, text="Étape 6 — IA & Prédiction",
                 bg=C_AIL, fg=C_AI,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Label(col, text="Analyse intelligente du pipeline",
                 bg=C_AIL, fg=C_AI,
                 font=("Helvetica", 8)).pack(anchor="w")

        body = tk.Frame(sb, bg=WHITE)
        body.pack(fill="both", expand=True)

        # Model info
        badge = tk.Frame(body, bg=C_AIL)
        badge.pack(fill="x", padx=12, pady=(14, 0))
        tk.Label(badge, text="Modèle",
                 bg=C_AIL, fg=C_AI,
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=8, pady=(5,0))
        tk.Label(badge, text=MODEL,
                 bg=C_AIL, fg=C_AI,
                 font=("Helvetica", 8)).pack(anchor="w", padx=8, pady=(0,5))

        self._sep(body)

        # API Key
        self._section(body, "CLÉ API GEMINI")
        tk.Label(body, text="Requis pour utiliser l'IA",
                 bg=WHITE, fg=TEXT3,
                 font=("Helvetica", 8)).pack(anchor="w", padx=14, pady=(0,4))
        self.api_key_var = tk.StringVar(value="AIzaSyB50-0bcalnTtnp3q4ZHMKuv7hvN8br40A")
        key_entry = tk.Entry(body, textvariable=self.api_key_var,
                             show="•", font=("Helvetica", 9),
                             bg=BG, fg=TEXT, relief="flat",
                             highlightbackground=BORDER,
                             highlightthickness=1)
        key_entry.pack(fill="x", padx=12, pady=(0,4))
        tk.Label(body, text="La clé n'est jamais sauvegardée",
                 bg=WHITE, fg=TEXT3,
                 font=("Helvetica", 7, "italic")).pack(anchor="w", padx=14)

        self._sep(body)

        # Quick prompts
        self._section(body, "ANALYSES RAPIDES")
        quick_prompts = [
            ("Analyser le pipeline complet",
             "Analyse complète du pipeline : identifie le type d'image, "
             "évalue la qualité de chaque étape et génère un rapport médical structuré."),
            ("Identifier le type d'image",
             "En te basant sur les statistiques (entropie, contraste, distribution), "
             "quel type d'image biologique s'agit-il probablement ? "
             "(cellules, bactéries, tissu, fluorescence, etc.)"),
            ("Suggérer des améliorations",
             "Analyse les paramètres utilisés dans chaque module et suggère "
             "des améliorations concrètes pour optimiser la segmentation cellulaire."),
            ("Rapport médical complet",
             "Génère un rapport médical complet et structuré de cette analyse "
             "microscopique, avec introduction, méthodes, résultats et conclusion."),
            ("Évaluer la qualité",
             "Évalue la qualité du traitement appliqué : le bruit a-t-il été "
             "bien éliminé ? Le contraste est-il suffisant ? La segmentation "
             "est-elle fiable ?"),
        ]
        for label, prompt in quick_prompts:
            btn = tk.Button(body, text=label,
                            bg=BG, fg=ACCENT,
                            font=("Helvetica", 9), relief="flat",
                            cursor="hand2", anchor="w",
                            padx=8, pady=4,
                            command=lambda p=prompt: self._send(p))
            btn.pack(fill="x", padx=8, pady=1)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=C_AIL))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=BG))

        self._sep(body)

        # Clear
        tk.Button(body, text="Effacer la conversation",
                  bg=BG, fg=TEXT3,
                  font=("Helvetica", 9), relief="flat",
                  cursor="hand2", pady=4,
                  command=self._clear).pack(fill="x", padx=12, pady=(0,12))

        # Status
        self.status_lbl = tk.Label(body, text="Prêt",
                                    bg=WHITE, fg=TEXT3,
                                    font=("Helvetica", 8))
        self.status_lbl.pack(padx=14, anchor="w")

    def _chat_area(self):
        main = tk.Frame(self, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=0)

        # ── Messages ──────────────────────────────────────────────────────────
        chat_frame = tk.Frame(main, bg=WHITE)
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 4))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)

        hdr = tk.Frame(chat_frame, bg=C_AIL, height=34)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Conversation avec l'IA",
                 bg=C_AIL, fg=C_AI,
                 font=("Helvetica", 10, "bold")).pack(side="left", padx=12, pady=7)
        self.model_lbl = tk.Label(hdr, text=MODEL,
                                   bg=C_AIL, fg=C_AI,
                                   font=("Helvetica", 8))
        self.model_lbl.pack(side="right", padx=12)

        self._msg_canvas = tk.Canvas(chat_frame, bg=WHITE, highlightthickness=0)
        msg_sb = tk.Scrollbar(chat_frame, orient="vertical",
                               command=self._msg_canvas.yview)
        self._msg_canvas.configure(yscrollcommand=msg_sb.set)
        msg_sb.pack(side="right", fill="y")
        self._msg_canvas.pack(side="left", fill="both", expand=True)

        self._msg_frame = tk.Frame(self._msg_canvas, bg=WHITE)
        self._msg_frame.columnconfigure(0, weight=1)
        self._msg_win = self._msg_canvas.create_window(
            (0, 0), window=self._msg_frame, anchor="nw")

        self._msg_canvas.bind(
            "<Configure>",
            lambda e: self._msg_canvas.itemconfig(self._msg_win, width=e.width))
        self._msg_frame.bind(
            "<Configure>",
            lambda e: self._msg_canvas.configure(
                scrollregion=self._msg_canvas.bbox("all")))
        self._msg_canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._msg_canvas.yview_scroll(
                int(-1*(e.delta/120)), "units"))

        self._welcome()

        # ── Input bar ─────────────────────────────────────────────────────────
        input_bar = tk.Frame(main, bg=WHITE,
                              highlightbackground=BORDER, highlightthickness=1)
        input_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        input_bar.columnconfigure(0, weight=1)

        self._input = tk.Text(input_bar, height=3,
                               font=("Helvetica", 10),
                               bg=WHITE, fg=TEXT, relief="flat",
                               wrap="word", padx=10, pady=8)
        self._input.grid(row=0, column=0, sticky="ew")
        self._input.bind("<Return>",    self._on_enter)
        self._input.bind("<Shift-Return>", lambda e: None)
        self._input.insert("1.0", "Posez votre question sur l'image...")
        self._input.config(fg=TEXT3)
        self._input.bind("<FocusIn>",  self._clear_placeholder)
        self._input.bind("<FocusOut>", self._restore_placeholder)

        send_frame = tk.Frame(input_bar, bg=WHITE)
        send_frame.grid(row=0, column=1, padx=8, pady=6)
        self._send_btn = tk.Button(
            send_frame, text="Envoyer",
            bg=C_AI, fg=WHITE,
            font=("Helvetica", 9, "bold"), relief="flat",
            cursor="hand2", padx=12, pady=6,
            command=lambda: self._send(self._get_input()))
        self._send_btn.pack()

        tk.Label(input_bar, text="Entrée pour envoyer · Maj+Entrée pour saut de ligne",
                 bg=WHITE, fg=TEXT3,
                 font=("Helvetica", 7)).grid(
                     row=1, column=0, columnspan=2,
                     sticky="w", padx=10, pady=(0,4))

    # ── Welcome message ───────────────────────────────────────────────────────

    def _welcome(self):
        self._add_message(
            "assistant",
            "Bonjour ! Je suis votre assistant IA spécialisé en analyse d'image "
            "microscopique.\n\n"
            "Je peux analyser les résultats de votre pipeline BioVision Lab et vous "
            "fournir :\n"
            "• Une identification du type d'image biologique\n"
            "• Une interprétation des statistiques (entropie, contraste, MSE...)\n"
            "• Des suggestions d'amélioration pour chaque module\n"
            "• Un rapport médical complet en français\n\n"
            "Commencez par entrer votre clé API Anthropic dans la sidebar, "
            "puis utilisez un bouton d'analyse rapide ou posez directement votre question."
        )

    # ── Message display ───────────────────────────────────────────────────────

    def _add_message(self, role: str, text: str):
        is_ai   = (role == "assistant")
        bg_msg  = C_AIL  if is_ai else C_USERL
        fg_msg  = C_AI   if is_ai else C_USER
        label   = "IA — BioVision" if is_ai else "Vous"
        align   = "w"

        outer = tk.Frame(self._msg_frame, bg=WHITE)
        outer.pack(fill="x", padx=12, pady=4)
        outer.columnconfigure(0, weight=1)

        # Role label
        tk.Label(outer, text=label,
                 bg=WHITE, fg=fg_msg,
                 font=("Helvetica", 8, "bold")).pack(anchor=align)

        # Bubble
        bubble = tk.Frame(outer, bg=bg_msg)
        bubble.pack(anchor=align, fill="x" if is_ai else "none",
                    ipadx=0)

        msg_lbl = tk.Label(bubble, text=text,
                            bg=bg_msg, fg=TEXT,
                            font=("Helvetica", 10),
                            wraplength=800,
                            justify="left", anchor="w",
                            padx=12, pady=8)
        msg_lbl.pack(fill="x", anchor="w")

        self._msg_frame.update_idletasks()
        self._msg_canvas.yview_moveto(1.0)

    def _add_loading(self):
        self._loading_frame = tk.Frame(self._msg_frame, bg=WHITE)
        self._loading_frame.pack(fill="x", padx=12, pady=4)
        tk.Label(self._loading_frame, text="IA — BioVision",
                 bg=WHITE, fg=C_AI,
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        self._loading_lbl = tk.Label(
            self._loading_frame,
            text="Analyse en cours...",
            bg=C_AIL, fg=C_AI,
            font=("Helvetica", 10, "italic"),
            padx=12, pady=8)
        self._loading_lbl.pack(anchor="w")
        self._msg_canvas.yview_moveto(1.0)

    def _remove_loading(self):
        if hasattr(self, "_loading_frame"):
            self._loading_frame.destroy()

    # ── Input helpers ─────────────────────────────────────────────────────────

    def _get_input(self) -> str:
        text = self._input.get("1.0", "end-1c").strip()
        if text == "Posez votre question sur l'image...":
            return ""
        return text

    def _clear_input(self):
        self._input.delete("1.0", "end")
        self._input.config(fg=TEXT)

    def _on_enter(self, event):
        if event.state & 0x1:  # Shift held
            return
        text = self._get_input()
        if text:
            self._send(text)
        return "break"

    def _clear_placeholder(self, event):
        if self._input.get("1.0", "end-1c") == "Posez votre question sur l'image...":
            self._input.delete("1.0", "end")
            self._input.config(fg=TEXT)

    def _restore_placeholder(self, event):
        if not self._input.get("1.0", "end-1c").strip():
            self._input.insert("1.0", "Posez votre question sur l'image...")
            self._input.config(fg=TEXT3)

    # ── Send & API ────────────────────────────────────────────────────────────

    def _send(self, text: str):
        if not text or self._loading:
            return

        api_key = self.api_key_var.get().strip()
        if not api_key:
            self._add_message("assistant",
                               "⚠ Veuillez entrer votre clé API Anthropic dans la sidebar.")
            return

        if not self.state.has_image():
            self._add_message("assistant",
                               "⚠ Aucune image chargée. Veuillez charger une image "
                               "et exécuter les modules avant d'utiliser l'IA.")
            return

        self._clear_input()
        self._add_message("user", text)
        self._history.append({"role": "user", "content": text})

        self._loading = True
        self._send_btn.config(state="disabled", bg=TEXT3)
        self.status_lbl.config(text="Analyse en cours...", fg=C_AI)
        self._add_loading()

        threading.Thread(target=self._call_api,
                         args=(api_key, text),
                         daemon=True).start()

    def _call_api(self, api_key: str, user_text: str):
        try:
            context = _build_context(self.state)
            full_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Données pipeline :\n{context}\n\n"
                f"Question : {user_text}"
            )

            payload = json.dumps({
                "contents": [{"parts": [{"text": full_prompt}]}]
            }).encode("utf-8")

            url = f"{API_URL}?key={api_key}"
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            reply = data["candidates"][0]["content"]["parts"][0]["text"]
            self._history.append({"role": "assistant", "content": reply})
            self.after(0, self._on_reply, reply, None)

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                err_msg = json.loads(body).get("error", {}).get("message", body)
            except Exception:
                err_msg = body[:300]
            self.after(0, self._on_reply, None, f"Erreur API ({e.code}) : {err_msg}")
        except Exception as e:
            self.after(0, self._on_reply, None, f"Erreur : {str(e)}")

    def _on_reply(self, reply, error):
        self._remove_loading()
        self._loading = False
        self._send_btn.config(state="normal", bg=C_AI)

        if error:
            self.status_lbl.config(text="Erreur", fg=C_ERR)
            self._add_message("assistant", f"⚠ {error}")
        else:
            self.status_lbl.config(text="Réponse reçue", fg=C_OK)
            self._add_message("assistant", reply)

    def _clear(self):
        self._history.clear()
        for w in self._msg_frame.winfo_children():
            w.destroy()
        self._welcome()
        self.status_lbl.config(text="Conversation effacée", fg=TEXT3)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sep(self, p):
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=10, pady=8)

    def _section(self, p, text):
        tk.Label(p, text=text, bg=WHITE, fg=TEXT3,
                 font=("Helvetica", 8, "bold")).pack(
                     anchor="w", padx=14, pady=(8,3))

    def refresh_image(self):
        # Quand on arrive sur ce panel, mettre à jour le status
        if self.state.has_image():
            done = sum(self.state.steps_done[:4])
            self.status_lbl.config(
                text=f"Image chargée · {done}/4 modules", fg=C_OK)
