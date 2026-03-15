import tkinter as tk
import threading
import json
import urllib.request
import urllib.error
import numpy as np
import time

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
C_ERR   = "#C0393B"

# ── Providers disponibles ─────────────────────────────────────────────────────
PROVIDERS = {
    "OpenRouter (gratuit)": {
        "url":     "https://openrouter.ai/api/v1/chat/completions",
        "model":   "arcee-ai/trinity-large-preview:free",
        "type":    "openrouter",
        "hint":    "Clé sur openrouter.ai — gratuit sans quota strict",
    },
    "Gemini Flash (Google)": {
        "url":     "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        "model":   "gemini-2.0-flash",
        "type":    "gemini",
        "hint":    "Clé sur aistudio.google.com — 15 req/min gratuit",
    },
    "Anthropic Claude": {
        "url":     "https://api.anthropic.com/v1/messages",
        "model":   "claude-haiku-4-5-20251001",
        "type":    "anthropic",
        "hint":    "Clé sur console.anthropic.com — payant",
    },
}

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
    lines = ["=== DONNÉES DU PIPELINE BIOVISION LAB ===\n"]
    lines.append(f"Fichier : {state.filename or 'inconnu'}")
    lines.append(f"Dimensions : {state.img_w} × {state.img_h} px\n")

    if state.original_pixels is not None:
        s = image_stats(state.original_pixels)
        lines.append("--- Image originale ---")
        lines.append(f"Min={s['min']}  Max={s['max']}  Moyenne={s['mean']}  "
                     f"Écart-type={s['std']}  Entropie={s['entropy']} bits")
        lines.append(f"Contraste={int(state.original_pixels.max())-int(state.original_pixels.min())} niveaux\n")

    if state.m1_log:
        l = state.m1_log
        lines.append("--- Module 1 — Prétraitement ---")
        lines.append(f"Bruit : {l.get('noise','—')} intensité={l.get('amount','—')}")
        lines.append(f"Filtre : {l.get('filter','—')} noyau={l.get('ksize','—')} sigma={l.get('sigma','—')}")
        lines.append(f"MSE={l.get('mse','—')}  Min={l.get('min','—')}  Max={l.get('max','—')}  Moy={l.get('mean','—')}\n")

    if state.m2_log:
        l = state.m2_log
        lines.append("--- Module 2 — Histogramme ---")
        lines.append(f"Opération={l.get('operation','—')}  bins={l.get('bins','—')}")
        lines.append(f"Min={l.get('min','—')}  Max={l.get('max','—')}  Entropie={l.get('entropy','—')} bits\n")

    if state.m3_log:
        l = state.m3_log
        lines.append("--- Module 3 — Fourier ---")
        lines.append(f"Filtre={l.get('filter','—')}  rayon={l.get('radius','—')} px\n")

    if state.m4_log:
        l = state.m4_log
        lines.append("--- Module 4 — Morphologie ---")
        lines.append(f"Opération={l.get('operation','—')}  seuil={l.get('seuil','—')}  itérations={l.get('iterations','—')}")
        lines.append(f"Objets détectés={l.get('count','—')}  Aire moyenne={l.get('mean_area','—')} px²\n")

    best = next((r for r in [state.m4_result, state.m3_result,
                              state.m2_result, state.m1_result] if r is not None),
                state.original_pixels)
    if best is not None:
        s = image_stats(best)
        lines.append("--- Résultat final ---")
        lines.append(f"Entropie={s['entropy']} bits  Contraste={int(best.max())-int(best.min())} niveaux")

    lines.append(f"\nModules complétés : {sum(state.steps_done[:4])}/4")
    return "\n".join(lines)


def _call_openrouter(url, model, api_key, prompt):
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 1500,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST", headers={
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer":  "https://biovision-lab.local",
        "X-Title":       "BioVision Lab",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _call_gemini(url, api_key, prompt):
    payload = json.dumps({
        "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}]}]
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url}?key={api_key}", data=payload, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_anthropic(url, model, api_key, prompt):
    payload = json.dumps({
        "model": model,
        "max_tokens": 1500,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST", headers={
        "Content-Type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["content"][0]["text"]


class AiPanel(tk.Frame):

    def __init__(self, parent, state):
        super().__init__(parent, bg=BG)
        self.state    = state
        self._history = []
        self._loading = False
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._sidebar()
        self._chat_area()

    def _sidebar(self):
        sb = tk.Frame(self, bg=WHITE, width=250)
        sb.grid(row=0, column=0, sticky="ns")
        sb.grid_propagate(False)
        tk.Frame(sb, bg=BORDER, width=1).place(relx=1, rely=0, relheight=1)

        # Title
        tb = tk.Frame(sb, bg=C_AIL, height=56)
        tb.pack(fill="x"); tb.pack_propagate(False)
        tk.Frame(tb, bg=C_AI, width=3).pack(side="left", fill="y")
        col = tk.Frame(tb, bg=C_AIL)
        col.pack(side="left", padx=12, pady=10)
        tk.Label(col, text="Étape 6 — IA & Prédiction",
                 bg=C_AIL, fg=C_AI, font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Label(col, text="Analyse intelligente du pipeline",
                 bg=C_AIL, fg=C_AI, font=("Helvetica", 8)).pack(anchor="w")

        body = tk.Frame(sb, bg=WHITE)
        body.pack(fill="both", expand=True)

        # Provider selector
        self._sep(body)
        self._section(body, "FOURNISSEUR IA")
        self.provider_var = tk.StringVar(value="OpenRouter (gratuit)")
        self.provider_menu = tk.OptionMenu(
            body, self.provider_var, *PROVIDERS.keys(),
            command=self._on_provider_change)
        self.provider_menu.config(
            bg=BG, fg=TEXT, font=("Helvetica", 9),
            relief="flat", highlightthickness=1,
            highlightbackground=BORDER, activebackground=C_AIL)
        self.provider_menu.pack(fill="x", padx=12, pady=(4,0))

        self.hint_lbl = tk.Label(body, text=PROVIDERS["OpenRouter (gratuit)"]["hint"],
                                  bg=WHITE, fg=TEXT3,
                                  font=("Helvetica", 7, "italic"),
                                  wraplength=210, justify="left")
        self.hint_lbl.pack(anchor="w", padx=14, pady=(4,0))

        # API Key
        self._sep(body)
        self._section(body, "CLÉ API")
        self.api_key_var = tk.StringVar()
        tk.Entry(body, textvariable=self.api_key_var,
                 show="•", font=("Helvetica", 9),
                 bg=BG, fg=TEXT, relief="flat",
                 highlightbackground=BORDER,
                 highlightthickness=1).pack(fill="x", padx=12, pady=(0,4))
        tk.Label(body, text="Jamais sauvegardée sur disque",
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
             "En te basant sur les statistiques, quel type d'image biologique "
             "s'agit-il probablement ? (cellules, bactéries, tissu, fluorescence...)"),
            ("Suggérer des améliorations",
             "Analyse les paramètres et suggère des améliorations concrètes "
             "pour optimiser la segmentation cellulaire."),
            ("Rapport médical complet",
             "Génère un rapport médical complet avec introduction, méthodes, "
             "résultats et conclusion."),
            ("Évaluer la qualité",
             "Le bruit a-t-il été bien éliminé ? Le contraste est-il suffisant ? "
             "La segmentation est-elle fiable ?"),
        ]
        for label, prompt in quick_prompts:
            btn = tk.Button(body, text=label, bg=BG, fg=ACCENT,
                            font=("Helvetica", 9), relief="flat",
                            cursor="hand2", anchor="w", padx=8, pady=3,
                            command=lambda p=prompt: self._send(p))
            btn.pack(fill="x", padx=8, pady=1)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=C_AIL))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=BG))

        self._sep(body)
        tk.Button(body, text="Effacer la conversation",
                  bg=BG, fg=TEXT3, font=("Helvetica", 9),
                  relief="flat", cursor="hand2", pady=4,
                  command=self._clear).pack(fill="x", padx=12, pady=(0,8))

        self.status_lbl = tk.Label(body, text="Prêt",
                                    bg=WHITE, fg=TEXT3, font=("Helvetica", 8))
        self.status_lbl.pack(padx=14, anchor="w")

    def _on_provider_change(self, choice):
        self.hint_lbl.config(text=PROVIDERS[choice]["hint"])

    def _chat_area(self):
        main = tk.Frame(self, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=0)

        chat_frame = tk.Frame(main, bg=WHITE)
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10,4))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)

        hdr = tk.Frame(chat_frame, bg=C_AIL, height=34)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="Conversation avec l'IA",
                 bg=C_AIL, fg=C_AI,
                 font=("Helvetica", 10, "bold")).pack(side="left", padx=12, pady=7)
        self.model_lbl = tk.Label(hdr, text="—", bg=C_AIL, fg=C_AI,
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
            (0,0), window=self._msg_frame, anchor="nw")

        self._msg_canvas.bind("<Configure>",
            lambda e: self._msg_canvas.itemconfig(self._msg_win, width=e.width))
        self._msg_frame.bind("<Configure>",
            lambda e: self._msg_canvas.configure(
                scrollregion=self._msg_canvas.bbox("all")))
        self._msg_canvas.bind_all("<MouseWheel>",
            lambda e: self._msg_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        self._welcome()

        input_bar = tk.Frame(main, bg=WHITE,
                              highlightbackground=BORDER, highlightthickness=1)
        input_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0,10))
        input_bar.columnconfigure(0, weight=1)

        self._input = tk.Text(input_bar, height=3, font=("Helvetica", 10),
                               bg=WHITE, fg=TEXT, relief="flat",
                               wrap="word", padx=10, pady=8)
        self._input.grid(row=0, column=0, sticky="ew")
        self._input.bind("<Return>", self._on_enter)
        self._input.bind("<Shift-Return>", lambda e: None)
        self._input.insert("1.0", "Posez votre question sur l'image...")
        self._input.config(fg=TEXT3)
        self._input.bind("<FocusIn>",  self._clear_placeholder)
        self._input.bind("<FocusOut>", self._restore_placeholder)

        sf = tk.Frame(input_bar, bg=WHITE)
        sf.grid(row=0, column=1, padx=8, pady=6)
        self._send_btn = tk.Button(sf, text="Envoyer",
                                    bg=C_AI, fg=WHITE,
                                    font=("Helvetica", 9, "bold"),
                                    relief="flat", cursor="hand2",
                                    padx=12, pady=6,
                                    command=lambda: self._send(self._get_input()))
        self._send_btn.pack()

        tk.Label(input_bar,
                 text="Entrée pour envoyer · Maj+Entrée pour saut de ligne",
                 bg=WHITE, fg=TEXT3, font=("Helvetica", 7)).grid(
                     row=1, column=0, columnspan=2,
                     sticky="w", padx=10, pady=(0,4))

    def _welcome(self):
        self._add_message("assistant",
            "Bonjour ! Je suis votre assistant IA spécialisé en analyse d'image "
            "microscopique.\n\n"
            "Nouveau : choisissez votre fournisseur IA dans la sidebar :\n"
            "• OpenRouter (gratuit, recommandé) — créer un compte sur openrouter.ai\n"
            "• Gemini Flash (Google) — clé sur aistudio.google.com\n"
            "• Claude (Anthropic) — clé sur console.anthropic.com\n\n"
            "Entrez votre clé API puis utilisez un bouton d'analyse rapide.")

    def _add_message(self, role, text):
        is_ai  = (role == "assistant")
        bg_msg = C_AIL  if is_ai else C_USERL
        fg_msg = C_AI   if is_ai else C_USER
        label  = "IA — BioVision" if is_ai else "Vous"

        outer = tk.Frame(self._msg_frame, bg=WHITE)
        outer.pack(fill="x", padx=12, pady=4)
        outer.columnconfigure(0, weight=1)
        tk.Label(outer, text=label, bg=WHITE, fg=fg_msg,
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        bubble = tk.Frame(outer, bg=bg_msg)
        bubble.pack(anchor="w", fill="x")
        tk.Label(bubble, text=text, bg=bg_msg, fg=TEXT,
                 font=("Helvetica", 10), wraplength=800,
                 justify="left", anchor="w", padx=12, pady=8).pack(fill="x")
        self._msg_frame.update_idletasks()
        self._msg_canvas.yview_moveto(1.0)

    def _add_loading(self):
        self._loading_frame = tk.Frame(self._msg_frame, bg=WHITE)
        self._loading_frame.pack(fill="x", padx=12, pady=4)
        tk.Label(self._loading_frame, text="IA — BioVision",
                 bg=WHITE, fg=C_AI, font=("Helvetica", 8, "bold")).pack(anchor="w")
        tk.Label(self._loading_frame, text="Analyse en cours...",
                 bg=C_AIL, fg=C_AI, font=("Helvetica", 10, "italic"),
                 padx=12, pady=8).pack(anchor="w")
        self._msg_canvas.yview_moveto(1.0)

    def _remove_loading(self):
        if hasattr(self, "_loading_frame"):
            self._loading_frame.destroy()

    def _get_input(self):
        text = self._input.get("1.0", "end-1c").strip()
        return "" if text == "Posez votre question sur l'image..." else text

    def _clear_input(self):
        self._input.delete("1.0", "end")
        self._input.config(fg=TEXT)

    def _on_enter(self, event):
        if event.state & 0x1:
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

    def _send(self, text):
        if not text or self._loading:
            return
        api_key = self.api_key_var.get().strip()
        if not api_key:
            self._add_message("assistant",
                               "⚠ Entrez votre clé API dans la sidebar.")
            return
        if not self.state.has_image():
            self._add_message("assistant",
                               "⚠ Chargez une image d'abord.")
            return

        provider = PROVIDERS[self.provider_var.get()]
        self.model_lbl.config(text=provider["model"])

        self._clear_input()
        self._add_message("user", text)
        self._history.append({"role": "user", "content": text})
        self._loading = True
        self._send_btn.config(state="disabled", bg=TEXT3)
        self.status_lbl.config(text="Analyse en cours...", fg=C_AI)
        self._add_loading()

        threading.Thread(target=self._call_api,
                         args=(provider, api_key, text),
                         daemon=True).start()

    def _call_api(self, provider, api_key, user_text):
        context     = _build_context(self.state)
        full_prompt = f"Données pipeline :\n{context}\n\nQuestion : {user_text}"

        try:
            ptype = provider["type"]
            if ptype == "openrouter":
                reply = _call_openrouter(
                    provider["url"], provider["model"], api_key, full_prompt)
            elif ptype == "gemini":
                reply = _call_gemini(provider["url"], api_key, full_prompt)
            elif ptype == "anthropic":
                reply = _call_anthropic(
                    provider["url"], provider["model"], api_key, full_prompt)
            else:
                reply = "Provider inconnu."

            self._history.append({"role": "assistant", "content": reply})
            self.after(0, self._on_reply, reply, None)

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                err = json.loads(body).get("error", {})
                msg = err.get("message", body[:300])
                code = err.get("code", e.code)
            except Exception:
                msg  = body[:300]
                code = e.code

            if code == 429:
                self.after(0, self._on_reply, None,
                           "Quota dépassé (429). Attendre 1 minute et réessayer, "
                           "ou changer de fournisseur (OpenRouter recommandé).")
            else:
                self.after(0, self._on_reply, None,
                           f"Erreur API ({code}) : {msg}")

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

    def _sep(self, p):
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", padx=10, pady=8)

    def _section(self, p, text):
        tk.Label(p, text=text, bg=WHITE, fg=TEXT3,
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=14, pady=(8,3))

    def refresh_image(self):
        if self.state.has_image():
            done = sum(self.state.steps_done[:4])
            self.status_lbl.config(text=f"Image chargée · {done}/4 modules",
                                    fg=C_OK)