# BioVision Lab

> Pipeline d'analyse d'image microscopique   
> Bioinformatique & IA pour la Médecine de Précision  

---

## Description

BioVision Lab est une application desktop **Python/Tkinter** de traitement d'image médicale et microscopique. Elle implémente un pipeline d'analyse biologique en **6 étapes séquentielles**, avec tous les algorithmes codés **manuellement en NumPy** — sans OpenCV, scipy.ndimage, ni aucune bibliothèque de traitement d'image haut niveau.

L'interface est organisée comme un vrai workflow d'analyse cellulaire : chaque étape produit un résultat qui devient l'entrée de l'étape suivante, jusqu'à un rapport final généré par intelligence artificielle.

---

## Pipeline complet

```
Image microscopique
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Étape 01 — Prétraitement & Débruitage  (M1)        │
│  Bruit S&P / Gaussien → Filtre médian / gaussien    │
│  Recommandation automatique filtre selon bruit      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Étape 02 — Histogramme & Contraste     (M2)        │
│  Étirement · Égalisation · Seuillage                │
│  Statistiques + visualisation matplotlib            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Étape 03 — Transformée de Fourier      (M3)        │
│  FFT Cooley-Tukey 2D from scratch                   │
│  Passe-bas · Passe-haut · Passe-bande               │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Étape 04 — Morphologie & Segmentation  (M4)        │
│  Érosion · Dilatation · Opening · Closing           │
│  Top Hat · Black Hat · Gradient · Comptage          │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Étape 05 — Rapport d'analyse                       │
│  Résumé des 4 étapes · Statistiques comparées       │
│  Comparaison avant/après · Bilan global             │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Étape 06 — IA & Prédiction                         │
│  Chat interactif · Identification type d'image      │
│  Rapport médical · Suggestions d'amélioration       │
└─────────────────────────────────────────────────────┘
```

---

## Fonctionnalités

### Module 1 — Prétraitement (`core/filters.py`)

| Opération | Description | Usage microscopique |
|-----------|-------------|---------------------|
| Bruit Poivre & Sel | Pixels aléatoires 0/255 (boucle manuelle) | Simulation capteur CCD défectueux |
| Bruit Gaussien | Box-Muller transform, pixel par pixel | Simulation bruit thermique |
| Filtre Médian | Tri de fenêtre glissante, sans bibliothèque | Éliminer S&P sans flouter les contours |
| Filtre Gaussien | Noyau analytique + convolution 2D manuelle | Lisser le bruit continu (fluorescence) |
| Filtre Moyenneur | Convolution avec noyau uniforme | Lissage rapide, prévisualisation |
| Filtre Laplacien | Noyau `[[0,-1,0],[-1,4,-1],[0,-1,0]]` | Détecter membranes et filaments |

**Recommandation automatique** : Poivre & Sel → filtre médian pré-sélectionné (badge vert). Bruit Gaussien → filtre gaussien.  
**Affichage** : 3 panneaux — image originale | image bruitée | image filtrée.

---

### Module 2 — Histogramme (`core/histogram.py`)

| Opération | Formule | Usage microscopique |
|-----------|---------|---------------------|
| Histogramme | Comptage manuel par boucle, bins 16–256 | Diagnostiquer l'exposition |
| Étirement de contraste | `(I - min) / (max - min) × 255` | Corriger image pâle |
| Égalisation | CDF : `T(r) = round((L-1) × CDF(r) / N)` | Révéler détails cachés (fluorescence) |
| Seuillage binaire | `255 si I ≥ seuil, sinon 0` | Préparer la segmentation |

**Statistiques** : min, max, moyenne, écart-type, entropie de Shannon (bits).  
**Interprétation biologique** affichée automatiquement pour chaque opération.

---

### Module 3 — Transformée de Fourier (`core/fft_manual.py`)

Implémentation **100% from scratch** — aucune utilisation de `numpy.fft` :

```
fft1d()               — Bit-reversal + butterfly O(N log N)
fft2d()               — Décomposition ligne/colonne
ifft2d()              — FFT inverse pour reconstruction
fft_shift()           — Centrage des basses fréquences
apply_frequency_mask()— Masques circulaires passe-bas/haut/bande
spectrum_image()      — Visualisation spectre (log scale)
```

| Filtre | Rayon conseillé | Usage microscopique |
|--------|-----------------|---------------------|
| Spectre | — | Diagnostiquer artéfacts périodiques |
| Passe-bas | R = 20–40 | Lisser bruit haute fréquence |
| Passe-haut | R = 20–40 | Renforcer contours, détecter membranes |
| Passe-bande | R = 15–30 | Isoler granules, séparer noyau/cytoplasme |

---

### Module 4 — Morphologie (`core/morpho.py`)

| Opération | Formule | Usage microscopique |
|-----------|---------|---------------------|
| Érosion | `min` voisinage 3×3 | Séparer cellules jointives |
| Dilatation | `max` voisinage 3×3 | Fermer lacunes membranaires |
| Opening | `Dilate(Erode(A))` | Supprimer bruit binaire avant comptage |
| Closing | `Erode(Dilate(A))` | Reconstruire noyaux fragmentés |
| Top Hat | `A - Opening(A)` | Extraire granules cytoplasmiques |
| Black Hat | `Closing(A) - A` | Détecter vacuoles et inclusions |
| Gradient | `Dilate(A) - Erode(A)` | Tracer membranes cellulaires |

**Comptage** : flood fill 4-connexe → nombre d'objets + aire moyenne en px².  
**Rapport** : bouton "Générer rapport complet" → bilan de l'analyse.

---

### Module 5 — Rapport d'analyse (`ui/report_panel.py`)

Panel de synthèse automatique à la fin du pipeline :

- **Comparaison visuelle** : image originale vs résultat final côte à côte
- **Résumé des 4 étapes** : opération + paramètres + interprétation biologique par module
- **Statistiques comparées** : tableau min/max/moyenne/écart-type/entropie à chaque étape
- **Histogrammes comparés** : original vs résultat final
- **Bilan global** : 4 métriques clés + interprétation textuelle automatique avec recommandations

Se rafraîchit automatiquement à chaque navigation vers l'étape 05.

---

### Module 6 — IA & Prédiction (`ui/ai_panel.py`)

Chat interactif avec un LLM analysant les résultats du pipeline :

**L'IA reçoit automatiquement :** toutes les statistiques, les paramètres utilisés, les logs de chaque module et les résultats de comptage cellulaire.

**5 analyses rapides disponibles :**
- Analyser le pipeline complet
- Identifier le type d'image biologique
- Suggérer des améliorations
- Générer un rapport médical complet
- Évaluer la qualité du traitement

**Fournisseurs supportés :**

| Fournisseur | Modèle par défaut | Coût |
|-------------|-------------------|------|
| OpenRouter (recommandé) | `arcee-ai/trinity-large-preview:free` | Gratuit |
| Gemini Flash (Google) | `gemini-2.0-flash` | Gratuit (15 req/min) |
| Claude (Anthropic) | `claude-haiku-4-5-20251001` | Payant |

**Modèles gratuits OpenRouter** (une seule clé `sk-or-v1-...` pour tous) :

```
arcee-ai/trinity-large-preview:free      ← recommandé, fonctionne bien
meta-llama/llama-3.2-3b-instruct:free
google/gemma-3-4b-it:free
google/gemma-3-12b-it:free
mistralai/mistral-small-3.1-24b-instruct:free
qwen/qwen-2.5-72b-instruct:free
deepseek/deepseek-r1:free
```

---

## Architecture

```
biovision_lab/
│
├── main.py                     # Point d'entrée
├── requirements.txt
│
├── app/
│   ├── state.py                # AppState : pixels, logs M1-M4, steps_done[6]
│   └── main_window.py          # Fenêtre principale + barre pipeline 6 étapes
│
├── core/                       # Algorithmes purs — zéro tkinter, zéro OpenCV
│   ├── histogram.py
│   ├── filters.py
│   ├── fft_manual.py
│   └── morpho.py
│
├── ui/                         # Panels Tkinter
│   ├── filter_panel.py         # M1
│   ├── hist_panel.py           # M2
│   ├── fourier_panel.py        # M3
│   ├── morpho_panel.py         # M4
│   ├── report_panel.py         # M5 — rapport synthèse
│   └── ai_panel.py             # M6 — chat IA multi-fournisseurs
│
└── assets/
    ├── guide_utilisation.docx
    ├── README.docx
    └── samples/
```

### Règle de dépendance

```
main.py → app/main_window.py → ui/*_panel.py → core/*.py
```

`core/` ne contient **jamais** `import tkinter`. Le module IA utilise uniquement `urllib` (stdlib).

### AppState — pipeline de données

```python
AppState
├── original_pixels              # Image source
├── m1_result / m2_result / m3_result / m4_result
├── m1_log / m2_log / m3_log / m4_log   # Paramètres de chaque module
└── steps_done[6]                # Flags de complétion

pipeline_input(step)  # remonte automatiquement la meilleure entrée disponible
```

---

## Installation

```bash
# Prérequis : Python 3.8+
pip install numpy pillow matplotlib

# Lancer
python main.py
```

### Dépendances

| Package | Usage | Version |
|---------|-------|---------|
| `numpy` | Base de tous les algorithmes | ≥ 1.20 |
| `pillow` | Chargement et affichage images | ≥ 9.0 |
| `matplotlib` | Histogramme M2 | ≥ 3.5 |
| `tkinter` | Interface graphique | stdlib |
| `urllib` | Appels API IA (M6) | stdlib |

---

## Utilisation rapide

1. `python main.py`
2. **"Charger un échantillon"** → PNG/JPG/TIFF
3. **Étape 01** : choisir bruit → filtre recommandé s'active → Appliquer
4. **Étape 02** : choisir opération → Calculer
5. **Étape 03** : choisir filtre → Calculer FFT
6. **Étape 04** : choisir opération morpho → Appliquer → voir comptage
7. **Étape 05** : cliquer "Actualiser" → rapport synthèse
8. **Étape 06** : entrer clé OpenRouter → analyses rapides IA

### Obtenir une clé API gratuite (module IA)

1. [openrouter.ai](https://openrouter.ai) → créer un compte
2. **Keys** → **Create Key**
3. Clé `sk-or-v1-...` → coller dans la sidebar Étape 06
4. Une seule clé = accès à tous les modèles gratuits

### Types d'images recommandées

| Type | Format | Résolution |
|------|--------|------------|
| Coupes histologiques (HES, Giemsa) | PNG, TIFF | 512×512 à 1024×1024 |
| Microscopie optique | PNG, JPG | 256×256 à 512×512 |
| Fluorescence (DAPI, GFP) | TIFF, PNG | 512×512 |
| Images bactériologiques | PNG, JPG | 256×256 à 512×512 |

> ⚠️ Images > 2000 px : recadrer avant traitement (FFT très lente).

---

## Formules implémentées

```python
Y = 0.299·R + 0.587·G + 0.114·B                    # niveaux de gris
out = (I - Imin) / (Imax - Imin) × 255             # étirement
T(r) = round((L-1) × CDF(r) / N)                   # égalisation
G(x,y) = exp(-(x²+y²) / 2σ²) / (2πσ²)             # noyau gaussien
K = [[0,-1,0], [-1,4,-1], [0,-1,0]]                # laplacien
z = √(-2·ln(u1)) × cos(2π·u2)                      # Box-Muller
X[k] = Σ x[n]·e^(-j2πkn/N)  O(N log N)            # FFT Cooley-Tukey
Erode  = min { A(x+i, y+j) | (i,j) ∈ B }          # érosion
Dilate = max { A(x+i, y+j) | (i,j) ∈ B }          # dilatation
TH(I) = I - Opening(I)                              # top hat
H = -Σ p(i)·log₂(p(i))                             # entropie Shannon
```

---

## Documents

| Fichier | Contenu |
|---------|---------|
| `assets/guide_utilisation.docx` | Guide complet avec scénarios biologiques |
| `assets/README.docx` | Documentation Word |
| `explication_algorithmes.pdf` | Explication ligne par ligne de chaque algorithme |

---

## Auteur

**Jihane El Khraibi**  
