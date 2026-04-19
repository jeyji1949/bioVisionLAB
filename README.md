# BioVision Lab

> Pipeline d'analyse d'image microscopique   
> Bioinformatique & IA pour la Médecine de Précision  

---

## Description

BioVision Lab est une application desktop **Python/Tkinter** de traitement d'image médicale et microscopique. Elle implémente un pipeline d'analyse biologique en **7 étapes séquentielles**, avec tous les algorithmes codés **manuellement en NumPy** — sans OpenCV, scipy.ndimage, ni aucune bibliothèque de traitement d'image haut niveau.

L'interface est organisée comme un vrai workflow d'analyse cellulaire : chaque étape produit un résultat qui devient l'entrée de l'étape suivante, jusqu'à un rapport final généré par intelligence artificielle et une classification ML sur données réelles.

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
│  Étape 05 — Rapport d'analyse           (M5)        │
│  Résumé des 4 étapes · Statistiques comparées       │
│  Module C (k-means) · Module D (GLCM Haralick)      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Étape 06 — IA & Prédiction             (M6)        │
│  Chat interactif · Identification type d'image      │
│  Rapport médical · Contexte C+D injecté             │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Étape 07 — ML / BloodMNIST             (M7)        │
│  Random Forest from scratch (HOG + k-means NumPy)   │
│  CNN PyTorch · 8 classes de cellules sanguines      │
│  Matrice de confusion · Accuracy par classe         │
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

---

### Module 2 — Histogramme (`core/histogram.py`)

| Opération | Formule | Usage microscopique |
|-----------|---------|---------------------|
| Histogramme | Comptage manuel par boucle, bins 16–256 | Diagnostiquer l'exposition |
| Étirement de contraste | `(I - min) / (max - min) × 255` | Corriger image pâle |
| Égalisation | CDF : `T(r) = round((L-1) × CDF(r) / N)` | Révéler détails cachés (fluorescence) |
| Seuillage binaire | `255 si I ≥ seuil, sinon 0` | Préparer la segmentation |

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

---

### Module C — Classification cellulaire (`core/cell_classifier.py`)

Analyse automatique lancée depuis M5 sur le résultat de M4 :

| Composant | Détail |
|-----------|--------|
| Flood-fill | Détection d'objets 4-connexe from scratch |
| Features | Aire, périmètre, circularité, intensité moyenne, écart-type |
| K-means | Implémenté from scratch, k=3 clusters, 10 restarts |
| PCA 2D | Projection from scratch pour scatter plot |
| Labels | Noyau / Cytoplasme / Débris / Incertain (heuristique circularité) |

---

### Module D — Texture GLCM (`core/glcm.py`)

Analyse de texture de Haralick lancée depuis M5 :

| Composant | Détail |
|-----------|--------|
| Quantisation | 256 → 8 niveaux de gris |
| GLCM | 4 directions (0°, 45°, 90°, 135°), distances [1, 2] |
| Features | Contraste, Énergie, Homogénéité, Entropie, Corrélation, Dissimilarité |
| Profil | Interprétation biologique automatique (fine/modérée/rugueuse) |

---

### Module 5 — Rapport d'analyse (`ui/report_panel.py`)

- Comparaison visuelle originale vs résultat final
- Résumé des 4 étapes avec paramètres et interprétations biologiques
- **Carte Module C** : comptage noyaux / cytoplasme / débris, inertie k-means, variance PCA
- **Carte Module D** : 6 features Haralick, profil texture, interprétation histologique
- Bilan global : entropie, contraste, luminosité, objets détectés

---

### Module 6 — IA & Prédiction (`ui/ai_panel.py`)

Chat interactif avec LLM analysant les résultats du pipeline complet (M1→M4 + C + D) :

**7 analyses rapides disponibles :**
- Analyser le pipeline complet
- Identifier le type d'image biologique
- Interpréter la classification cellulaire (Module C)
- Interpréter la texture GLCM (Module D)
- Générer un rapport médical complet
- Suggérer des améliorations
- Évaluer la qualité du traitement

**Fournisseurs supportés :**

| Fournisseur | Modèle | Coût |
|-------------|--------|------|
| OpenRouter (recommandé) | `arcee-ai/trinity-large-preview:free` | Gratuit |
| Gemini Flash (Google) | `gemini-2.0-flash` | Gratuit (15 req/min) |
| Claude (Anthropic) | `claude-haiku-4-5-20251001` | Payant |

---

### Module 7 — ML / BloodMNIST (`core/m7_model.py` + `ui/m7_panel.py`)

Classification de cellules sanguines sur le dataset **BloodMNIST** (17 092 images 28×28 RGB, 8 classes) :

#### Branche A — Random Forest (from scratch, NumPy uniquement)

| Composant | Détail |
|-----------|--------|
| Features | HOG from scratch (7×7 cellules, 9 orientations) + 10 stats globales = 451 dimensions |
| Arbre | CART with Gini impurity, from scratch |
| Forêt | Bootstrap sampling + random feature subsets (√d) |
| Résultats | Accuracy globale, accuracy par classe, matrice de confusion 8×8 |

#### Branche B — CNN (PyTorch)

| Composant | Détail |
|-----------|--------|
| Architecture | Conv(3→16) + MaxPool + Conv(16→32) + MaxPool + Dense(1568→128) + Dense(128→8) |
| Entraînement | Adam, CrossEntropyLoss, CPU, ~170k paramètres |
| Résultats | Courbes loss/accuracy par epoch, accuracy par classe, matrice de confusion |

**8 classes BloodMNIST :** Basophile · Éosinophile · Érythroblaste · Granulocyte imm. · Lymphocyte · Monocyte · Neutrophile · Plaquette

---

## Architecture

```
biovision_lab/
│
├── main.py
├── requirements.txt
│
├── app/
│   ├── state.py            # AppState : pixels, logs M1-M4, mc/md/m7 results, steps_done[7]
│   └── main_window.py      # Fenêtre principale + barre pipeline 7 étapes
│
├── core/                   # Algorithmes purs — zéro tkinter, zéro OpenCV
│   ├── histogram.py
│   ├── filters.py
│   ├── fft_manual.py
│   ├── morpho.py
│   ├── cell_classifier.py  # Module C — k-means from scratch
│   ├── glcm.py             # Module D — GLCM Haralick from scratch
│   └── m7_model.py         # Module M7 — Random Forest + CNN PyTorch
│
├── ui/
│   ├── filter_panel.py     # M1
│   ├── hist_panel.py       # M2
│   ├── fourier_panel.py    # M3
│   ├── morpho_panel.py     # M4
│   ├── report_panel.py     # M5 — rapport + cartes C et D
│   ├── ai_panel.py         # M6 — chat IA multi-fournisseurs
│   └── m7_panel.py         # M7 — interface RF + CNN BloodMNIST
│
└── assets/
    ├── guide_utilisation.docx
    └── samples/
```

---

## Installation

```bash
# Dépendances de base
pip install numpy pillow matplotlib

# Module M7 (optionnel)
pip install medmnist torch
```

### Dépendances

| Package | Usage | Requis |
|---------|-------|--------|
| `numpy` | Base de tous les algorithmes | Oui |
| `pillow` | Chargement et affichage images | Oui |
| `matplotlib` | Histogramme M2 | Oui |
| `tkinter` | Interface graphique | stdlib |
| `urllib` | Appels API IA (M6) | stdlib |
| `torch` | CNN M7 | M7 uniquement |
| `medmnist` | Dataset BloodMNIST | M7 uniquement |

---

## Utilisation rapide

1. `python main.py`
2. **"Charger un échantillon"** → PNG/JPG/TIFF
3. **Étape 01** : choisir bruit → filtre recommandé → Appliquer
4. **Étape 02** : choisir opération → Calculer
5. **Étape 03** : choisir filtre → Calculer FFT
6. **Étape 04** : opération morpho → Appliquer → comptage cellulaire
7. **Étape 05** : "Actualiser" → rapport + analyse C (k-means) + D (GLCM)
8. **Étape 06** : clé OpenRouter → analyses rapides IA
9. **Étape 07** : "Charger BloodMNIST" → entraîner RF et/ou CNN

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
Circularity = 4π·Area / Perimeter²                 # circularité (Module C)
GLCM(i,j) = P(i→j) normalisé sur 4 directions      # texture Haralick
Gini(t) = 1 - Σ p_k²                               # impureté (Random Forest)
HOG = histogramme orientations pondéré magnitude    # features M7
```

---

## Auteur

**Jihane El Khraibi**  