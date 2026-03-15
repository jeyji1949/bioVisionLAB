# BioVision Lab

> Pipeline d'analyse d'image microscopique
> Bioinformatique & IA pour la Médecine de Précision  


---

## Description

BioVision Lab est une application desktop **Python/Tkinter** de traitement d'image médicale et microscopique. Elle implémente un pipeline d'analyse biologique en **4 modules séquentiels**, avec tous les algorithmes codés **manuellement en NumPy** — sans OpenCV, scipy.ndimage, ni aucune bibliothèque de traitement d'image haut niveau.

L'interface est organisée comme un vrai workflow d'analyse cellulaire : chaque étape produit un résultat qui devient l'entrée de l'étape suivante.

---

## Pipeline

```
Image microscopique
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Étape 01 — Prétraitement & Débruitage  (M1)        │
│  Bruit S&P / Gaussien → Filtre médian / gaussien    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Étape 02 — Histogramme & Contraste     (M2)        │
│  Étirement · Égalisation · Seuillage                │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Étape 03 — Transformée de Fourier      (M3)        │
│  FFT Cooley-Tukey 2D · Passe-bas/haut/bande         │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  Étape 04 — Morphologie & Segmentation  (M4)        │
│  Érosion · Dilatation · Comptage cellulaire         │
└─────────────────────────────────────────────────────┘
```

---

## Fonctionnalités

### Module 1 — Prétraitement

| Opération | Description |
|-----------|-------------|
| Bruit Poivre & Sel | Pixels aléatoires blancs/noirs (boucle manuelle) |
| Bruit Gaussien | Box-Muller transform, pixel par pixel |
| Filtre Médian | Tri de fenêtre glissante, sans bibliothèque |
| Filtre Gaussien | Noyau analytique + convolution 2D manuelle |
| Filtre Moyenneur | Convolution avec noyau uniforme |
| Filtre Laplacien | Noyau `[[0,-1,0],[-1,4,-1],[0,-1,0]]` |

**Recommandation automatique** : sélection de Poivre & Sel → filtre médian pré-sélectionné. Bruit Gaussien → filtre gaussien.  
**Affichage** : 3 panneaux côte à côte — image originale | image bruitée | image filtrée.

---

### Module 2 — Histogramme

| Opération | Formule |
|-----------|---------|
| Histogramme | Comptage manuel par boucle, bins ajustables (16–256) |
| Étirement de contraste | `out = (I - min) / (max - min) × 255` |
| Égalisation | Méthode CDF : `T(r) = round((L-1) × CDF(r) / N)` |
| Seuillage binaire | `out = 255 si I ≥ seuil, sinon 0` |

**Statistiques calculées** : min, max, moyenne, écart-type, entropie de Shannon (bits).  
**Visualisation** : histogramme matplotlib intégré en bas de l'interface.

---

### Module 3 — Transformée de Fourier

Implémentation **100% from scratch** de l'algorithme FFT Cooley-Tukey radix-2 :

```
fft1d()   — Bit-reversal permutation + butterfly O(N log N)
fft2d()   — Décomposition ligne/colonne
ifft2d()  — FFT inverse pour reconstruction
fft_shift() — Centrage des basses fréquences
```

| Filtre | Application biologique |
|--------|------------------------|
| Spectre de magnitude | Visualiser les fréquences dominantes (log scale) |
| Passe-bas (rayon R) | Éliminer le bruit haute fréquence, adoucir l'image |
| Passe-haut (rayon R) | Renforcer les contours cellulaires, détecter les membranes |
| Passe-bande | Isoler une plage de textures — séparer noyau du cytoplasme |

---

### Module 4 — Morphologie

| Opération | Usage biologique |
|-----------|-----------------|
| Érosion | Séparer les cellules jointives |
| Dilatation | Fermer les lacunes membranaires |
| Opening | Supprimer le bruit binaire sans déformer les cellules |
| Closing | Reconstruire les noyaux fragmentés |
| Top Hat | Extraire les granules cytoplasmiques |
| Black Hat | Détecter les vacuoles et inclusions |
| Gradient | Tracer les membranes cellulaires |

**Comptage automatique** : flood fill 4-connexe → nombre d'objets + aire moyenne en px².

---

## Architecture

```
biovision_lab/
│
├── main.py                     # Point d'entrée
├── requirements.txt
│
├── app/
│   ├── state.py                # AppState partagé entre modules
│   └── main_window.py          # Fenêtre principale + barre pipeline
│
├── core/                       # Algorithmes purs — zéro tkinter
│   ├── histogram.py            # to_grayscale, compute_histogram, stretch,
│   │                           # equalize, threshold, image_stats
│   ├── filters.py              # convolve2d, mean, gaussian, median,
│   │                           # laplacian, add_salt_pepper, add_gaussian_noise
│   ├── fft_manual.py           # fft1d (Cooley-Tukey), fft2d, ifft2d,
│   │                           # fft_shift, apply_frequency_mask, spectrum_image
│   └── morpho.py               # erode, dilate, opening, closing,
│                               # top_hat, black_hat, morpho_gradient, run_morpho
│
├── ui/                         # Panels Tkinter — appellent core/
│   ├── filter_panel.py         # M1 : 3 panneaux + sidebar contrôles
│   ├── hist_panel.py           # M2 : 2 images + histogramme matplotlib
│   ├── fourier_panel.py        # M3 : source | spectre | IFFT
│   └── morpho_panel.py         # M4 : 2 panneaux + rapport cellulaire
│
└── assets/
    ├── guide_utilisation.docx  # Guide d'utilisation complet
    └── samples/                # Images de test
```

### Règle de dépendance

```
main.py → app/main_window.py → ui/*_panel.py → core/*.py
```

`core/` ne contient **jamais** `import tkinter`. Les fonctions prennent un `np.ndarray` et retournent un `np.ndarray`.

---

## Installation

```bash
# Prérequis : Python 3.8+
pip install numpy pillow matplotlib

# Lancer l'application
python main.py
```

### Dépendances

| Package | Usage | Version |
|---------|-------|---------|
| `numpy` | Tableaux — base de tous les algorithmes | ≥ 1.20 |
| `pillow` | Chargement et affichage des images | ≥ 9.0 |
| `matplotlib` | Histogramme dans le Module 2 | ≥ 3.5 |
| `tkinter` | Interface graphique | stdlib |

---

## Utilisation rapide

1. Lancer `python main.py`
2. Cliquer **"Charger un échantillon"** — sélectionner une image PNG/JPG/TIFF
3. Naviguer entre les 4 étapes via la **barre pipeline** en haut
4. Dans chaque module, configurer les paramètres dans la sidebar gauche
5. Cliquer **"Appliquer"** / **"Calculer"** pour exécuter
6. Cliquer **"Envoyer vers →"** pour passer l'image au module suivant
7. En M4, cliquer **"Générer rapport complet"** pour le bilan d'analyse

### Types d'images recommandées

- Coupes histologiques (HES, Giemsa, Gram)
- Microscopie optique (champ clair, contraste de phase)
- Images de fluorescence en niveaux de gris (DAPI, GFP)
- Images bactériologiques (cultures, frottis sanguin)
- Résolution conseillée : **256×256 à 1024×1024 px**

> ⚠️ Les images très grandes (> 2000 px) ralentissent significativement les algorithmes manuels. Préférer un recadrage avant traitement.

---

## Formules implémentées

```python
# Niveaux de gris
Y = 0.299·R + 0.587·G + 0.114·B

# Étirement de contraste
out(x,y) = (I(x,y) - Imin) / (Imax - Imin) × 255

# Égalisation (CDF)
T(r) = round((L-1) × CDF(r) / N)

# Noyau gaussien
G(x,y) = exp(-(x² + y²) / 2σ²) / (2πσ²)

# Laplacien
K = [[0,-1,0], [-1,4,-1], [0,-1,0]]

# FFT Cooley-Tukey (butterfly)
X[k] = Σ x[n] · e^(-j2πkn/N)   avec bit-reversal + O(N log N)

# Top Hat
TH(I) = I - Opening(I)

# Entropie Shannon
H = -Σ p(i) · log₂(p(i))   [bits]
```

---

## Auteur

**Jihane El Khraibi**  
Master BIAM — Bioinformatique et Intelligence Artificielle pour la Médecine de Précision  
Faculté des Sciences Dhar El Mahraz, Fès, Maroc