# Documentation - Implémentation Manuelle des Filtres d'Image

## Vue d'ensemble

Le module `filters.py` implémente les **algorithmes de filtrage spatial** couramment utilisés en traitement d'image. Ces fonctions opèrent directement sur les pixels de l'image (domaine spatial) contrairement aux opérations fréquentielles. Elles permettent le lissage, la détection de contours, le débruitage et l'ajout de bruit.

---

## 1. `_pad(img: np.ndarray, pad: int) -> np.ndarray`

### Description
Fonction utilitaire qui **ajoute une bordure de padding** autour de l'image. Essential pour appliquer des filtres près des bords sans perdre d'informations.

### Implémentation
```python
def _pad(img: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(img, ((pad, pad), (pad, pad)), mode="edge")
```

### Paramètres
- `img`: Image originale (matrice 2D)
- `pad`: Nombre de pixels à ajouter de chaque côté

### Modes de padding
- `mode="edge"`: Utilise les **pixels de bordure** pour remplir (réplication du bord)

### Exemple visuel
```
Image originale (3×3):      Après padding=1:
┌─────┐                     ┌─────────┐
│A B C│                     │A A B C C│
│D E F│        →            │A A B C C│
│G H I│                     │D D E F F│
└─────┘                     │G G H I I│
                            │G G H I I│
                            └─────────┘
                            (5×5)
```

### Utilité
- Permet d'appliquer un noyau (kernel) sur **tous les pixels**, y compris les bords
- Évite la perte d'informations aux frontières de l'image
- Utilise les pixels existants pour extrapoler (mode "edge")

### Complexité
- **Temps**: O(pad × (h + w))
- **Espace**: O((h + 2×pad) × (w + 2×pad))

---

## 2. `_gaussian_kernel(ksize: int, sigma: float) -> np.ndarray`

### Description
Génère un **noyau gaussien normalisé** qui est la base du filtrage gaussien. La fonction gaussienne 2D est très utilisée pour le lissage d'image.

### Implémentation
```python
def _gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    center = ksize // 2
    k = np.zeros((ksize, ksize), dtype=np.float64)
    
    # Évaluer la fonction gaussienne en chaque point
    for y in range(ksize):
        for x in range(ksize):
            dx, dy = x - center, y - center
            k[y, x] = np.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
    
    # Normaliser pour que la somme = 1
    return k / k.sum()
```

### Formule mathématique
La **fonction gaussienne 2D** (fonction de densité de probabilité normale):

$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

Dans ce code (sans le terme de normalisation constant, mais avec normalisation finale):

$$K[y, x] = e^{-\frac{(x - center)^2 + (y - center)^2}{2\sigma^2}}$$

### Paramètres
- `ksize`: Taille du noyau (ksize × ksize). Doit être impair (3, 5, 7, ...)
- `sigma`: Écart-type de la gaussienne
  - **Sigma petit** (0.5): Fonction très piquée, lissage léger
  - **Sigma grand** (2.0): Fonction aplatie, lissage agressif

### Étapes de calcul

#### **1. Créer la grille**
```
Pour ksize=5, sigma=1.0:
Center = 5 // 2 = 2

Calculer δx, δy depuis le centre pour chaque position:
  (-2,-2) (-1,-2) (0,-2) (1,-2) (2,-2)
  (-2,-1) (-1,-1) (0,-1) (1,-1) (2,-1)
  (-2, 0) (-1, 0) (0, 0) (1, 0) (2, 0)    ← (0,0) est le centre
  (-2, 1) (-1, 1) (0, 1) (1, 1) (2, 1)
  (-2, 2) (-1, 2) (0, 2) (1, 2) (2, 2)
```

#### **2. Évaluer la gaussienne**
```python
distance² = (x - center)² + (y - center)²
valeur = exp(- distance² / (2 × sigma²))
```

#### **3. Normaliser**
```python
somme = somme de tous les éléments du noyau
noyau = noyau / somme   # Propriété: somme finale = 1.0
```

### Exemple numérique
Pour `ksize=3, sigma=1.0`:

```
Gaussienne brute:
0.0432  0.0821  0.0432
0.0821  0.3679  0.0821
0.0432  0.0821  0.0432

Somme = 1.000

Après normalisation (déjà 1.0 en ce cas)
```

### Propriétés mathématiques
- **Somme des éléments = 1**: Préserve la luminosité moyenne
- **Symétrique**: k[y][x] = k[x][y]
- **Séparable**: Peut être décomposé en produit externe de deux vecteurs 1D

### Visualisation 3D
```
       ↑ Valeur
       │
    0.4├─────────────────
       │        ╱╲
       │       ╱  ╲
       │      ╱    ╲
       │     ╱      ╲
    0.0├────────────────→ Distance du centre
        Pour sigma=1.0, max ≈ 0.1592
```

### Complexité
- **Temps**: O(ksize²)
- **Espace**: O(ksize²)

---

## 3. `convolve2d(pixels: np.ndarray, kernel: np.ndarray) -> np.ndarray`

### Description
Applique la **convolution 2D** - opération fondamentale du traitement d'image. Glisse un noyau (kernel) sur l'image en calculant la somme des produits pixel-par-pixel.

### Implémentation
```python
def convolve2d(pixels: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w   = pixels.shape
    kh, kw = kernel.shape
    pad    = kh // 2  # Padding basé sur la taille du kernel
    padded = _pad(pixels, pad)
    out    = np.zeros((h, w), dtype=np.float64)
    
    for y in range(h):
        for x in range(w):
            # Extraire la région et multiplier élément par élément
            out[y, x] = np.sum(padded[y:y+kh, x:x+kw] * kernel)
    
    return np.clip(out, 0, 255).astype(np.uint8)
```

### Formule mathématique
$$I_{out}[y, x] = \sum_{dy=-k/2}^{k/2} \sum_{dx=-k/2}^{k/2} I_{pad}[y+dy, x+dx] \times K[dy+k/2, dx+k/2]$$

### Étapes d'exécution

#### **1. Padding**
L'image est paddée avec `pad = ksize // 2` pour préserver les dimensions

#### **2. Fenêtrage (Windowing)**
Pour chaque pixel (y, x):
- Extraire une fenêtre carrée de taille kh × kw centrée en (y, x)
- Fenêtre: `padded[y:y+kh, x:x+kw]`

#### **3. Opération élémentaire**
```
Région de l'image:    Noyau:          Produit:
[a b c]               [k1 k2 k3]      [a·k1  b·k2  c·k3]
[d e f]      ×        [k4 k5 k6]  =   [d·k4  e·k5  f·k6]
[g h i]               [k7 k8 k9]      [g·k7  h·k8  i·k9]

Résultat: somme(produit) = a·k1 + b·k2 + ... + i·k9
```

#### **4. Clipping et conversion**
```python
np.clip(valeur, 0, 255)  # Restriction à [0, 255]
astype(np.uint8)         # Conversion en entier 8-bit
```

### Exemple visuel complet
```
Image:                Kernel Median 3×3:    Résultat au pixel (1,1):
[10 20 30]            [1/9 1/9 1/9]        somme = 10·(1/9) + 20·(1/9) + ...
[40 50 60]       ×    [1/9 1/9 1/9]    =               + 50·(1/9) + ... + 90·(1/9)
[70 80 90]            [1/9 1/9 1/9]                  = 360/9 = 40
```

### Applications
- Avec noyau de 1: Filtre moyenne (lissage)
- Avec noyau gaussien: Lissage gaussien
- Avec noyau [−1 2 −1]: Détection de contours
- Avec noyau [0 −1 0; −1 4 −1; 0 −1 0]: Laplacien

### Complexité
- **Temps**: O(h × w × kh × kw)
- **Espace**: O(h × w)

---

## 4. `mean_filter(pixels: np.ndarray, ksize: int = 3) -> np.ndarray`

### Description
Applique un **filtre moyenne** (box filter). Remplace chaque pixel par la **moyenne des pixels voisins**. Effet: lissage de l'image, réduction du bruit.

### Implémentation
```python
def mean_filter(pixels: np.ndarray, ksize: int = 3) -> np.ndarray:
    # Créer un noyau constant
    k = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    return convolve2d(pixels, k)
```

### Noyau utilisé
Pour `ksize=3`:
```
Noyau:
[1/9  1/9  1/9]
[1/9  1/9  1/9]
[1/9  1/9  1/9]

Tous les coefficients égaux, divisés par le nombre d'éléments
```

### Formule
$$I_{out}[y, x] = \frac{1}{ksize^2} \sum_{dy=-k/2}^{k/2} \sum_{dx=-k/2}^{k/2} I[y+dy, x+dx]$$

### Avantages
✅ Simple et rapide
✅ Bon pour débruitage général
✅ Préserve assez bien les contours grossiers

### Inconvénients
❌ Crée un flou uniforme
❌ Mauvais pour les contours fins
❌ Crée des artefacts "blocs" près des bords

### Exemple d'effet
```
Image originale:      Après mean_filter (ksize=3):
[1 2 1]               [2 2 2]
[2 2 2]        →      [2 2 2]
[1 2 1]               [2 2 2]

Tous les pixels prennent la valeur moyenne = 2
```

### Complexité
- **Temps**: O(h × w × ksize²)
- **Espace**: O(h × w)

---

## 5. `gaussian_filter(pixels: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray`

### Description
Applique un **filtre gaussien** - le filtre de lissage le plus utilisé en pratique. Donne plus de poids aux pixels proches qu'aux éloignés, produisant un lissage naturel et progressif.

### Implémentation
```python
def gaussian_filter(pixels: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    k = _gaussian_kernel(ksize, sigma)  # Générer le noyau
    return convolve2d(pixels, k)
```

### Noyau gaussien (exemple ksize=5, sigma=1.0)
```
       0.010  0.015  0.016  0.015  0.010
       0.015  0.034  0.038  0.034  0.015
0.016  0.038  0.044  0.038  0.016
       0.015  0.034  0.038  0.034  0.015
       0.010  0.015  0.016  0.015  0.010

Les poids décroissent vers les bords (effet de cloche)
```

### Paramètres clés

#### **sigma (écart-type)**
- **Petit sigma (0.5)**: Lissage léger, détails préservés
- **Grand sigma (2.0)**: Lissage agressif, plus floue
- Relation: Plus sigma est grand, plus la gaussienne est aplatie

#### **ksize (taille du kernel)**
- Doit être **impair** (3, 5, 7, 9, ...)
- Plus grand = plus de pixels considérés
- Rarement au-delà de 9-11 (bonne approximation de la gaussienne)

### Formule
$$G[y, x] = e^{-\frac{(y-center)^2 + (x-center)^2}{2\sigma^2}}$$

### Avantages sur mean_filter
✅ Lissage plus naturel (mathématiquement justifié)
✅ Préserve mieux les contours
✅ Propriété séparable (peut être optimisé)
✅ Base théorique (théorie des probabilités)

### Visualisation de l'effet
```
σ = 0.5 (fort lissage)     σ = 1.0 (normal)        σ = 2.0 (léger lissage)
[···················]      [··········]            [······]
[··0.044··········]        [··0.044···]            [0.038]
[···············]          [········]              [····]
  Très piqué              Équilibré                 Très aplati
```

### Complexité
- **Temps**: O(h × w × ksize²)
- **Espace**: O(h × w)

---

## 6. `median_filter(pixels: np.ndarray, ksize: int = 3) -> np.ndarray`

### Description
Le **filtre médian** remplace chaque pixel par la **médiane des pixels voisins**. Excellent pour éliminer le bruit "poivre & sel" tout en préservant les contours.

### Implémentation
```python
def median_filter(pixels: np.ndarray, ksize: int = 3) -> np.ndarray:
    h, w   = pixels.shape
    pad    = ksize // 2
    padded = _pad(pixels, pad)
    out    = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            # Extraire la fenêtre et aplatir
            window = padded[y:y+ksize, x:x+ksize].ravel()
            # Trouver la médiane
            out[y, x] = np.sort(window)[len(window) // 2]
    
    return out
```

### Formule
```
Médiane(window) = Element du milieu de la suite triée
```

### Étapes

#### **1. Extraire la fenêtre**
```python
window = padded[y:y+ksize, x:x+ksize].ravel()
# ravel() aplatit la matrice ksize×ksize en vecteur 1D
```

Exemple pour ksize=3:
```
Fenêtre 2D:     Après ravel():
[10 20 30]  →   [10, 20, 30, 40, 50, 60, 70, 80, 90]
[40 50 60]
[70 80 90]
```

#### **2. Trier**
```python
sorted_window = np.sort(window)
# [10, 20, 30, 40, 50, 60, 70, 80, 90]
```

#### **3. Prendre le milieu**
```python
median = sorted_window[len(window) // 2]
# Pour 9 éléments: index = 9 // 2 = 4
# Élément à index 4 = 50 (le 5e élément) ✓
```

### Exemple avec bruit
```
Fenêtre avec bruit poivre & sel:
[  0  50  100]  (0 = "poivre"/noir)
[ 50 100   0 ]
[100 100  50]

Valeurs triées: [0, 0, 50, 50, 50, 50, 100, 100, 100]
Médiane (index 4) = 50  ← Élimine complètement le bruit!
```

### Propriétés mathématiques
- **Préservation des contours**: Les contours verticaux/horizontaux restent nets
- **Robustesse**: Moins sensible aux valeurs aberrantes (contrairement à la moyenne)
- **Non-linéaire**: Ne peut pas être exprimé comme convolution linéaire

### Avantages
✅ Excellent pour bruit poivre & sel
✅ Préserve les contours nets
✅ Pas d'artefacts "fantômes"
✅ Robuste aux outliers

### Inconvénients
❌ Plus coûteux (tri à chaque pixel)
❌ Moins bon pour bruit gaussien
❌ Peut créer des zones "plates" (quantification)

### Complexité
- **Temps**: O(h × w × ksize² × log(ksize²)) pour trier
- **Espace**: O(h × w + ksize²)

---

## 7. `laplacian_filter(pixels: np.ndarray) -> np.ndarray`

### Description
Le **filtre laplacien** détecte les **variations d'intensité rapides** (contours). C'est un opérateur dérivé du second ordre qui met en évidence les transitions nettes.

### Implémentation
```python
def laplacian_filter(pixels: np.ndarray) -> np.ndarray:
    # Noyau laplacien standard
    k = np.array([[0, -1,  0],
                  [-1,  4, -1],
                  [0, -1,  0]], dtype=np.float64)
    
    h, w   = pixels.shape
    padded = _pad(pixels, 1)  # Padding de 1 (kernel est 3×3)
    out    = np.zeros((h, w), dtype=np.float64)
    
    for y in range(h):
        for x in range(w):
            # Appliquer le flux du kernel
            out[y, x] = abs(np.sum(padded[y:y+3, x:x+3] * k))
    
    return np.clip(out, 0, 255).astype(np.uint8)
```

### Noyau laplacien
```
[0, -1, 0]             Représentation:
[-1, 4, -1]                  0  -1  0
[0, -1, 0]                  -1   4 -1
                              0  -1  0

Somme des coefficients = 0 (filtre passe-haut)
```

### Formule mathématique
Le **Laplacien discret** (dérivée du second ordre):

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

Approximation discrète:
$$L[y, x] = 4 \cdot I[y, x] - I[y+1, x] - I[y-1, x] - I[y, x+1] - I[y, x-1]$$

### Calcul pas à pas

Pour cette région d'image:
```
[a  b  c]
[d  e  f]
[g  h  i]

Opération:        4·e  - b - d - f - h
              = 4·e - (b + d + f + h)
              = e - (b + d + f + h - 3·e)
              = différence du centre avec la moyenne voisine
```

### Interpétation
- **Valeur forte**: Transition rapide (contour ou bord)
- **Valeur faible**: Zone uniforme
- **Utilité**: Détection de contours, estimation de la netteté

### Exemple avec contour
```
Zone uniforme:          Contour net:
[100 100 100]           [100 100 100]
[100 100 100]    →      [100   0   0]
[100 100 100]           [ 0    0   0]

Laplacien:              Laplacien:
[  0   0   0]           [ 0  400 400]
[  0   0   0]    →      [400 1200 400]
[  0   0   0]           [400 400   0]
```

### Étapes

#### **1. Calcul de la réponse**
```python
réponse = 4*e - (b + d + f + h)
        = 4*e - somme(voisins)
```

#### **2. Prise de la valeur absolue**
```python
abs(réponse)  # Les contours peuvent avoir des signes différents
```

#### **3. Clipping**
```python
np.clip(valeur, 0, 255)  # Restriction à la plage valide
```

### Propriétés
- **Zéro en régions uniformes**: Pas de réponse dans les zones plates
- **Maximal aux contours**: Pics aux transitions
- **Sensible au bruit**: Amplifie le bruit haute fréquence

### Variantes
Il existe plusieurs noyaux laplaciens:

Laplacien standard (4-connexe):
```
[ 0 -1  0]
[-1  4 -1]
[ 0 -1  0]
```

Laplacien 8-connexe (include diagonales):
```
[-1 -1 -1]
[-1  8 -1]
[-1 -1 -1]
```

### Utilisation
- Détection de contours
- Estimation de la netteté
- Segmentation d'image
- Souvent combiné avec seuillage

### Complexité
- **Temps**: O(h × w)
- **Espace**: O(h × w)

---

## 8. `add_salt_pepper(pixels: np.ndarray, amount: float = 0.05) -> np.ndarray`

### Description
Ajoute du **bruit "poivre & sel"** (salt-and-pepper noise) à l'image. Chaque pixel touché devient soit blanc (255, "sel") soit noir (0, "poivre") aléatoirement.

### Implémentation
```python
def add_salt_pepper(pixels: np.ndarray, amount: float = 0.05) -> np.ndarray:
    out = pixels.copy()
    n   = int(pixels.size * amount)  # Nombre de pixels à corrompre
    
    for _ in range(n):
        # Choisir un pixel aléatoire
        y = np.random.randint(0, pixels.shape[0])
        x = np.random.randint(0, pixels.shape[1])
        
        # Changer en noir (0) ou blanc (255) avec prob 50%
        out[y, x] = 255 if np.random.random() < 0.5 else 0
    
    return out
```

### Paramètres
- `amount`: Proportion de pixels à corrompre
  - `0.05` = 5% des pixels
  - `0.1` = 10% des pixels
  - `0.5` = 50% des pixels

### Étapes

#### **1. Calculer le nombre de pixels**
```python
n = int(pixels.size * amount)
# Pour image 100×100 et amount=0.05:
# n = int(10000 × 0.05) = 500 pixels
```

#### **2. Boucle de corruption**
```python
for _ in range(n):
    # Coordonnée aléatoire
    y = random(0, hauteur)
    x = random(0, largeur)
    
    # 50/50: noir ou blanc
    couleur = 255 si random() < 0.5 sinon 0
    out[y, x] = couleur
```

### Visualisation
```
Image originale:    Après add_salt_pepper(0.05):
[100 150 200]       [100 255   0]   (pixels blancs/noirs aléatoires)
[120 160 180]  →    [120 160 255]
[110 170 190]       [  0 170 190]

Caractéristique: pixels extrêmes (0 ou 255) isolés
```

### Comparaison avec autres bruits
- **Poivre & sel**: Valeurs extrêmes (0 ou 255)
- **Gaussien**: Valeurs aléatoires normalement distribuées
- **Uniforme**: Valeurs aléatoires uniformément distribuées

### Utilisation
- Test de robustesse des filtres
- Évaluation de la qualité des méthodes de débruitage
- Simulation de dégradation image

### Complexité
- **Temps**: O(n) où n = nombre de pixels corrompus
- **Espace**: O(h × w)

---

## 9. `add_gaussian_noise(pixels: np.ndarray, sigma: float = 25.0) -> np.ndarray`

### Description
Ajoute du **bruit gaussien** (Gaussian noise) - le type de bruit le plus courant en imagerie réelle. Chaque pixel est perturbé par une variable aléatoire suivant une distribution normale.

### Implémentation
```python
def add_gaussian_noise(pixels: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    out = np.zeros_like(pixels, dtype=np.float64)
    h, w = pixels.shape
    
    for y in range(h):
        for x in range(w):
            # Générer une variable normale standard avec Box-Muller
            u1 = max(np.random.random(), 1e-10)  # Évite log(0)
            u2 = np.random.random()
            z  = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            
            # Ajouter au pixel
            out[y, x] = pixels[y, x] + z * sigma
    
    return np.clip(out, 0, 255).astype(np.uint8)
```

### Paramètres
- `sigma`: Écart-type du bruit
  - `5.0`: Bruit léger
  - `25.0`: Bruit modéré (par défaut)
  - `50.0`: Bruit fort

### Formule mathématique
$$I_{bruit}[y, x] = I[y, x] + Z \cdot \sigma$$

où $Z \sim \mathcal{N}(0, 1)$ (normale standard)

### Algorithme: Transformation de Box-Muller

Génère deux variables normales indépendantes à partir de deux variables uniformes:

#### **1. Générer deux uniformes**
```python
u1 = random(0, 1)  # Première uniforme
u2 = random(0, 1)  # Deuxième uniforme
```

#### **2. Transformer avec Box-Muller**
```python
z = sqrt(-2 × log(u1)) × cos(2π × u2)
```

C'est la **formule de Box-Muller**, qui produit $z \sim \mathcal{N}(0, 1)$

#### **3. Mettre à l'échelle**
```python
bruit = z × sigma
```

#### **4. Ajouter au pixel**
```python
pixel_bruiteux = pixel_original + bruit
```

### Avantage du clipping à 1e-10
```python
u1 = max(np.random.random(), 1e-10)
```
- Évite `log(0)` qui est indéfini
- u1 = 0 est probabilité nulle (continue)
- `1e-10` est un très petit nombre acceptable

### Visualisation mathématique
La distribution du bruit:
```
Densité de probabilité gaussienne avec σ = 25:

                ╱╲
              ╱    ╲
            ╱        ╲
          ╱            ╲
        ╱                ╲
      ──┴────────────────┴──→ Valeur du bruit
    -100               +100
         68% entre -25 et +25
         95% entre -50 et +50
```

### Exemple numérique
```
Pixel original: 100
σ = 25

z₁ = 0.5  → bruit₁ = 0.5 × 25 = 12.5    → 100 + 12.5 = 112.5
z₂ = -1.2 → bruit₂ = -1.2 × 25 = -30   → 100 - 30 = 70
z₃ = 0.1  → bruit₃ = 0.1 × 25 = 2.5    → 100 + 2.5 = 102.5

Après clipping à [0, 255] et conversion uint8
```

### Propriétés statistiques
- **Moyenne des perturbations**: 0 (pas de biais)
- **Écart-type des perturbations**: sigma
- **Support**: Théoriquement (-∞, +∞), mais 99.7% des valeurs dans [μ - 3σ, μ + 3σ]

### Comparaison avec poivre & sel
```
                Gaussien          Poivre & Sel
Valeurs       Continues         Discrètes (0, 255)
Distribution  Normale           Uniforme aux extrêmes
Réaliste      Très (capteurs)   Moins (défaut capteur)
Filtrage      Gaussien/Médian   Médian
```

### Utilisation
- Simulation de bruit capteur réel
- Test de robustesse de filtres
- Évaluation de débruiteurs
- Augmentation de données (data augmentation)

### Complexité
- **Temps**: O(h × w) - une évaluation Box-Muller par pixel
- **Espace**: O(h × w)

---

## 10. `NOISE_FILTER_MAP` - Mapping Bruit-Filtre

### Description
Dictionnaire qui **recommande les filtres appropriés** selon type de bruit présent dans l'image.

### Implémentation
```python
NOISE_FILTER_MAP = {
    "Poivre & Sel": {
        "recommended": "Médian",
        "others": ["Moyenneur", "Gaussien"]
    },
    "Gaussien": {
        "recommended": "Gaussien",
        "others": ["Moyenneur", "Médian"]
    },
    "Aucun bruit": {
        "recommended": None,
        "others": ["Médian", "Gaussien", "Moyenneur", "Laplacien"]
    },
}
```

### Structure
- **Clé**: Type de bruit détecté
- **Valeur**: Dictionnaire avec:
  - `"recommended"`: Filtre recommandé (None si aucun)
  - `"others"`: Alternatives possibles

### Rationale scientifique

#### **Poivre & Sel → Médian**
✅ **Optimal**: Filtre médian spécialisé pour ce type de bruit
✅ Élimine les pixels isolés extrêmes
✅ Préserve les contours

Alternatives:
- Moyenneur: Moins efficace mais acceptable
- Gaussien: Moins bon, floutera les contours

#### **Gaussien → Gaussien**
✅ **Optimal**: Filtre gaussien mathématiquement justifié
✅ Théorie du signal: Gaussien optimal pour bruit gaussien (Wiener filter)
✅ Lissage progressif

Alternatives:
- Moyenneur: Acceptable mais moins naturel
- Médian: Moins approprié pour ce bruit

#### **Aucun bruit → None**
✅ **Recommandation**: Ne pas filtrer (risque d'artefacts)
⚠️ Options si nécessaire:
- Médian: Faible risque de dégradation
- Gaussien: Peut légèrement lisser
- Moyenneur: Peut créer flou blocks

### Utilisation en code
```python
bruit_type = "Gaussien"
filtre_rec = NOISE_FILTER_MAP[bruit_type]["recommended"]
# filtre_rec = "Gaussien"

autres = NOISE_FILTER_MAP[bruit_type]["others"]
# autres = ["Moyenneur", "Médian"]
```

### Arbre de décision
```
Détecter type de bruit
         ↓
┌─────────────────────────────┐
│                             │
Poivre & Sel           Gaussien           Autre
│                             │             │
Médian             Gaussien        Essayer plusieurs
│                             │             │
(Alternatives:)      (Alternatives:)    (Comparer résultats)
 - Moyenneur         - Moyenneur
 - Gaussien          - Médian
```

### Complexité
- **Temps**: O(1) - recherche dans dictionnaire
- **Espace**: O(1) - taille fixe

---

## Résumé des Complexités

| Fonction | Temps | Espace | Utilité |
|----------|-------|--------|---------|
| `_pad` | O(pad(h+w)) | O((h+p)(w+p)) | Bordure pour noyau |
| `_gaussian_kernel` | O(ksize²) | O(ksize²) | Générer noyau |
| `convolve2d` | O(h·w·kh·kw) | O(h·w) | Base linéaire |
| `mean_filter` | O(h·w·ksize²) | O(h·w) | Lissage rapide |
| `gaussian_filter` | O(h·w·ksize²) | O(h·w) | Lissage naturel |
| `median_filter` | O(h·w·ksize²·log k²) | O(h·w) | Débruitage points |
| `laplacian_filter` | O(h·w) | O(h·w) | Contours nets |
| `add_salt_pepper` | O(n) | O(h·w) | Bruit extrêmes |
| `add_gaussian_noise` | O(h·w) | O(h·w) | Bruit réaliste |

---

## Algorithmes Clés: Résumé Théorique

### Convolution 2D
```
L'opération fondamentale:
Output = ∑∑ Image(n,m) × Kernel(n,m)

Propriétés:
- Linéaire: conv(a·I₁ + b·I₂, K) = a·conv(I₁,K) + b·conv(I₂,K)
- Commutative: I * K = K * I
- Associative: (I * K₁) * K₂ = I * (K₁ * K₂)
```

### Noyaux communs
```
Moyenne (3×3):          Sobel-X (détecte contours verticaux):
[1 1 1] / 9             [-1 0 1]
[1 1 1]                 [-2 0 2]
[1 1 1]                 [-1 0 1]

Identité (pas de changement):
[0 0 0]
[0 1 0]
[0 0 0]
```

### Padding modes
- **edge**: Réplique les pixels de bordure
- **zero**: Ajoute des zéros
- **reflect**: Reflet du contenu
- **wrap**: Enroule (périodique)

---

## Bonnes pratiques d'utilisation

### Ordre de traitement
```
Image bruitée
    ↓
1. Ajuster sigma (bruit)
2. Choisir ksize impair
3. Appliquer filtre approprié
4. Évaluer résultat (sharpness, artefacts)
5. Optionnel: itérer ou combiner filtres
```

### Éviter les artefacts courants
- ❌ **Blou excessif**: Trop grand sigma
- ❌ **Artefacts "blocks"**: Valeurs trop petites ou mal normalisées
- ❌ **Halos autour contours**: Filtre média mal appliqué
- ✅ **Solution**: Paramètres adaptés + évaluation visuelle

---

## Références théoriques

- **Convolution 2D**: Fondement du traitement d'image spatial
- **Fonction Gaussienne**: Modèle probabiliste optimal (Wiener filter)
- **Filtre Médian**: Filtrage morphologique (ordre statistique)
- **Laplacien**: Opérateur dérivé pour détection
- **Box-Muller**: Génération efficace de normales

