# Documentation - Histogrammes et Ajustements d'Image

## Vue d'ensemble

Le module `histogram.py` implémente les **algorithmes d'analyse et d'ajustement d'histogrammes**, outils fondamentaux du traitement d'image. L'histogramme représente la distribution des intensités de pixels. Ces fonctions permettent d'améliorer le contraste, d'analyser les propriétés statistiques et d'optimiser l'utilisation de la plage dynamique.

---

## 1. `to_grayscale(img: np.ndarray) -> np.ndarray`

### Description
Convertit une image **RGB en niveaux de gris** (grayscale). Si l'image est déjà en niveaux de gris, la retourne simplement en uint8.

### Implémentation
```python
def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    return (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
```

### Formule mathématique
La conversion utilise la **formule de luminance ITU-R BT.601**:

$$\text{Gray} = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

Les coefficients reflètent la sensibilité de l'oeil humain aux couleurs:
- **Rouge (0.299)**: Faible sensibilité
- **Vert (0.587)**: Sensibilité maximale
- **Bleu (0.114)**: Très faible sensibilité

### Paramètres
- `img`: Image RGB (H × W × 3) ou déjà en niveaux de gris (H × W)

### Étapes

#### **1. Vérifier les dimensions**
```python
if img.ndim == 2:  # Déjà grayscale
    return img.astype(np.uint8)
```
- `ndim = 2`: Image grayscale 2D
- `ndim = 3`: Image couleur RGB 3D

#### **2. Extraire les canaux**
```
img[:,:,0]  →  Canal Rouge (R)
img[:,:,1]  →  Canal Vert (G)
img[:,:,2]  →  Canal Bleu (B)
```

#### **3. Combinaison pondérée**
```python
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

#### **4. Conversion de type**
```python
astype(np.uint8)  # Restriction à [0, 255]
```

### Exemple numérique
```
Pixel RGB: (255, 128, 64)
         R    G      B
Gray = 0.299×255 + 0.587×128 + 0.114×64
     = 76.245 + 75.136 + 7.296
     = 158.677 ≈ 159
```

### Pourquoi ces coefficients?
La vision humaine:
- Détecte bien les variations en vert (photosensibilité maximale)
- Moins sensible au rouge
- Très peu sensible au bleu

Cette formule reproduit cette sensibilité pour un rendu naturel.

### Variantes
```
# Moyenne simple (moins naturelle):
Gray = (R + G + B) / 3

# Seulement le canal vert:
Gray = G

# Luminance sRGB (autre standard):
Gray = 0.2126*R + 0.7152*G + 0.0722*B
```

### Complexité
- **Temps**: O(h × w) - une opération par pixel
- **Espace**: O(h × w)

---

## 2. `compute_histogram(pixels: np.ndarray, bins: int = 256) -> np.ndarray`

### Description
Calcule l'**histogramme des intensités** - compte combien de pixels ont chaque valeur d'intensité. Permet d'analyser la distribution des niveaux de gris.

### Implémentation
```python
def compute_histogram(pixels: np.ndarray, bins: int = 256) -> np.ndarray:
    hist = np.zeros(bins, dtype=np.int64)
    scale = bins / 256  # Facteur de normalisation
    for v in pixels.ravel():
        b = min(int(v * scale), bins - 1)  # Indice du bin
        hist[b] += 1
    return hist
```

### Paramètres
- `pixels`: Image en niveaux de gris (H × W)
- `bins`: Nombre de bacs d'histogramme (par défaut 256)
  - `bins=256`: 1 bac par niveau d'intensité
  - `bins=64`: 4 niveaux par bac (compression)
  - `bins=8`: 32 niveaux par bac (forte compression)

### Étapes

#### **1. Initialiser l'histogramme**
```python
hist = np.zeros(bins, dtype=np.int64)
# Array avec 'bins' cellules, tout à 0
```

#### **2. Calculer le facteur d'échelle**
```python
scale = bins / 256
# Si bins=256: scale = 1.0 (pas de changement)
# Si bins=128: scale = 0.5 (réduction)
# Si bins=64:  scale = 0.25 (forte réduction)
```

#### **3. Aplatir l'image**
```python
for v in pixels.ravel():  # ravel() = flattening 2D → 1D
```

#### **4. Compter par bac**
```python
b = min(int(v * scale), bins - 1)
# Mapper une valeur [0, 256) en indice [0, bins)
# min() pour éviter l'indice out-of-bounds
hist[b] += 1  # Incrémenter le compteur
```

### Exemple détaillé
Pour image 2×2 avec intensités [200, 100, 150, 250] et bins=4:

```
scale = 4 / 256 = 0.015625

Pixel 200: b = int(200 × 0.015625) = int(3.125) = 3     → hist[3]++
Pixel 100: b = int(100 × 0.015625) = int(1.5625) = 1    → hist[1]++
Pixel 150: b = int(150 × 0.015625) = int(2.34375) = 2   → hist[2]++
Pixel 250: b = int(250 × 0.015625) = int(3.90625) = 3   → hist[3]++

Résultat: hist = [0, 1, 1, 2]
          (0 pixels dans bac 0, 1 dans bac 1, 1 dans bac 2, 2 dans bac 3)
```

### Visualisation
```
Histogramme avec bins=256 (largeur normalisée):
│
│  ╭╮
│  ╭╮╭╮
│╭╮╭╮╭╮╭╮
├┼┼┼┼┼┼┼┼┤
0└────────256┘ Intensité
  Noir      Blanc

Chaque barre = nombre de pixels à cette intensité
```

### Propriétés
- **Somme de l'histogramme = nombre de pixels** = h × w
- **Chaque bin independant**: Pas d'information spatiale
- **Invulnérable à la permutation**: Deux images différentes peuvent avoir même histo

### Cas d'usage
- Détecter des images surexposées/sous-exposées
- Analyser le contraste
- Vérifier la distribution
- Base pour égalisation d'histogramme

### Complexité
- **Temps**: O(h × w)
- **Espace**: O(bins)

---

## 3. `compute_cdf(hist: np.ndarray) -> np.ndarray`

### Description
Calcule la **Fonction de Distribution Cumulative (CDF)** à partir de l'histogramme. La CDF compte le nombre cumulatif de pixels jusqu'à chaque intensité.

### Implémentation
```python
def compute_cdf(hist: np.ndarray) -> np.ndarray:
    cdf = np.zeros_like(hist)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf
```

### Formule mathématique
$$\text{CDF}[i] = \sum_{j=0}^{i} \text{hist}[j]$$

Chaque élément = somme de cet élément et tous les précédents.

### Étapes

#### **1. Initialiser**
```python
cdf = np.zeros_like(hist)  # Même shape et type que hist
```

#### **2. Première valeur**
```python
cdf[0] = hist[0]  # Point de départ
```

#### **3. Boucle cumulative**
```python
for i in range(1, len(hist)):
    cdf[i] = cdf[i-1] + hist[i]
    # Ajouter l'élément current à la somme cumulative
```

### Exemple numérique
```
Histogramme hist:  [5, 3, 2, 4, 1]
                    │  │  │  │  │
CDF[0] = 5          5
CDF[1] = 5 + 3 =    8
CDF[2] = 8 + 2 =    10
CDF[3] = 10 + 4 =   14
CDF[4] = 14 + 1 =   15

Résultat cdf:       [5, 8, 10, 14, 15]
```

### Visualisation
```
Histogramme (discret):    CDF (croissance monotone):
│  ╭╮                      │               ╱╱╱
│  ╭╮  ╭╮                  │             ╱╱
│  ╭╮  ╭╮  ╭╮              │           ╱╱
│  ╭╮  ╭╮  ╭╮ ╭╮           │        ╱╱
├──┴┴──┴┴──┴┴─┴┴─          ├────╱╱╱────
0  1  2  3  4  5        0  1  2  3  4  5

Croissance d'escalier → pente douce (cumulative)
```

### Propriétés mathématiques
- **Toujours croissant**: CDF[i] ≤ CDF[i+1]
- **CDF[0] = hist[0]**: Première valeur
- **CDF[-1] = somme(hist)**: Somme totale (nombre de pixels)
- **Monotone non-décroissante**: Jamais de baisse

### Utilité
```
CDF est la base pour:
1. Équalisaton d'histogramme (normalisation)
2. Spécification d'histogramme
3. Analyse cumulée de la distribution
4. Calcul de centiles
```

### Exemple d'utilisation
```python
hist = compute_histogram(image)
cdf = compute_cdf(hist)

# Trouver le nombre de pixels sous intensité 128:
pixels_under_128 = cdf[128]

# Pourcentage de pixels sombres:
pct_dark = cdf[128] / cdf[255] * 100
```

### Complexité
- **Temps**: O(bins) - une passe unique
- **Espace**: O(bins)

---

## 4. `stretch_contrast(pixels: np.ndarray) -> np.ndarray`

### Description
**Étire le contraste** en mappant les valeurs min-max de l'image sur [0, 255]. Utilise la plage dynamique complète. Utile pour les images sous-exposées ou surexposées.

### Implémentation
```python
def stretch_contrast(pixels: np.ndarray) -> np.ndarray:
    mn, mx = int(pixels.min()), int(pixels.max())
    
    # Cas trivial: image uniforme
    if mx == mn:
        return pixels.copy()
    
    out = np.zeros_like(pixels)
    h, w = pixels.shape
    
    # Transformation linéaire pixel par pixel
    for y in range(h):
        for x in range(w):
            out[y, x] = round((int(pixels[y, x]) - mn) / (mx - mn) * 255)
    
    return out
```

### Formule mathématique
$$I_{\text{out}}[y, x] = \frac{I[y, x] - \min(I)}{(\max(I) - \min(I))} \times 255$$

Transformation linéaire qui remapie [min, max] vers [0, 255].

### Paramètres
- `pixels`: Image en niveaux de gris
- Retourne: Image avec contraste étendu

### Étapes

#### **1. Trouver min et max**
```python
mn = int(pixels.min())  # Valeur minimale dans l'image
mx = int(pixels.max())  # Valeur maximale dans l'image
```

#### **2. Vérifier l'unicité**
```python
if mx == mn:
    # Tous les pixels ont la même valeur
    # Impossible de normaliser (division par 0)
    return pixels.copy()
```

#### **3. Normalisation linéaire**
```python
stretched = (pixel - mn) / (mx - mn) * 255
```

**Breakdown**:
- `(pixel - mn)`: Translate minimum vers 0
- `/ (mx - mn)`: Scale pour que range = 1.0
- `* 255`: Scale pour range [0, 255]

#### **4. Arrondir et itérer**
```python
out[y, x] = round(...)  # Arrondir au pixel entier
```

### Exemple numérique
```
Image originale: min=50, max=200
Range original = 200 - 50 = 150

Pixel valeur 50:   (50 - 50) / 150 * 255 = 0 / 150 * 255 = 0       ✓ min → 0
Pixel valeur 125:  (125 - 50) / 150 * 255 = 75 / 150 * 255 = 127.5 ≈ 128 (milieu)
Pixel valeur 200:  (200 - 50) / 150 * 255 = 150 / 150 * 255 = 255  ✓ max → 255

Résultat: Image utilise [0, 255] au complet!
```

### Visualisation
```
Avant:                        Après stretch_contrast:
Intensité:                    Intensité:
255 │           ╭────╮        255 │      ╭─────────╮
    │           │    │            │      │         │
128 │           │    │         128│──────│─────────│
    │           │    │            │      │         │
  0 │╭─────────╯    │          0 │╭─────│────────╯
              50   200                 0       255

Peu de contraste       →        Contraste maximum
```

### Avantages
✅ Utilise plage dynamique complète
✅ Améliore la visualisation
✅ Simple et rapide
✅ Préserve les relations de luminosité

### Inconvénients
❌ Sensible aux outliers (1 pixel noir suffit)
❌ Amplifie le bruit aux extrêmes
❌ Non adaptatif (global, pas local)

### Cas d'usage
- Images sous-exposées
- Amélioration de contraste simple
- Prétraitement d'image
- Normalisation avant autre traitement

### Complexité
- **Temps**: O(h × w)
- **Espace**: O(h × w)

---

## 5. `equalize_histogram(pixels: np.ndarray) -> np.ndarray`

### Description
**Égalise l'histogramme** pour distribuer les pixels uniformément sur [0, 255]. Améliore le contraste en particulier pour les images mal exposées. Basée sur la CDF.

### Implémentation
```python
def equalize_histogram(pixels: np.ndarray) -> np.ndarray:
    N = pixels.size        # Nombre total de pixels
    L = 256                # Plage (0 à 255)
    
    # Calculer histogramme et CDF
    hist = compute_histogram(pixels, L)
    cdf  = compute_cdf(hist)
    
    # Trouver le premier CDF non-zéro
    cdf_min = int(next(v for v in cdf if v > 0))
    
    # Construire la lookup table (LUT)
    lut = np.zeros(L, dtype=np.uint8)
    for i in range(L):
        lut[i] = round(((int(cdf[i]) - cdf_min) / (N - cdf_min)) * (L - 1))
    
    # Appliquer la LUT
    out = np.zeros_like(pixels)
    h, w = pixels.shape
    for y in range(h):
        for x in range(w):
            out[y, x] = lut[pixels[y, x]]
    
    return out
```

### Formule mathématique
La transformée d'égalisation:

$$I_{\text{out}}[y, x] = \frac{\text{CDF}(I[y, x]) - \text{CDF}_{\min}}{N - \text{CDF}_{\min}} \times (L - 1)$$

où:
- $\text{CDF}(v)$ = nombre cumulatif de pixels ≤ v
- $\text{CDF}_{\min}$ = premier CDF non-zéro
- $N$ = nombre total de pixels
- $L$ = 256 (plage de sortie)

### Étapes détaillées

#### **1. Calculer histogramme et CDF**
```python
hist = compute_histogram(pixels, 256)
cdf = compute_cdf(hist)
```

#### **2. Trouver CDF minimum non-zéro**
```python
cdf_min = int(next(v for v in cdf if v > 0))
```
**Pourquoi?** Éviter division par zéro si certaines intensités ne sont pas présentes.

Exemple:
```
hist = [0, 0, 5, 3, 2, 0, ...]  // Pas de pixels à intensité 0, 1
cdf = [0, 0, 5, 8, 10, 10, ...]
cdf_min = 5  // Première valeur > 0
```

#### **3. Construire lookup table (LUT)**
```python
for i in range(256):
    # Normaliser CDF en [0, 1]
    cdf_norm = (cdf[i] - cdf_min) / (N - cdf_min)
    # Mapper à [0, 255]
    lut[i] = round(cdf_norm * 255)
```

#### **4. Appliquer la LUT**
```python
for y in range(h):
    for x in range(w):
        out[y, x] = lut[pixels[y, x]]  # Recherche directe
```

### Exemple détaillé
```
Image 2×2: [[10, 20],
            [10, 30]]

N = 4
hist = [0, 0, 2, 1, 1, 0, ...]

Après compute_cdf:
cdf = [0, 0, 2, 3, 4, 4, ...]
cdf_min = 2

LUT pour chaque intensité i:
i=10: lut[10] = round((2 - 2) / (4 - 2) * 255) = round(0 / 2 * 255) = 0
i=20: lut[20] = round((3 - 2) / (4 - 2) * 255) = round(1 / 2 * 255) = 128
i=30: lut[30] = round((4 - 2) / (4 - 2) * 255) = round(2 / 2 * 255) = 255

Appliquer LUT:
out[0,0] = lut[10] = 0
out[0,1] = lut[20] = 128
out[1,0] = lut[10] = 0
out[1,1] = lut[30] = 255

Résultat: [[0, 128],
           [0, 255]]

Histogramme d'entrée:  [0, 0, 2, 1, 1, ...]
Histogramme de sortie: [2, 0, 0, 0, 1, 0, ..., 1]  ← plus uniforme!
```

### Visualisation
```
Avant égalisation:        Après égalisation:
│╭╮╭╮                     │  ╱╮ ╱╮
│╭╮╭╮  ╭╮                 │ ╱ ╰╱ ╰╱
│╭╮╭╮  ╭╮  ╭╮             │╱
├┼───────┤                ├──────────
0      255               0        255

Distribution inégale   →  Distribution plus uniforme
Histogramme "plat"
```

### Propriétés mathématiques
- **Monotone croissante**: lut[i] ≤ lut[i+1]
- **lut[0] = 0**: Valeur minimale → 0
- **lut[255] = 255**: Valeur maximale → 255
- **Théorique**: Se rapproche d'une distribution uniforme

### Avantages
✅ Améliore contraste automatiquement
✅ Pas de paramètres à ajuster
✅ Théoriquement justifié
✅ Rapide avec LUT

### Inconvénients
❌ Peut amplifier le bruit
❌ Peut créer des bandes (posterization)
❌ Contraste peut devenir excessif
❌ Non adaptatif

### Variantes
```
Égalisation adaptative (CLAHE):
- Applique localement sur petites régions
- Meilleur résultat mais plus complexe

Spécification d'histogramme:
- Force une distribution cible spécifique
```

### Complexité
- **Temps**: O(h × w + 256)
- **Espace**: O(256)

---

## 6. `threshold_binary(pixels: np.ndarray, seuil: int) -> np.ndarray`

### Description
Applique un **seuillage binaire** - convertit l'image en noir et blanc pur (0 ou 255) selon un seuil. Utile pour la segmentation simple.

### Implémentation
```python
def threshold_binary(pixels: np.ndarray, seuil: int) -> np.ndarray:
    out = np.zeros_like(pixels)
    h, w = pixels.shape
    
    for y in range(h):
        for x in range(w):
            out[y, x] = 255 if pixels[y, x] >= seuil else 0
    
    return out
```

### Formule mathématique
$$I_{\text{out}}[y, x] = \begin{cases}
255 & \text{si } I[y, x] \geq \text{seuil} \\
0 & \text{sinon}
\end{cases}$$

Décision binaire basée sur seuil.

### Paramètres
- `pixels`: Image en niveaux de gris
- `seuil`: Valeur seuil [0, 255]
  - Pixels ≥ seuil → 255 (blanc)
  - Pixels < seuil → 0 (noir)

### Étapes
```python
# Pour chaque pixel:
if pixel >= seuil:
    out_pixel = 255  # Blanc
else:
    out_pixel = 0    # Noir
```

### Exemple
```
Image originale:      Seuil = 150:
[100 150 200]         [  0 255 255]
[80  160 190]   →     [  0 255 255]
[120 140 170]         [  0   0 255]

100 < 150  → 0
150 ≥ 150  → 255
200 ≥ 150  → 255
etc.
```

### Visualisation
```
Niveaux gris:         Seuil binaire:
256 ┤╷╷╷╷╷╷╷╷╷╷╷╷╷  255 ┤       ╭─────┐
    ├╫╫╫╫╫╫╫╫╫       │       │
128 │╱╱╱╱╱╱╱╯╱╱╱  →      │───────┤
    │╱╱╱╱╱╱╱╱╱╱╱      │   │
  0 └╯╯╯╯╯╯╯╯╯╯╯      0 ┤   └─────┘
    0             255    0      seuil     255

Gradient continu      Deux niveaux uniquement
```

### Choix du seuil
```
seuil = 0:     Tout blanc (255)
seuil = 50:    Pixels clairs → blanc
seuil = 128:   Milieu (équilibré)
seuil = 200:   Pixels très clairs uniquement
seuil = 255:   Tout noir (0)
```

### Méthodes de sélection
```
1. Manuel: Choisir empiriquement
2. Otsu: Minimiser intra-classe variance
3. Moyenne: Utiliser la moyenne de l'image
4. Médiane: Utiliser la médiane de l'image
```

### Avantages
✅ Très rapide
✅ Résultat simple et clair
✅ Bon pour segmentation nette
✅ Base pour morphologie mathématique

### Inconvénients
❌ Perte d'information nuancée
❌ Sensible au seuil choisi
❌ Difficile pour images complexes
❌ Crée des artefacts "jagged"

### Cas d'usage
- Séparation fond/objet
- Document scannés
- Images binaires (codes barres, QR)
- Détection de présence/absence
- Pré-traitement pour morphologie

### Complexité
- **Temps**: O(h × w)
- **Espace**: O(h × w)

---

## 7. `image_stats(pixels: np.ndarray) -> dict`

### Description
Calcule **statistiques complètes de l'image**: min, max, moyenne, écart-type, et entropie. Utile pour analyser les propriétés globales.

### Implémentation
```python
def image_stats(pixels: np.ndarray) -> dict:
    flat = pixels.ravel().astype(np.float64)
    
    # Statistiques basiques
    mean = float(np.mean(flat))
    std  = float(np.std(flat))
    mn   = int(pixels.min())
    mx   = int(pixels.max())
    
    # Histogramme et entropie
    hist    = compute_histogram(pixels, 256)
    total   = pixels.size
    entropy = 0.0
    
    for c in hist:
        if c > 0:
            p = c / total  # Probabilité d'une intensité
            entropy -= p * np.log2(p)  # Shannon entropy
    
    return {
        "min": mn, 
        "max": mx, 
        "mean": round(mean, 1),
        "std": round(std, 1), 
        "entropy": round(entropy, 2)
    }
```

### Statistiques expliquées

#### **1. Min et Max**
```python
mn = int(pixels.min())  # Valeur minimale
mx = int(pixels.max())  # Valeur maximale
```

**Interprétation**:
- Min proche de 0: Image foncée
- Max proche de 255: Image claire
- Plage [min, max]: Utilisation de la dynamique

#### **2. Moyenne**
```python
mean = float(np.mean(flat))
```

Formule:
$$\text{mean} = \frac{1}{N} \sum_{i=0}^{N-1} I[i]$$

**Interprétation**:
- Luminosité moyenne
- < 128: Image plutôt foncée
- > 128: Image plutôt claire

#### **3. Écart-type**
```python
std = float(np.std(flat))
```

Formule:
$$\text{std} = \sqrt{\frac{1}{N} \sum_{i=0}^{N-1} (I[i] - \text{mean})^2}$$

**Interprétation**:
- Variation des intensités
- Std ≈ 0: Image uniforme (peu de variation)
- Std élevé: Contraste/variabilité importante

#### **4. Entropie Shannon**
```python
entropy -= p * np.log2(p)  # Pour chaque intensité avec prob p
```

Formule:
$$H = -\sum_{i=0}^{255} p_i \cdot \log_2(p_i)$$

où $p_i$ = probabilité d'intensité i

**Interprétation**:
- Plage: [0, 8] bits
- Low entropy (0-2): Image très simple, peu de variation
- High entropy (6-8): Image riche en détails
- Entropy = 8 si toutes intensités équiprobables

**Exemple**:
```
Image uniforme (tous pixels = 128):
  hist = [0, ..., N, ..., 0]
  p[128] = 1.0
  entropy = -1.0 * log2(1.0) = 0 bits  ← Très prévisible

Image aléatoire (chaque intensité = N/256):
  hist = [N/256, ..., N/256]
  p[i] = 1/256 pour chaque i
  entropy = -256 × (1/256 × log2(1/256)) = 8 bits  ← Imprévisible
```

### Exemple complet
```
Image 3×3 avec valeurs:
[[100, 150, 200],
 [120, 140, 180],
 [110, 130, 190]]

Valeurs: [100, 150, 200, 120, 140, 180, 110, 130, 190]
Mine: 100
Max: 200

Moyenne: (100+150+200+120+140+180+110+130+190) / 9 = 1220 / 9 ≈ 135.6

Écart-type:
  Déviations: [−35.6, 14.4, 64.4, −15.6, 4.4, 44.4, −25.6, −5.6, 54.4]
  Variances: [1267, 207, 4147, 243, 19, 1971, 655, 31, 2959]
  Moyenne var: 11500 / 9 ≈ 1278
  Std: √1278 ≈ 35.8

Entropie: (calcul détaillé basé sur l'histogramme)
```

### Propriétés

#### **Min et Max**
- Toujours dans [0, 255]
- Plage = max - min

#### **Moyenne**
- Valeur attendue de pixels
- Pas d'information spatiale

#### **Écart-type**
- 0 si image uniforme
- Maximal pour contraste fort

#### **Entropie**
- 0 si complètement prévisible
- 8 si perfectly random
- Mesure d'information

### Interprétation globale
```
Image bien exposée:
  - min ≥ 10, max ≤ 245 (utilise dynamique, pas extrêmes)
  - mean proche de 128 (balanced)
  - std élevé (bon contraste)
  - entropy 5-7 (riche en information)

Image sous-exposée:
  - max << 255
  - mean << 128
  - entropy faible (peu de détails)

Image surexposée:
  - min >> 0
  - mean >> 128
  - histogram bloqué à droite
```

### Utilité
- Détailler qualité d'exposition
- Décider si traitement nécessaire
- Comparer avant/après filtrage
- Détecter problèmes d'acquisition

### Complexité
- **Temps**: O(h × w + 256)
- **Espace**: O(h × w)

---

## Résumé des Complexités

| Fonction | Temps | Espace | Utilité |
|----------|-------|--------|---------|
| `to_grayscale` | O(h·w) | O(h·w) | Conversion RGB → Gris |
| `compute_histogram` | O(h·w) | O(bins) | Distribution niveaux |
| `compute_cdf` | O(bins) | O(bins) | Cumul histogramme |
| `stretch_contrast` | O(h·w) | O(h·w) | Contraste linéaire |
| `equalize_histogram` | O(h·w) | O(256) | Contraste automatique |
| `threshold_binary` | O(h·w) | O(h·w) | Segmentation 2 niveaux |
| `image_stats` | O(h·w) | O(256) | Analyse statistique |

---

## Concepts Mathématiques Clés

### Luminance (Grayscale)
```
Gray = 0.299·R + 0.587·G + 0.114·B

Basé sur la sensibilité oculaire:
- Vert: Sensibilité maximale (0.587)
- Rouge: Moyenne (0.299)
- Bleu: Très faible (0.114)
```

### Histogramme
```
Distribution discrète des intensités
Propriétés:
- Non-négatif
- Somme = N (nombre de pixels)
- Pas d'info spatiale
```

### CDF (Fonction de distribution cumulative)
```
CDF[i] = ∑(j=0 à i) hist[j]

Propriétés:
- Monotone non-décroissante
- CDF[0] ≥ 0
- CDF[255] = N
```

### Étirement de contraste
```
Transformation linéaire [min, max] → [0, 255]
Formule: (p - min) / (max - min) × 255

Cas dégénéré: min = max → impossible
```

### Égalisation d'histogramme
```
Objectif: Rendre distribution uniforme
Méthode: Utiliser CDF cumulative
Résultat: Contraste amélioré mais peut être excessif
```

### Seuillage binaire
```
Transformation 1-bit
Décision: pixel ≥ seuil ?

Variantes:
- Seuil fixe (manuel)
- Seuil adaptatif (Otsu)
- Seuil local (différent par région)
```

### Entropie Shannon
```
H = -∑ p_i · log₂(p_i)

Mesure d'incertitude/information
Plage: [0, 8] bits pour image 8-bit
```

---

## Bonnes pratiques

### Ordre de traitement
```
Image brute
    ↓
1. Conversion grayscale (si couleur)
2. Analyser avec image_stats()
3. Choisir traitement:
   - Contraste faible → stretch ou equalize
   - Besoin segmentation → threshold
4. Afficher résultat et comparer
```

### Éviter les pièges
- ❌ Égaliser image très bruitée
- ❌ Seuil trop bas/haut → perte information
- ❌ Absence de validité contrainte grayscale
- ✅ Analyser avant → traiter conscieusement

### Choix de paramètres
```
Stretch_contrast: Pas de paramètre (global)
Equalize_histogram: Pas de paramètre (global)
Threshold: Seuil dépend du cas d'usage
```

---

## Références théoriques

- **Luminance ITU-R BT.601**: Standard de conversion RGB → Grayscale
- **Histogramme**: Analyse statistique fondamentale
- **CDF**: Base pour transformations d'intensité
- **Égalisation d'histogramme**: Adaptive histogram equalization (CLAHE)
- **Entropie Shannon**: Théorie de l'information (mesure d'incertitude)
- **Seuillage Otsu**: Seuil optimal minimisant variance intra-classe

