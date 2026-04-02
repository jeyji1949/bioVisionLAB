# Documentation - Implémentation Manuelle des Algorithmes FFT

## Vue d'ensemble

Le module `fft_manual.py` implémente les algorithmes **FFT (Fast Fourier Transform)** de manière manuelle, sans utiliser les bibliothèques optimisées comme `numpy.fft`. Cette implémentation est pédagogique et permet de comprendre les mécanismes internes de la transformée de Fourier rapide.

---

## 1. `_next_pow2(n: int) -> int`

### Description
Trouve la prochaine puissance de 2 supérieure ou égale à `n`. C'est une fonction utilitaire cruciale car les algorithmes FFT fonctionnent mieux avec des tailles qui sont des puissances de 2.

### Implémentation
```python
def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1  # Décalage binaire à gauche: p = p * 2
    return p
```

### Algorithme
1. Commence avec `p = 1` (2⁰)
2. Double `p` répétitivement (`p <<= 1` équivaut à `p *= 2`)
3. S'arrête quand `p ≥ n`
4. Retourne la puissance de 2 trouvée

### Exemple
- `_next_pow2(100)` → `128` (2⁷)
- `_next_pow2(64)` → `64` (2⁶)
- `_next_pow2(1)` → `1` (2⁰)

### Complexité
- **Temps**: O(log n) - nombre de doublement = log₂(n)
- **Espace**: O(1)

---

## 2. `fft1d(re: np.ndarray, im: np.ndarray, invert: bool = False`

### Description
La **Transformée de Fourier Rapide 1D** est l'algorithme clé. Elle décompose une séquence en ses composantes fréquentielles. Utilise l'algorithme **Cooley-Tukey** avec décimation par fréquence.

### Paramètres
- `re`: Partie réelle des nombres complexes (modifiée in-place)
- `im`: Partie imaginaire des nombres complexes (modifiée in-place)
- `invert`: Si `True`, effectue la IFFT (Transformée Inverse)

### Structure de l'algorithme

#### **Étape 1: Bit-Reversal Permutation**
```python
j = 0
for i in range(1, n):
    bit = n >> 1
    while j & bit:
        j ^= bit
        bit >>= 1
    j ^= bit
    if i < j:
        re[i], re[j] = re[j], re[i]
        im[i], im[j] = im[j], im[i]
```

**Objectif**: Réorganiser les éléments dans un ordre basé sur l'inversion des bits de leurs indices.

**Exemple** (pour n=8):
- Indice 1 (binaire: 001) → 4 (binaire: 100)
- Indice 2 (binaire: 010) → 2 (binaire: 010) - pas de changement
- Indice 3 (binaire: 011) → 6 (binaire: 110)

**Fonctionnement**:
- Pour chaque index `i`, calcule `j` = index avec bits inversés
- Échange les éléments à positions `i` et `j` si `i < j`
- Utilise des opérations binaires: `>>` (décalage), `&` (ET), `^` (XOR)

#### **Étape 2: Papillon FFT (Butterfly Operations)**
```python
length = 2
while length <= n:
    sign = -1.0 if not invert else 1.0
    ang = sign * 2.0 * np.pi / length
    wr, wi = np.cos(ang), np.sin(ang)  # Racine primitive de l'unité
    
    for i in range(0, n, length):
        cur_r, cur_i = 1.0, 0.0
        for k in range(length // 2):
            # Éléments du premier groupe
            u_r = re[i + k]
            u_i = im[i + k]
            
            # Éléments du second groupe avec rotation
            v_r = re[i + k + length//2] * cur_r - im[i + k + length//2] * cur_i
            v_i = re[i + k + length//2] * cur_i + im[i + k + length//2] * cur_r
            
            # Opération papillon
            re[i + k]             = u_r + v_r
            im[i + k]             = u_i + v_i
            re[i + k + length//2] = u_r - v_r
            im[i + k + length//2] = u_i - v_i
            
            # Mise à jour du facteur de rotation
            new_r = cur_r * wr - cur_i * wi
            cur_i = cur_r * wi + cur_i * wr
            cur_r = new_r
    
    length <<= 1  # length *= 2
```

**Concept clé**: L'opération "papillon" combine deux nombres complexes selon la formule:
```
X_even = x + w * x_odd
X_odd  = x - w * x_odd
```

où `w` est une racine primitive de l'unité (nombre complexe avec magnitude 1).

**Déroulement**:
1. Commence avec des groupes de 2 éléments
2. Puis groupe de 4, 8, 16, ... jusqu'à n
3. À chaque étape, double la taille des groupes
4. Pour chaque groupe, applique l'opération papillon

#### **Étape 3: Normalisation (pour IFFT)**
```python
if invert:
    re /= n
    im /= n
```
Division par `n` pour la transformée inverse (propriété mathématique de la FFT).

### Complexité
- **Temps**: O(n log n) - bien meilleur que DFT naïve en O(n²)
- **Espace**: O(1) - modifie les arrays in-place

### Exemple graphique (n=4)
```
Entrée: [a, b, c, d]

Après bit-reversal: [a, c, b, d]

Étape 1 (length=2):
  [a+c, a-c, b+d, b-d]

Étape 2 (length=4):
  Applique papillons finaux avec rotations appropriées
  → Résultat FFT final
```

---

## 3. `fft2d(pixels: np.ndarray)`

### Description
Applique la **FFT 2D** à une image. Effectue d'abord la FFT sur les lignes, puis sur les colonnes (approche séparable).

### Algorithme
```python
def fft2d(pixels: np.ndarray):
    h, w = pixels.shape
    W = _next_pow2(w)      # Arrondir la largeur à puissance de 2
    H = _next_pow2(h)      # Arrondir la hauteur à puissance de 2
    
    # Créer arrays complexes padés avec zéros
    re = np.zeros((H, W), dtype=np.float64)
    im = np.zeros((H, W), dtype=np.float64)
    re[:h, :w] = pixels.astype(np.float64)
    
    # Appliquer FFT 1D sur chaque ligne
    for y in range(H):
        row_r = re[y].copy()
        row_i = im[y].copy()
        fft1d(row_r, row_i, invert=False)
        re[y] = row_r
        im[y] = row_i
    
    # Appliquer FFT 1D sur chaque colonne
    for x in range(W):
        col_r = re[:, x].copy()
        col_i = im[:, x].copy()
        fft1d(col_r, col_i, invert=False)
        re[:, x] = col_r
        im[:, x] = col_i
    
    return re, im, W, H
```

### Étapes
1. **Padding**: Les dimensions sont arrondies à la puissance de 2 la plus proche
2. **Remplissage**: L'image originale est placée en haut-gauche, le reste rempli de zéros
3. **FFT par rangées**: Chaque ligne subit la FFT 1D
4. **FFT par colonnes**: Chaque colonne subit la FFT 1D
5. **Résultats**: Retourne les parties réelle et imaginaire, ainsi que les dimensions

### Retour
- `re`: Partie réelle du spectre (H × W)
- `im`: Partie imaginaire du spectre (H × W)
- `W, H`: Dimensions paddées

### Complexité
- **Temps**: O(W·H·log(W)·log(H))
- **Espace**: O(W·H)

### Visualisation
```
Image originale (h×w)          Après padding (H×W)
┌─────────┐                    ┌──────────────┐
│ ░░░░░░░ │                    │ ░░░░░░░░░░░░ │
│ ░░░░░░░ │    →               │ ░░░░░░░░░░░░ │
│ ░░░░░░░ │                    │ ░░░░░░░░░░░░ │
└─────────┘                    │ ░░░░░░░░░░░░ │
                               └──────────────┘
                               (remplie de zéros)
```

---

## 4. `ifft2d(re: np.ndarray, im: np.ndarray, W: int, H: int) -> None`

### Description
La **Transformée de Fourier Inverse 2D**. Reconvertit le domaine fréquentiel en domaine spatial.

### Algorithme
```python
def ifft2d(re: np.ndarray, im: np.ndarray, W: int, H: int) -> None:
    # Appliquer IFFT 1D sur chaque ligne
    for y in range(H):
        row_r = re[y].copy()
        row_i = im[y].copy()
        fft1d(row_r, row_i, invert=True)  # ← invert=True
        re[y] = row_r
        im[y] = row_i
    
    # Appliquer IFFT 1D sur chaque colonne
    for x in range(W):
        col_r = re[:, x].copy()
        col_i = im[:, x].copy()
        fft1d(col_r, col_i, invert=True)  # ← invert=True
        re[:, x] = col_r
        im[:, x] = col_i
```

### Différence avec fft2d
- Appelle `fft1d` avec `invert=True`
- Ce qui active la normalisation par `n` (division par n)
- **Modifie les arrays in-place**

### Propriété mathématique
```
FFT(IFFT(X)) = X  (Propriété de réciprocité)
```

---

## 5. `fft_shift(re: np.ndarray, im: np.ndarray, W: int, H: int)`

### Description
Effectue le **décalage FFT** (FFT shift). Réorganise le spectre de Fourier pour placer les basses fréquences au centre et les hautes fréquences aux bords.

### Algorithme
```python
def fft_shift(re: np.ndarray, im: np.ndarray, W: int, H: int):
    hw, hh = W // 2, H // 2
    shift_r = np.zeros_like(re)
    shift_i = np.zeros_like(im)
    
    for y in range(H):
        for x in range(W):
            nx = (x + hw) % W  # Décalage circulaire
            ny = (y + hh) % H
            shift_r[ny, nx] = re[y, x]
            shift_i[ny, nx] = im[y, x]
    
    return shift_r, shift_i
```

### Transformation
- Le point `(0, 0)` (composante DC = moyenne) se déplace au centre
- Les hautes fréquences sont déplacées des coins vers le centre
- Utilise l'**arithmétique modulaire** pour le décalage circulaire

### Visualisation
```
Avant FFT shift (coin en haut-gauche):
┌─────────┐
│ DC  ...│
│ ...  ...│
└─────────┘

Après FFT shift (centre):
┌─────────┐
│ ...  ...│
│ ... DC ..│
└─────────┘
```

### Complexité
- **Temps**: O(W·H)
- **Espace**: O(W·H)

---

## 6. `apply_frequency_mask(re, im, W, H, filter_type, radius)`

### Description
Applique un **masque fréquentiel** pour filtrer certaines composantes. Utilisé pour les filtres passe-bas, passe-haut et passe-bande.

### Algorithme
```python
def apply_frequency_mask(re, im, W, H, filter_type, radius):
    cx, cy = W // 2, H // 2  # Centre du spectre
    
    for y in range(H):
        for x in range(W):
            # Calcule distance du centre (dans l'espace fréquentiel)
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Détermine le masque selon le type de filtre
            if filter_type == "passe-bas":
                mask = 1.0 if dist <= radius else 0.0
            elif filter_type == "passe-haut":
                mask = 0.0 if dist <= radius else 1.0
            else:  # "passe-bande"
                mask = 1.0 if radius <= dist <= radius * 1.6 else 0.0
            
            # Applique le masque (multiplication complexe)
            re[y, x] *= mask
            im[y, x] *= mask
```

### Types de filtres

#### **Passe-bas** (Low Pass)
```
mask = 1 si distance ≤ radius, sinon 0
Effet: Réduit les hautes fréquences (lisse l'image)
```

#### **Passe-haut** (High Pass)
```
mask = 0 si distance ≤ radius, sinon 1
Effet: Réduit les basses fréquences (met en évidence les détails)
```

#### **Passe-bande** (Band Pass)
```
mask = 1 si radius ≤ distance ≤ radius × 1.6, sinon 0
Effet: Garde une bande de fréquences particulière
```

### Visualisation spatiale
```
Passe-bas:              Passe-haut:            Passe-bande:
┌─────────┐            ┌─────────┐            ┌─────────┐
│█████████│            │░░░░░░░░░│            │░░█████░░│
│█████████│            │░░█████░░│            │░░█████░░│
│█████████│            │░░█████░░│            │░░█████░░│
└─────────┘            └─────────┘            └─────────┘
(blanc=1, noir=0)
```

### Complexité
- **Temps**: O(W·H)
- **Espace**: O(1) - modifie in-place

---

## 7. `spectrum_image(re, im, W, H, orig_h, orig_w)`

### Description
Convertit le spectre de Fourier complexe en **image du spectre displayable**. Calcule la magnitude et l'applique une échelle logarithmique pour mieux visualiser.

### Algorithme
```python
def spectrum_image(re, im, W, H, orig_h, orig_w):
    # Créer l'array de sortie aux dimensions originales
    logs = np.zeros((orig_h, orig_w), dtype=np.float64)
    
    # Calculer la magnitude et l'échelle logarithmique
    for y in range(orig_h):
        for x in range(orig_w):
            # Magnitude d'un nombre complexe: |z| = sqrt(re² + im²)
            mag = np.sqrt(re[y, x]**2 + im[y, x]**2)
            
            # Échelle logarithmique: log(1 + mag)
            # (log1p évite les problèmes numériques près de 0)
            logs[y, x] = np.log1p(mag)
    
    # Normaliser à [0, 255] pour affichage
    mx = logs.max()
    if mx > 0:
        logs = logs / mx * 255
    
    return logs.astype(np.uint8)
```

### Étapes détaillées

#### **1. Calcul de la magnitude**
Pour un nombre complexe z = a + bi:
```
|z| = sqrt(a² + b²)
```
Représente l'amplitude de la fréquence.

#### **2. Transformation logarithmique**
```
log(1 + magnitude)
```
**Pourquoi logarithmique?**
- Le spectre a souvent des valeurs extrêmement variées
- Les composantes DC (basses fréquences) dominent numériquement
- L'échelle log compresse les grandes valeurs et amplifie les petites
- Permet de voir tous les détails du spectre (ne reste pas noyé par les DC)

**Exemple**:
```
Magnitude brute:   [1, 100, 10000]
Après log:         [0.69, 4.61, 9.21]  ← beaucoup mieux distribuées!
```

#### **3. Normalisation à [0, 255]**
```
image_display = (logs / max(logs)) * 255
```
Convertit à la plage uint8 pour affichage d'image standard.

### Exemple visuel
```
Spectre complexe:    Magnitude:         Logarithme:      Affichage:
1+2i                  |1+2i|≈2.24        log(1+2.24)≈1.2   ≈75
100+50i              |100+50i|≈111.8    log(1+111.8)≈4.7  ≈255
10+5i                |10+5i|≈11.2       log(1+11.2)≈2.5   ≈155
```

### Complexité
- **Temps**: O(orig_h × orig_w)
- **Espace**: O(orig_h × orig_w)

---

## Résumé des Complexités

| Fonction | Temps | Espace |
|----------|-------|--------|
| `_next_pow2` | O(log n) | O(1) |
| `fft1d` | O(n log n) | O(1) |
| `fft2d` | O(W·H·log W·log H) | O(W·H) |
| `ifft2d` | O(W·H·log W·log H) | O(1) |
| `fft_shift` | O(W·H) | O(W·H) |
| `apply_frequency_mask` | O(W·H) | O(1) |
| `spectrum_image` | O(orig_h × orig_w) | O(orig_h × orig_w) |

---

## Concepts Mathématiques Clés

### Transformée de Fourier Discrète (DFT)
```
X[k] = Σ(n=0 to N-1) x[n] * e^(-2πikn/N)
```
Décompose un signal en ses composantes fréquentielles.

### FFT vs DFT
- **DFT naïve**: O(n²)
- **FFT (Cooley-Tukey)**: O(n log n)
- **Speedup**: Pour n=1024, FFT est ~100× plus rapide!

### Nombres complexes en FFT
```
Pair réel-imaginaire: (a, b) = a + bi
Multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                              = (ac - bd) + (ad + bc)i
```

### Racines primitives de l'unité
Les facteurs de rotation dans la FFT:
```
w = e^(-2πi/N) = cos(-2π/N) + i*sin(-2π/N)
```
Ces nombres complexes ont magnitude 1 et distribuent les phases.

---

## Cas d'usage dans le projet

1. **Analyse fréquentielle d'images**: Identifier les motifs et structures
2. **Filtering spatial**: Appliquer des filtres passe-bas/haut/bande
3. **Compression**: Garder seulement les coefficients importants
4. **Débruitage**: Éliminer les hautes fréquences bruitées

---

## Références

- **Cooley-Tukey FFT**: Algorithme classique de décomposition
- **Bit-Reversal**: Technique de réorganisation d'index
- **Butterfly Operations**: Opérations élémentaires de la FFT
- **Magnitude et Phase**: Représentation cartésienne vs polaire

