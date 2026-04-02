# Documentation - Morphologie Mathématique

## Vue d'ensemble

Le module `morpho.py` implémente les **opérateurs de morphologie mathématique**, branche de l'imagerie numérique qui traite les formes et structures des objets. La morphologie utilise la **dilatation** et l'**érosion** comme opérateurs de base pour effectuer des opérations complexes comme le nettoyage, la segmentation et l'extraction de contours sur des images binaires.

---

## 1. `_threshold(pixels: np.ndarray, seuil: int) -> np.ndarray`

### Description
Fonction **interne** de seuillage binaire. Convertit l'image en niveaux de gris en image binaire (0 ou 255) à partir d'un seuil.

### Implémentation
```python
def _threshold(pixels: np.ndarray, seuil: int) -> np.ndarray:
    out = np.zeros_like(pixels)
    h, w = pixels.shape
    for y in range(h):
        for x in range(w):
            out[y, x] = 255 if pixels[y, x] >= seuil else 0
    return out
```

### Formule
$$I_{\text{out}}[y, x] = \begin{cases}
255 & \text{si } I[y, x] \geq \text{seuil} \\
0 & \text{sinon}
\end{cases}$$

### Utilité
La morphologie mathématique travaille sur des **images binaires**. Ce seuillage prépare l'image pour les opérations suivantes.

### Exemple
```
Image gris:       Après threshold (seuil=128):
[100 150 200]     [  0 255 255]
[120 160 180]  →  [  0 255 255]
[110 140 170]     [  0   0 255]
```

### Complexité
- **Temps**: O(h × w)
- **Espace**: O(h × w)

---

## 2. `erode(pixels: np.ndarray) -> np.ndarray`

### Description
L'**érosion morphologique** est un opérateur qui **rétrécit les objets blancs** de l'image. Remplace chaque pixel par le **minimum** de sa fenêtre 3×3. Avec un élément structurant carré, c'est un filtre de minimum.

### Implémentation
```python
def erode(pixels: np.ndarray) -> np.ndarray:
    h, w = pixels.shape
    out  = np.zeros_like(pixels)
    
    for y in range(h):
        for x in range(w):
            mn = 255  # Valeur maximale initiale
            
            # Balayer la fenêtre 3×3
            for ky in range(-1, 2):      # dy = -1, 0, +1
                for kx in range(-1, 2):  # dx = -1, 0, +1
                    # Coordonnées avec gestion des bords
                    py = max(0, min(h-1, y+ky))
                    px = max(0, min(w-1, x+kx))
                    
                    # Garder la valeur minimale
                    if pixels[py, px] < mn:
                        mn = pixels[py, px]
            
            out[y, x] = mn
    
    return out
```

### Formule mathématique
L'érosion par un élément structurant B:

$$E_B(I)[y, x] = \min_{(dy, dx) \in B} I[y + dy, x + dx]$$

Pour un élément structurant carré 3×3:
$B = \{(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)\}$

### Algorithme détaillé

#### **1. Para chaque pixel (y, x)**
```
Fenêtre 3×3 centrée en (y, x):
│ (y-1,x-1)  (y-1,x)  (y-1,x+1) │
│ (y,  x-1) [y, x]  (y,  x+1) │
│ (y+1,x-1)  (y+1,x)  (y+1,x+1) │
```

#### **2. Gestion des bords (boundary conditions)**
```python
py = max(0, min(h-1, y+ky))  # Clamp en [0, h-1]
px = max(0, min(w-1, x+kx))  # Clamp en [0, w-1]
```
Réplique les pixels de bordure (mode "edge").

#### **3. Trouver le minimum**
```python
mn = min(tous les pixels dans la fenêtre)
```

### Exemple visuel

```
Image binaire originale (W = blanc/255, B = noir/0):
B W W          Après érosion:
W W W   →      B W B
B W W          B B B

Le W central reste si TOUS les voisins incluent W
Sinon → B (un seul B suffit pour qu'il devienne B)
```

#### **Exemple détaillé pixel-par-pixel**
```
Fenêtre autour (0,0):
    x=-1  x=0  x=1
y=-1: 0    0    127  (clampé à bord)
y=0:  0    100  200
y=1:  50   150  200

min(0, 0, 127, 0, 100, 200, 50, 150, 200) = 0
→ out[0, 0] = 0
```

### Visualisation de l'effet global
```
Avant érosion:           Après 1 érosion:      Après 2 érosions:
┌─────────────┐         ┌────────┐            ┌──────┐
│ ███████████ │         │ ██████ │            │ ████ │
│ ███████████ │         │ ██████ │            │ ████ │
│ ███████████ │    →    │ ██████ │       →    │ ████ │
│ ███████████ │         │ ██████ │            │ ████ │
└─────────────┘         └────────┘            └──────┘

Les objets blancs rétrécissent
Les trous noirs grandissent
```

### Propriétés mathématiques
- **Idempotent**: erode(erode(I, n)) avec n itérations = arrêt après quelques cycles
- **Monotone**: Si I₁ ⊆ I₂ alors erode(I₁) ⊆ erode(I₂)
- **Dual avec dilatation**: erode(I) = ¬dilate(¬I)
- **Anti-extensive**: erode(I) ⊆ I

### Cas d'usage
- Réduire objets blancs
- Supprimer petites structures (bruit)
- Séparer objets connectés

### Complexité
- **Temps**: O(h × w × 9) = O(h × w)
- **Espace**: O(h × w)

---

## 3. `dilate(pixels: np.ndarray) -> np.ndarray`

### Description
La **dilatation morphologique** est le **dual de l'érosion**. Elle **agrandit les objets blancs**. Remplace chaque pixel par le **maximum** de sa fenêtre 3×3.

### Implémentation
```python
def dilate(pixels: np.ndarray) -> np.ndarray:
    h, w = pixels.shape
    out  = np.zeros_like(pixels)
    
    for y in range(h):
        for x in range(w):
            mx = 0  # Valeur minimale initiale
            
            # Balayer la fenêtre 3×3
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    # Coordonnées avec gestion des bords
                    py = max(0, min(h-1, y+ky))
                    px = max(0, min(w-1, x+kx))
                    
                    # Garder la valeur maximale
                    if pixels[py, px] > mx:
                        mx = pixels[py, px]
            
            out[y, x] = mx
    
    return out
```

### Formule mathématique
La dilatation par élément structurant B:

$$D_B(I)[y, x] = \max_{(dy, dx) \in B} I[y + dy, x + dx]$$

### Algorithme
Identique à l'érosion sauf:
- Trouver le **maximum** au lieu du minimum
- Initialiser avec `mx = 0` (valeur minimale)

### Exemple visuel

```
Image binaire originale:
B W B          Après dilatation:
W W W   →      W W W
B W B          W W W

Un seul W voisin suffir pour que le pixel devienne W
```

#### **Exemple détaillé pixel-par-pixel**
```
Fenêtre autour (0,1):
    x=0   x=1  x=2
y=0: 0    255  0
y=1: 255  255  255
y=2: 0    255  0

max(0, 255, 0, 255, 255, 255, 0, 255, 0) = 255
→ out[0, 1] = 255
```

### Visualisation de l'effet global
```
Avant dilatation:        Après 1 dilatation:   Après 2 dilatations:
┌────┐                  ┌──────────┐          ┌────────────────┐
│ ██ │                  │ ████████ │          │ ████████████████│
│ ██ │            →     │ ████████ │      →   │ ████████████████│
│ ██ │                  │ ████████ │          │ ████████████████│
└────┘                  └──────────┘          └────────────────┘

Les objets blancs grandissent
Les trous noirs rétrécissent
```

### Propriétés mathématiques
- **Extensif**: dilate(I) ⊇ I
- **Monotone**: Si I₁ ⊆ I₂ alors dilate(I₁) ⊆ dilate(I₂)
- **Dual avec érosion**: dilate(I) = ¬erode(¬I)
- **Associatif**: dilate(dilate(I)) = dilate(dilate(I))

### Cas d'usage
- Agrandir objets blancs
- Fermer trous petits
- Connecter objets proches
- Renforcer structures

### Complexité
- **Temps**: O(h × w)
- **Espace**: O(h × w)

---

## 4. `_iter(fn, pixels, n) -> np.ndarray`

### Description
Fonction **utilitaire** qui applique itérativement un opérateur `n` fois. Permet de comparer l'effet de plusieurs applications consécutives.

### Implémentation
```python
def _iter(fn, pixels, n):
    for _ in range(n):
        pixels = fn(pixels)
    return pixels
```

### Utilité
```python
# Plutôt que:
result = erode(erode(erode(image)))

# On écrit:
result = _iter(erode, image, 3)
```

### Paramètres
- `fn`: Fonction à appliquer (par ex: `erode`, `dilate`)
- `pixels`: Image d'entrée
- `n`: Nombre d'itérations

### Exemple
```python
# Appliquer érosion 4 fois
eroded_4x = _iter(erode, image, 4)

# Équivalent à:
temp = erode(image)  # 1ère érosion
temp = erode(temp)   # 2ème érosion
temp = erode(temp)   # 3ème érosion
result = erode(temp) # 4ème érosion
```

### Complexité
- **Temps**: O(n × h × w)
- **Espace**: O(h × w)

---

## 5. `opening(pixels: np.ndarray, iterations: int = 1) -> np.ndarray`

### Description
L'**ouverture (opening)** est une **érosion suivie d'une dilatation**. Élimine les petits objets blancs et lisse les contours. Conservateur en blanc: ne crée pas de blanc, ne supprime que de petites structures.

### Implémentation
```python
def opening(pixels: np.ndarray, iterations: int = 1) -> np.ndarray:
    return _iter(dilate, _iter(erode, pixels, iterations), iterations)
```

### Logique
```
Image originale
        ↓
    Érosion (n fois)  → Rétrécit objets, supprime petits bruits
        ↓
    Dilatation (n fois) → Restaure objets, lisse contours
        ↓
    Résultat: Image "nettoyée"
```

### Formule mathématique
$$\text{Opening}(I) = D_B(E_B(I))$$

Récursivement pour n itérations:
$$\text{Opening}_n(I) = D_B^n(E_B^n(I))$$

### Propriétés
- **Idempotent**: opening(opening(I)) = opening(I)
- **Anti-extensive**: opening(I) ⊆ I
- **Préserve structure**: Ne crée pas de nouveaux objets blancs

### Exemple visuel

```
Image avec bruit (petits points blancs):
█ ░ ░ ░ █
░ ░ ░ ░ ░
░ ░ █ ░ ░
░ ░ ░ ░ ░
█ ░ ░ ░ █

Après opening (iterations=1):
█ ░ ░ ░ █
░ ░ ░ ░ ░
░ ░ ░ ░ ░      ← Petits points supprimés
░ ░ ░ ░ ░
█ ░ ░ ░ █

Après opening (iterations=2):
░ ░ ░ ░ ░
░ ░ ░ ░ ░
░ ░ ░ ░ ░      ← Even plus nettoyer, objets perdent pixels
░ ░ ░ ░ ░
░ ░ ░ ░ ░
```

### Cas d'usage
- **Débruitage**: Supprimer bruit blanc isolé
- **Nettoyage**: Vrai bruit blanc et petites impuretés
- **Segmentation**: Préparer l'image pour détection
- **Morphologie**: Base pour top-hat et gradient

### Variantes
```
Paramètre iterations:
- iterations=1: Léger nettoyage
- iterations=2: Modéré
- iterations=3+: Nettoyage agressif
```

### Complexité
- **Temps**: O(iterations × h × w)
- **Espace**: O(h × w)

---

## 6. `closing(pixels: np.ndarray, iterations: int = 1) -> np.ndarray`

### Description
La **fermeture (closing)** est une **dilatation suivie d'une érosion** - l'inverse de l'ouverture. Ferme les petits trous noirs et lisse les contours. Conservateur en noir: ne supprime pas de noir, ne crée que pour combler.

### Implémentation
```python
def closing(pixels: np.ndarray, iterations: int = 1) -> np.ndarray:
    return _iter(erode, _iter(dilate, pixels, iterations), iterations)
```

### Logique
```
Image originale
        ↓
    Dilatation (n fois) → Agrandit objets, ferme trous
        ↓
    Érosion (n fois)   → Restaure objets, lisse contours
        ↓
    Résultat: Image "comblée"
```

### Formule mathématique
$$\text{Closing}(I) = E_B(D_B(I))$$

Récursivement pour n itérations:
$$\text{Closing}_n(I) = E_B^n(D_B^n(I))$$

### Propriétés
- **Idempotent**: closing(closing(I)) = closing(I)
- **Extensif**: closing(I) ⊇ I
- **Préserve noir**: Ne supprime pas de pixels noirs

### Exemple visuel

```
Image avec trous noirs:
█ █ █ █ █
█ ░ ░ ░ █
█ ░ ░ ░ █
█ ░ ░ ░ █
█ █ █ █ █

Après closing (iterations=1):
█ █ █ █ █
█ █ █ █ █
█ █ █ █ █      ← Trous comblés
█ █ █ █ █
█ █ █ █ █

Après closing (iterations=2):
█ █ █ █ █
█ █ █ █ █
█ █ █ █ █      ← Objets agrandis, contours moins nets
█ █ █ █ █
█ █ █ █ █
```

### Cas d'usage
- **Comblement de trous**: Fermer cavités dedans objets
- **Lissage**: Adoucir contours irréguliers
- **Nettoyage noir**: Supprimer faux trous
- **Préparation segmentation**: Objets plus compacts

### Comparaison opening vs closing
```
        Opening (erode then dilate)
Image originale
        ↓
     Érosion     ← Supprime blanc
        ↓
    Dilatation   ← Restaure blanc

        Closing (dilate then erode)
Image originale
        ↓
    Dilatation   ← Ajoute blanc
        ↓
     Érosion     ← Supprime blanc excessif, restaure noir
```

### Complexité
- **Temps**: O(iterations × h × w)
- **Espace**: O(h × w)

---

## 7. `top_hat(pixels: np.ndarray, iterations: int = 1) -> np.ndarray`

### Description
Le **top-hat (chapeau haut-de-forme)** extrait les **petites structures blanches** qui disparaissent avec l'ouverture. Utile pour la détection de petits objets ou défauts.

### Implémentation
```python
def top_hat(pixels: np.ndarray, iterations: int = 1) -> np.ndarray:
    opened = opening(pixels, iterations)
    diff   = pixels.astype(np.int16) - opened.astype(np.int16)
    return np.clip(diff, 0, 255).astype(np.uint8)
```

### Formule mathématique
$$\text{Top-hat}(I) = I - \text{Opening}(I)$$

Différence: ce qui a été enlevé par l'ouverture.

### Étapes

#### **1. Appliquer opening**
```python
opened = opening(pixels, iterations)
```
Élimine les petites structures blanches.

#### **2. Calculer la différence**
```python
diff = pixels - opened
```
Ce qui n'est **pas** dans `opened` = ce qui a été enlevé.

#### **3. Clipping et conversion**
```python
result = clip(diff, 0, 255)  # Gérer les valeurs négatives
```

### Exemple visuel

```
Image original:
█ ░ ░ ░ █
░ ░ ░ ░ ░
░ ░ █ ░ ░  (petit objet au centre)
░ ░ ░ ░ ░
█ ░ ░ ░ █

Opening (supprime petit objet):
█ ░ ░ ░ █
░ ░ ░ ░ ░
░ ░ ░ ░ ░
░ ░ ░ ░ ░
█ ░ ░ ░ █

Top-hat (original - opening):
░ ░ ░ ░ ░
░ ░ ░ ░ ░
░ ░ █ ░ ░  ← SEULEMENT les petites structures
░ ░ ░ ░ ░
░ ░ ░ ░ ░
```

### Propriétés
- **Linéarité**: Top-hat(a·I) = a·Top-hat(I)
- **Extraction**: Décorticage des petites structures
- **Additivité**: I = Opening(I) + Top-hat(I)

### Cas d'usage
- **Détection de défauts**: Petites anomalies dans surface
- **Extraction de détails**: Tout ce qui est petit et blanc
- **Amélioration de contraste**: Pour très petits objets
- **Inspection de surface**: Rayures, taches, poussière

### Exemple réel
```
Wafer semi-conducteur avec défauts microscopiques:

Image originale:
[surface légèrement bruitée avec défauts]

Après top-hat:
[seulement les défauts en blanc, surface unie en noir]
```

### Complexité
- **Temps**: O(iterations × h × w)
- **Espace**: O(h × w)

---

## 8. `black_hat(pixels: np.ndarray, iterations: int = 1) -> np.ndarray`

### Description
Le **black-hat (chapeau noir)** est le **dual du top-hat**. Extrait les **petites structures noires** (trous) qui disparaissent avec la fermeture. Utile pour détecter petits trous ou cavités.

### Implémentation
```python
def black_hat(pixels: np.ndarray, iterations: int = 1) -> np.ndarray:
    closed = closing(pixels, iterations)
    diff   = closed.astype(np.int16) - pixels.astype(np.int16)
    return np.clip(diff, 0, 255).astype(np.uint8)
```

### Formule mathématique
$$\text{Black-hat}(I) = \text{Closing}(I) - I$$

Différence: ce qui a été ajouté par la fermeture.

### Étapes

#### **1. Appliquer closing**
```python
closed = closing(pixels, iterations)
```
Combine les petits trous.

#### **2. Calculer la différence**
```python
diff = closed - pixels
```
Ce qui a été **ajouté** par la fermeture = petits trous.

#### **3. Clipping et conversion**
```python
result = clip(diff, 0, 255)
```

### Exemple visuel

```
Image original:
█ ░ █ ░ █
░ ░ ░ ░ ░
█ ░ ░ ░ █
░ ░ ░ ░ ░
█ ░ █ ░ █

Closing (remplit trous):
█ █ █ █ █
░ ░ ░ ░ ░
█ ░ ░ ░ █
░ ░ ░ ░ ░
█ █ █ █ █

Black-hat (closing - original):
░ █ ░ █ ░
░ ░ ░ ░ ░
░ ░ ░ ░ ░  ← SEULEMENT les petits trous remplis
░ ░ ░ ░ ░
░ █ ░ █ ░
```

### Propriétés
- **Linéarité**: Black-hat(a·I) = a·Black-hat(I)
- **Extraction de creux**: Détecte cavités
- **Additivité**: Closing(I) = I + Black-hat(I)

### Cas d'usage
- **Détection de cavités**: Petits trous/creux
- **Inspection de remplissage**: Objets mal remplis
- **Extraction de défauts creux**: Contre-partie du top-hat
- **Analyse de porosité**: Cavités dans matériaux

### Comparaison top-hat vs black-hat
```
Top-hat:       Extrait structures BLANCHES petit
               I - Opening(I)
               Bruits blanc, défauts convexes

Black-hat:     Extrait structures NOIRES petites
               Closing(I) - I
               Cavités, trous, défauts concaves
```

### Complexité
- **Temps**: O(iterations × h × w)
- **Espace**: O(h × w)

---

## 9. `morpho_gradient(pixels: np.ndarray) -> np.ndarray`

### Description
Le **gradient morphologique** calcule la **différence entre dilatation et érosion**. Extrait les **contours/edges** de l'image. Détecte les transitions d'intensité.

### Implémentation
```python
def morpho_gradient(pixels: np.ndarray) -> np.ndarray:
    d = dilate(pixels).astype(np.int16)
    e = erode(pixels).astype(np.int16)
    return np.clip(d - e, 0, 255).astype(np.uint8)
```

### Formule mathématique
$$\text{Gradient}(I) = D_B(I) - E_B(I)$$

Où D = dilatation, E = érosion.

### Étapes

#### **1. Dilatation**
```python
d = dilate(pixels)
```
Agrandit objets blancs.

#### **2. Érosion**
```python
e = erode(pixels)
```
Rétrécit objets blancs.

#### **3. Différence**
```python
gradient = d - e
```
Là où les deux diffèrent = les contours.

### Exemple visuel

```
Image originale:
░ ░ ░ ░ ░
░ █ █ █ ░
░ █ █ █ ░
░ █ █ █ ░
░ ░ ░ ░ ░

Dilatation:
░ █ █ █ ░
█ █ █ █ █
█ █ █ █ █
█ █ █ █ █
░ █ █ █ ░

Érosion:
░ ░ ░ ░ ░
░ ░ █ ░ ░
░ █ █ █ ░
░ ░ █ ░ ░
░ ░ ░ ░ ░

Gradient (dilate - erode):
░ █ █ █ ░
█ █ ░ █ █
█ ░ ░ ░ █  ← Contours du carré
█ █ ░ █ █
░ █ █ █ ░
```

#### **Explication**
- **Points où dilate = erode**: Intérieur de l'objet → 0 (noir)
- **Points où dilate ≠ erode**: Contact avec le bord → > 0 (blanc dans gradient)
- **Résultat**: "Squelette" des contours

### Propriétés mathématiques
- **Linéarité**: Gradient(a·I) = a·Gradient(I)
- **Symétrie**: Gradient(I) ≥ 0 toujours
- **Localisation**: Réponse maximale aux edges

### Cas d'usage
- **Détection de contours**: Délimiter objets
- **Segmentation**: Séparer objets
- **Analyse de forme**: Squelette des objets
- **Mesure de bordure**: Épaisseur de transition

### Avantages sur Sobel/Laplacien
```
Morpho gradient:  Basé sur min/max, peu sensible au bruit
Sobel:            Derivées numériques, sensible au bruit
Laplacien:        Second ordre, très sensible au bruit

=> Morpho gradient est robuste pour images binaires
```

### Complexité
- **Temps**: O(h × w)
- **Espace**: O(h × w)

---

## 10. `run_morpho(pixels: np.ndarray, op: str, seuil: int, iterations: int) -> np.ndarray`

### Description
Fonction **principale orchestratrice** qui enchaîne le **seuillage binaire** puis **applique l'opérateur morphologique** choisi. Point d'entrée pour l'interface utilisateur.

### Implémentation
```python
def run_morpho(pixels: np.ndarray, op: str, seuil: int, iterations: int) -> np.ndarray:
    # Étape 1: Convertir en binaire
    binary = _threshold(pixels, seuil)
    
    # Étape 2: Appliquer opérateur
    ops = {
        "Érosion":    lambda p: _iter(erode,  p, iterations),
        "Dilatation": lambda p: _iter(dilate, p, iterations),
        "Opening":    lambda p: opening(p, iterations),
        "Closing":    lambda p: closing(p, iterations),
        "Top Hat":    lambda p: top_hat(p, iterations),
        "Black Hat":  lambda p: black_hat(p, iterations),
        "Gradient":   lambda p: morpho_gradient(p),
    }
    
    return ops[op](binary)
```

### Paramètres
```python
pixels      # Image originale en niveaux gris (H × W)
op          # Opérateur morphologique à appliquer:
            #   "Érosion", "Dilatation", "Opening", "Closing"
            #   "Top Hat", "Black Hat", "Gradient"
seuil       # Pour seuillage: pixels ≥ seuil → 255, sinon 0
iterations  # Nombre de répétitions de l'opérateur
```

### Pipeline complet

```
Image gris (H × W, valeurs [0, 255])
        ↓
1. Seuillage binaire (seuil) → Image binaire (0 ou 255)
        ↓
2. Opérateur morpho (op, iterations) → Image morpho
        ↓
3. Résultat final
```

### Exemple complet

```python
# Chargement
image_gris = load_image("photo.jpg")  # [0, 255]

# Exécution
result = run_morpho(
    pixels=image_gris,
    op="Opening",
    seuil=128,
    iterations=2
)

# Steps internes:
# 1. _threshold(image_gris, 128)       → binaire
# 2. opening(binaire, iterations=2)    → 4 × morpho
# 3. Retour résultat
```

### Sélection d'opérateur

| Opérateur | Utilité | Pre-req |
|-----------|---------|---------|
| Érosion | Rétrécir blanc, supprimer bruit | Bruit blanc isolé |
| Dilatation | Agrandir blanc, remplir trous | Besoin expansion |
| Opening | Nettoyage général | Bruit blanc + petits objets |
| Closing | Comblement trous | Cavités internes |
| Top Hat | Détecter défauts blancs | Inspection surface |
| Black Hat | Détecter cavités | Cavités à scanner |
| Gradient | Délimiter contours | Besoin segmentation |

### Guide de paramètres

#### **Choix du seuil**
```
Seuil bas (50-100):     Plus de pixels blancs → plus de bruit
Seuil moyen (128):      Équilibré
Seuil haut (150-180):   Moins de blanc → risque perte info
```

#### **Nombre d'itérations**
```
iterations=1:   Effet léger
iterations=2:   Modéré (recommandé)
iterations=3+:  Agressif, risque perte structure
```

### Cas d'usage pratiques

#### **1. Débruitage d'une photo**
```python
run_morpho(image, op="Opening", seuil=128, iterations=2)
```

#### **2. Segmentation d'objet**
```python
run_morpho(image, op="Closing", seuil=100, iterations=1)
```

#### **3. Détection de défaut**
```python
run_morpho(image, op="Top Hat", seuil=120, iterations=2)
```

#### **4. Extraction de contours**
```python
run_morpho(image, op="Gradient", seuil=128, iterations=1)
```

### Complexité totale
- **Temps**: O(iterations × h × w)
- **Espace**: O(h × w)

---

## Résumé des Complexités

| Fonction | Temps | Espace | Utilité |
|----------|-------|--------|---------|
| `_threshold` | O(h·w) | O(h·w) | Binaire |
| `erode` | O(h·w) | O(h·w) | Rétrécir blanc |
| `dilate` | O(h·w) | O(h·w) | Agrandir blanc |
| `_iter` | O(n·h·w) | O(h·w) | Itération |
| `opening` | O(n·h·w) | O(h·w) | Nettoyage |
| `closing` | O(n·h·w) | O(h·w) | Comblement |
| `top_hat` | O(n·h·w) | O(h·w) | Détails blancs |
| `black_hat` | O(n·h·w) | O(h·w) | Détails noirs |
| `morpho_gradient` | O(h·w) | O(h·w) | Contours |
| `run_morpho` | O(n·h·w) | O(h·w) | Orchestration |

---

## Concepts Fondamentaux

### Morphologie Mathématique
Branche de l'imagerie basée sur la théorie des ensembles.

**Opérateurs de base**:
- **Érosion**: Minimum local (rétrécit blanc)
- **Dilatation**: Maximum local (agrandit blanc)

**Opérateurs composés**:
- **Opening** = Érosion + Dilatation
- **Closing** = Dilatation + Érosion

### Propriétés fondamentales

```
1. Idempotence:
   opening(opening(I)) = opening(I)
   closing(closing(I)) = closing(I)

2. Dualité:
   erode(I) ≡ ¬dilate(¬I)
   dilate(I) ≡ ¬erode(¬I)

3. Décomposition:
   opening = "I avec bruit blanc enlevé"
   closing = "I avec cavités comblées"

4. Additivité:
   I = opening(I) + top_hat(I)
   closing(I) = I + black_hat(I)
```

### Élément structurant (SE)
Détermine la forme du voisinage utilisé:
```
SE = 3×3 carré (utilisé ici):
┌─────┐
│ 1 1 1 │
│ 1 1 1 │
│ 1 1 1 │
└─────┘

SE = croix:
    1
  1 1 1
    1

SE = diamant:
  1 1 1
1 1 1 1
  1 1 1
```

---

## Bonnes pratiques

### 1. Seuillage adaptatif
```python
# Plutôt que de fixer seuil = 128 pour toute image
# Utiliser:
seuil = (image.min() + image.max()) // 2  # Médiane
# Ou analyser histogramme

seuil = compute_histogram(image)  # Otsu
```

### 2. Ordre des opérateurs
```
Pour débruitage général:
  Meilleur: opening (enlève bruit blanc)
  Moins bon: erode seule (perte structure)
  Mauvais: dilate seule (ajoute bruit)

Pour comblement:
  Meilleur: closing (remplit trous)
  Moins bon: dilate seule (agrandit trop)
```

### 3. Itérations conservatrices
```
Commencer par iterations=1
Augmenter progressivement
Ne pas dépasser 3-4 itérations (risque artefacts)
```

### 4. Combinaisons d'opérateurs
```
Nettoyage complet:
  result = opening(closing(image))
  # Ferme trous, puis enlève bruit blanc

Extraction complète:
  white_details = top_hat(image)
  black_details = black_hat(image)
  all_details = white_details + black_details
```

### 5. Visualisation progressive
```python
for i in range(1, 4):
    result = run_morpho(image, "Opening", 128, i)
    display(f"Opening iteration {i}", result)
    # Voir progression de l'effet
```

---

## Comparaisons avec autres méthodes

### Morpho vs Filtres spaciaux
```
            | Morpho | Filtres |
Bruit blanc | Excellent | Bon (gaussien) |
Contours | Nets | Flous |
Vitesse | Rapide | Rapide |
Paramètres | Simples | Plus |
```

### Opening vs Gaussian filter
```
Image avec bruit poivre:
█ ░ ░ ░ █
░ ░ ░ ░ ░
░ ░ █ ░ ░  ← petit bruit
░ ░ ░ ░ ░
█ ░ ░ ░ █

Après Gaussian:
█ ░ ░ ░ █
░ ░ ░ ░ ░
░ ░ ░ ░ ░  ← lissé, bruit réduit mais pas supprimé
░ ░ ░ ░ ░
█ ░ ░ ░ █

Après Opening:
█ ░ ░ ░ █
░ ░ ░ ░ ░
░ ░ ░ ░ ░  ← complètement éliminé!
░ ░ ░ ░ ░
█ ░ ░ ░ █
```

---

## Références théoriques

- **Morphologie mathématique**: Serra, J. (1982)
- **Érosion/Dilatation**: Opérateurs fondamentaux set-based
- **Opening/Closing**: Opérateurs composés idempotents
- **Top-hat/Black-hat**: Transformées différences
- **Gradient morpho**: Détection contours robuste

