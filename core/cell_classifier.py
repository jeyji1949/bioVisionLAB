"""
core/cell_classifier.py
=======================
Module C — Cell-type classifier, 100% NumPy, zero scikit-learn.

What it does (in plain terms)
------------------------------
1. Takes the binary image produced by M4 (morphology)
2. Finds every connected object using flood-fill  (extended from morpho_panel)
3. Measures 5 things about each object            (feature extraction)
4. Groups similar objects together using k-means  (unsupervised ML)
5. Labels each group as nucleus / cytoplasm / debris / uncertain

How it connects to code
--------------------------------------
  Your morpho_panel._count_objects() returns (count, mean_area).
  This module does the same flood-fill but returns the full list of
  objects with all their measurements — so M5 can use them.

Usage
-----
  from core.cell_classifier import CellClassifier

  clf = CellClassifier(n_clusters=3)
  results = clf.run(m4_binary_image, original_grayscale_image)

  results = {
      "objects":     list of dicts, one per detected object
      "clusters":    list of cluster assignments
      "labels":      list of human-readable labels per object
      "summary":     {"nucleus": 12, "cytoplasm": 8, "debris": 3}
      "features":    (n_objects, 5) numpy array
      "centroids":   (n_clusters, 5) numpy array  — k-means centroids
      "pca_coords":  (n_objects, 2) numpy array   — for scatter plot
  }
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Flood-fill — extended version of morpho_panel._flood()
#     Returns pixel coordinates of each object, not just its area
# ─────────────────────────────────────────────────────────────────────────────

def _flood_fill(binary, visited, y0, x0):
    """
    Same algorithm as morpho_panel._flood() but returns
    the full list of (y, x) pixel coordinates of the object.

    This lets us measure intensity, perimeter, shape — not just area.
    """
    stack  = [(y0, x0)]
    h, w   = binary.shape
    pixels = []

    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        if visited[y, x] or binary[y, x] == 0:
            continue
        visited[y, x] = 1
        pixels.append((y, x))
        stack += [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]

    return pixels


def extract_objects(binary):
    """
    Find all connected objects in a binary image.

    Parameters
    ----------
    binary : (H, W) uint8 array — output of M4 (values 0 or 255)

    Returns
    -------
    List of lists — each inner list is the (y,x) pixel coords of one object.
    Only objects with area >= 10 pixels are kept (noise filter).
    """
    bin_img = (binary > 127).astype(np.uint8)
    visited = np.zeros_like(bin_img)
    objects = []

    for y in range(bin_img.shape[0]):
        for x in range(bin_img.shape[1]):
            if bin_img[y, x] == 1 and visited[y, x] == 0:
                pixels = _flood_fill(bin_img, visited, y, x)
                if len(pixels) >= 10:          # ignore tiny noise specks
                    objects.append(pixels)

    return objects


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Feature extraction — 5 measurements per object
# ─────────────────────────────────────────────────────────────────────────────

def _perimeter(pixels, binary):
    """
    Count boundary pixels — pixels that have at least one background neighbour.
    This is the standard 4-connectivity perimeter definition.
    """
    h, w  = binary.shape
    count = 0
    for (y, x) in pixels:
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                count += 1
                break
            if binary[ny, nx] == 0:
                count += 1
                break
    return count


def _circularity(area, perimeter):
    """
    Circularity = 4π × area / perimeter²

    Perfect circle  → 1.0
    Elongated shape → < 1.0
    Very irregular  → close to 0

    This is the standard shape descriptor used in cell biology.
    """
    if perimeter == 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def extract_features(objects, original_pixels):
    """
    For each detected object, compute a 5-dimensional feature vector.

    Features
    --------
    0  area          number of pixels in the object
    1  perimeter     number of boundary pixels
    2  circularity   4π·area/perimeter²  (1=circle, 0=very irregular)
    3  mean_intensity average grey level of object pixels in original image
    4  std_intensity  standard deviation of grey levels (texture uniformity)

    Parameters
    ----------
    objects         : list of pixel-coord lists from extract_objects()
    original_pixels : (H,W) grayscale image — used for intensity features

    Returns
    -------
    features : (n_objects, 5) float64 array
    obj_info : list of dicts with raw measurements (for M5 display)
    """
    bin_img = np.zeros(original_pixels.shape, dtype=np.uint8)
    for obj in objects:
        for (y, x) in obj:
            bin_img[y, x] = 1

    features = []
    obj_info = []

    for obj in objects:
        ys = [p[0] for p in obj]
        xs = [p[1] for p in obj]

        area      = len(obj)
        perim     = _perimeter(obj, bin_img)
        circ      = _circularity(area, perim)

        intensities = np.array([original_pixels[y, x] for y, x in obj],
                               dtype=np.float64)
        mean_int  = float(intensities.mean())
        std_int   = float(intensities.std())

        features.append([area, perim, circ, mean_int, std_int])
        obj_info.append({
            "area":           area,
            "perimeter":      perim,
            "circularity":    round(circ, 3),
            "mean_intensity": round(mean_int, 1),
            "std_intensity":  round(std_int, 1),
            "centroid":       (int(np.mean(ys)), int(np.mean(xs))),
        })

    return np.array(features, dtype=np.float64), obj_info


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Normalisation — needed before k-means
#     Without this, area (hundreds of pixels) dominates circularity (0-1)
# ─────────────────────────────────────────────────────────────────────────────

def normalise(X):
    """
    Min-max normalise each feature column to [0, 1].

    Formula:  x_norm = (x - min) / (max - min)

    This ensures all 5 features contribute equally to the distance
    calculation in k-means — no single feature dominates just because
    it has larger absolute values.
    """
    X_norm = np.zeros_like(X)
    for col in range(X.shape[1]):
        mn = X[:, col].min()
        mx = X[:, col].max()
        if mx > mn:
            X_norm[:, col] = (X[:, col] - mn) / (mx - mn)
        else:
            X_norm[:, col] = 0.0
    return X_norm


# ─────────────────────────────────────────────────────────────────────────────
# 4.  K-means — from scratch
#     The entire algorithm is 4 steps repeated until convergence
# ─────────────────────────────────────────────────────────────────────────────

class KMeans:
    """
    K-means clustering implemented from scratch.

    The algorithm
    -------------
    1. Initialise k centroids randomly from the data points
    2. Assign each point to its nearest centroid (Euclidean distance)
    3. Recompute each centroid as the mean of its assigned points
    4. Repeat steps 2-3 until assignments stop changing

    Parameters
    ----------
    k         : number of clusters
    max_iter  : maximum number of iterations
    n_init    : number of random restarts (best result kept)
    seed      : random seed for reproducibility
    """

    def __init__(self, k=3, max_iter=100, n_init=10, seed=42):
        self.k        = k
        self.max_iter = max_iter
        self.n_init   = n_init
        self.seed     = seed
        self.centroids_  = None
        self.labels_     = None
        self.inertia_    = None

    def _euclidean(self, a, b):
        """Distance between point a and centroid b."""
        return float(np.sqrt(np.sum((a - b) ** 2)))

    def _assign(self, X, centroids):
        """Step 2 — assign each point to nearest centroid."""
        labels = np.zeros(len(X), dtype=int)
        for i, point in enumerate(X):
            distances = [self._euclidean(point, c) for c in centroids]
            labels[i] = int(np.argmin(distances))
        return labels

    def _update(self, X, labels):
        """Step 3 — recompute centroids as mean of assigned points."""
        centroids = np.zeros((self.k, X.shape[1]), dtype=np.float64)
        for cluster in range(self.k):
            mask = (labels == cluster)
            if mask.sum() > 0:
                centroids[cluster] = X[mask].mean(axis=0)
        return centroids

    def _inertia(self, X, labels, centroids):
        """Sum of squared distances to assigned centroid — lower is better."""
        total = 0.0
        for i, point in enumerate(X):
            total += self._euclidean(point, centroids[labels[i]]) ** 2
        return total

    def fit(self, X):
        """
        Run k-means with n_init random restarts, keep the best result.

        Parameters
        ----------
        X : (n_samples, n_features) float64 array
        """
        rng          = np.random.default_rng(self.seed)
        best_labels  = None
        best_cents   = None
        best_inertia = float("inf")

        for _ in range(self.n_init):
            # Step 1 — random initialisation (k-means++ would be better
            # but plain random is sufficient and easier to explain)
            idx       = rng.choice(len(X), self.k, replace=False)
            centroids = X[idx].copy()

            for _ in range(self.max_iter):
                labels     = self._assign(X, centroids)
                new_cents  = self._update(X, labels)
                # Convergence check — stop if centroids didn't move
                if np.allclose(centroids, new_cents, atol=1e-6):
                    break
                centroids = new_cents

            inertia = self._inertia(X, labels, centroids)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels  = labels.copy()
                best_cents   = centroids.copy()

        self.labels_    = best_labels
        self.centroids_ = best_cents
        self.inertia_   = best_inertia
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PCA — from scratch (2D projection for visualisation)
# ─────────────────────────────────────────────────────────────────────────────

def pca_2d(X):
    """
    Project n-dimensional feature vectors down to 2D for scatter plot.

    Steps
    -----
    1. Centre the data (subtract mean)
    2. Compute the covariance matrix  C = X^T X / (n-1)
    3. Find eigenvectors of C
    4. Project onto the 2 eigenvectors with largest eigenvalues

    The two axes of the scatter plot are the directions of
    maximum variance in the original 5D feature space.

    Returns
    -------
    coords : (n_samples, 2) array — x,y coordinates for scatter plot
    explained_variance : (2,) — % variance explained by each axis
    """
    X_c   = X - X.mean(axis=0)                    # step 1 — centre
    cov   = np.dot(X_c.T, X_c) / (len(X) - 1)    # step 2 — covariance
    vals, vecs = np.linalg.eigh(cov)              # step 3 — eigenvectors
    # eigh returns ascending order — reverse to get largest first
    idx   = np.argsort(vals)[::-1]
    vecs  = vecs[:, idx]
    vals  = vals[idx]
    # step 4 — project onto top 2 components
    coords = np.dot(X_c, vecs[:, :2])
    total  = vals.sum()
    explained = (vals[:2] / total * 100) if total > 0 else np.array([0.0, 0.0])
    return coords, explained


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Biological labelling
#     Maps cluster index → biological cell type
#     Based on which centroid has highest circularity (→ nucleus)
# ─────────────────────────────────────────────────────────────────────────────

CELL_TYPES = ["nucleus", "cytoplasm", "debris", "uncertain"]

# French labels for the UI
CELL_LABELS_FR = {
    "nucleus":    "Noyau",
    "cytoplasm":  "Cytoplasme",
    "debris":     "Débris",
    "uncertain":  "Incertain",
}

def assign_biological_labels(centroids, k):
    """
    Heuristic mapping from cluster index to cell type.

    Rules (based on domain knowledge)
    -----------------------------------
    - Highest circularity centroid  → nucleus   (round, compact)
    - Lowest circularity centroid   → debris    (irregular, fragmented)
    - Middle cluster(s)             → cytoplasm / uncertain

    Circularity is feature index 2 in our feature vector.

    Parameters
    ----------
    centroids : (k, 5) array of cluster centroids (normalised feature space)
    k         : number of clusters

    Returns
    -------
    cluster_to_type : dict  {cluster_index: "nucleus"/"cytoplasm"/...}
    """
    circ_col   = 2   # circularity is the 3rd feature
    circ_vals  = centroids[:, circ_col]
    sorted_idx = np.argsort(circ_vals)[::-1]   # highest circularity first

    cluster_to_type = {}
    type_list = CELL_TYPES[:k] if k <= len(CELL_TYPES) else CELL_TYPES

    # Most circular → nucleus, least circular → debris, rest → cytoplasm
    for rank, cluster_idx in enumerate(sorted_idx):
        if rank < len(type_list):
            cluster_to_type[int(cluster_idx)] = type_list[rank]
        else:
            cluster_to_type[int(cluster_idx)] = "uncertain"

    return cluster_to_type


# ─────────────────────────────────────────────────────────────────────────────
# 7.  High-level wrapper — the object you import from M5 / report_panel
# ─────────────────────────────────────────────────────────────────────────────

class CellClassifier:
    """
    Complete Module C pipeline in one object.

    Usage
    -----
      clf     = CellClassifier(n_clusters=3)
      results = clf.run(m4_result, original_pixels)

    Results dict
    ------------
      objects      : list of dicts — one per detected object with measurements
      labels       : list of str   — cell type per object ("nucleus", ...)
      labels_fr    : list of str   — French labels for UI
      summary      : dict          — counts per cell type
      summary_pct  : dict          — percentages per cell type
      features     : (N,5) array  — raw feature matrix
      features_norm: (N,5) array  — normalised feature matrix
      pca_coords   : (N,2) array  — for scatter plot
      pca_variance : (2,)  array  — % variance explained by PC1, PC2
      centroids    : (k,5) array  — k-means centroids
      inertia      : float         — k-means quality measure
      n_objects    : int           — total objects detected
    """

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def run(self, m4_result, original_pixels):
        """
        Run the full Module C pipeline.

        Parameters
        ----------
        m4_result       : (H,W) uint8 — binary image from M4 morphology
        original_pixels : (H,W) uint8 — grayscale image (for intensity features)
                          Use state.original_pixels or state.m1_result

        Returns
        -------
        dict with all results (see class docstring)
        """
        # ── Step 1: find all objects ──────────────────────────────────────────
        objects = extract_objects(m4_result)

        if len(objects) < self.n_clusters:
            # Not enough objects to cluster — return empty result
            return self._empty_result()

        # ── Step 2: measure features ──────────────────────────────────────────
        features, obj_info = extract_features(objects, original_pixels)

        # ── Step 3: normalise ─────────────────────────────────────────────────
        features_norm = normalise(features)

        # ── Step 4: k-means clustering ────────────────────────────────────────
        km = KMeans(k=self.n_clusters, max_iter=100, n_init=10, seed=42)
        km.fit(features_norm)

        # ── Step 5: biological labels ─────────────────────────────────────────
        cluster_to_type = assign_biological_labels(km.centroids_, self.n_clusters)
        labels    = [cluster_to_type[int(c)] for c in km.labels_]
        labels_fr = [CELL_LABELS_FR.get(l, l) for l in labels]

        # ── Step 6: summary counts and percentages ────────────────────────────
        summary     = {t: 0 for t in CELL_TYPES}
        for l in labels:
            if l in summary:
                summary[l] += 1

        n = len(labels)
        summary_pct = {
            t: round(summary[t] / n * 100, 1)
            for t in CELL_TYPES
        }

        # ── Step 7: PCA for visualisation ─────────────────────────────────────
        pca_coords, pca_variance = pca_2d(features_norm)

        # Add label to each object info dict
        for i, info in enumerate(obj_info):
            info["label"]    = labels[i]
            info["label_fr"] = labels_fr[i]
            info["cluster"]  = int(km.labels_[i])

        return {
            "objects":       obj_info,
            "labels":        labels,
            "labels_fr":     labels_fr,
            "summary":       summary,
            "summary_pct":   summary_pct,
            "features":      features,
            "features_norm": features_norm,
            "pca_coords":    pca_coords,
            "pca_variance":  pca_variance,
            "centroids":     km.centroids_,
            "inertia":       round(km.inertia_, 2),
            "n_objects":     n,
            "n_clusters":    self.n_clusters,
        }

    def _empty_result(self):
        return {
            "objects": [], "labels": [], "labels_fr": [],
            "summary": {t: 0 for t in CELL_TYPES},
            "summary_pct": {t: 0.0 for t in CELL_TYPES},
            "features": np.array([]), "features_norm": np.array([]),
            "pca_coords": np.array([]), "pca_variance": np.array([0.0, 0.0]),
            "centroids": np.array([]), "inertia": 0.0,
            "n_objects": 0, "n_clusters": self.n_clusters,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Context block for M6 AI chat (drop into ai_panel._build_context)
# ─────────────────────────────────────────────────────────────────────────────

def classifier_context_block(results):
    """
    Format Module C results as text for the AI chat context.

    Usage in ai_panel.py
    --------------------
      from core.cell_classifier import classifier_context_block
      ...
      if hasattr(state, "mc_results") and state.mc_results:
          lines.append(classifier_context_block(state.mc_results))
    """
    if not results or results["n_objects"] == 0:
        return "--- Module C — Cell classifier ---\nAucun objet détecté.\n"

    s   = results["summary"]
    pct = results["summary_pct"]
    lines = [
        "--- Module C — Cell classifier (k-means from scratch) ---",
        f"Total objects : {results['n_objects']}",
        f"Clusters      : {results['n_clusters']}",
        f"Inertia       : {results['inertia']}",
        "",
        "Cell type composition :",
        f"  Nucleus    : {s['nucleus']:3d}  ({pct['nucleus']}%)",
        f"  Cytoplasm  : {s['cytoplasm']:3d}  ({pct['cytoplasm']}%)",
        f"  Debris     : {s['debris']:3d}  ({pct['debris']}%)",
        f"  Uncertain  : {s['uncertain']:3d}  ({pct['uncertain']}%)",
        "",
        f"PCA variance explained : PC1={results['pca_variance'][0]:.1f}%"
        f"  PC2={results['pca_variance'][1]:.1f}%",
    ]
    return "\n".join(lines)