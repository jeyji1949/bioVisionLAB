"""
core/glcm.py
============
Module D — Haralick texture analysis via GLCM, 100% NumPy, zero scikit-learn.

What it does (in plain terms)
------------------------------
1. Takes a grayscale image (original or M1 output)
2. Quantises grey levels to 8 bins  (256 → 8 makes computation tractable)
3. Builds a Grey-Level Co-occurrence Matrix (GLCM) for 4 directions
4. Extracts 6 Haralick texture features from each GLCM
5. Averages over the 4 directions  (→ rotation-invariant descriptor)
6. Returns a human-readable texture profile for the report

How it connects to code
--------------------------------------
  Call GLCMAnalyzer().run(state.original_pixels) in report_panel.refresh_image().
  The result dict plugs straight into a new _md_card() in report_panel.py.

Background — why GLCM matters for microscopy
---------------------------------------------
  Nuclei have fine, uniform texture  → high energy, high homogeneity
  Cytoplasm has variable texture     → higher contrast, higher entropy
  Background is nearly uniform       → very high energy, near-zero contrast

  GLCM captures this by counting how often grey-level pairs (i, j)
  appear as neighbours in the image — a spatial statistics approach
  that goes well beyond simple histogram features.

Usage
-----
  from core.glcm import GLCMAnalyzer

  ana     = GLCMAnalyzer(levels=8, distances=[1, 2])
  results = ana.run(grayscale_image)

  results = {
      "glcm":          (L, L, n_dist, n_angle) array — raw matrices
      "features":      dict  — averaged Haralick features
      "features_per_angle": list of dicts — one per direction
      "profile":       str   — e.g. "Texture fine et uniforme"
      "profile_detail": str  — biological interpretation
      "levels":        int   — number of grey levels used
  }

References
----------
  Haralick, R.M. et al. (1973) "Textural Features for Image Classification"
  IEEE Transactions on Systems, Man, and Cybernetics, 3(6), 610–621.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Grey-level quantisation
#     256 grey levels → L levels  (reduces GLCM size from 256² to L²)
# ─────────────────────────────────────────────────────────────────────────────

def quantise(pixels: np.ndarray, levels: int = 8) -> np.ndarray:
    """
    Map pixel intensities [0, 255] → [0, levels-1].

    Formula:  q = floor(pixel / 256 * levels)   clamped to [0, levels-1]

    Using 8 levels is standard for microscopy GLCM — it captures
    meaningful texture differences while keeping computation fast.

    Parameters
    ----------
    pixels : (H, W) uint8 array
    levels : number of grey levels in the output (default 8)

    Returns
    -------
    (H, W) uint8 array with values in [0, levels-1]
    """
    q = np.floor(pixels.astype(np.float64) / 256.0 * levels).astype(np.int32)
    return np.clip(q, 0, levels - 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GLCM construction
#     Count how often grey-level pairs (i, j) appear as spatial neighbours
# ─────────────────────────────────────────────────────────────────────────────

# The 4 standard directions used in Haralick's original paper
# Each is a (dy, dx) offset from a pixel to its neighbour
ANGLES = [
    (0,  1),   # 0°   — horizontal neighbours
    (1,  1),   # 45°  — diagonal ↘
    (1,  0),   # 90°  — vertical neighbours
    (1, -1),   # 135° — diagonal ↙
]
ANGLE_NAMES = ["0°", "45°", "90°", "135°"]


def build_glcm(q_img: np.ndarray, levels: int,
               distance: int = 1) -> np.ndarray:
    """
    Build one GLCM for all 4 directions at a given pixel distance.

    The GLCM G[i, j, angle] counts how many times grey level i
    is followed by grey level j at the given distance and angle.
    Each matrix is then symmetrised (G = G + G^T) and normalised
    to sum to 1 — so it becomes a joint probability distribution P(i,j).

    Parameters
    ----------
    q_img    : (H, W) quantised image from quantise()
    levels   : number of grey levels (must match quantise() call)
    distance : pixel offset distance (1 = immediate neighbours)

    Returns
    -------
    glcm : (levels, levels, 4) float64 array — normalised, symmetric
           glcm[:, :, a] is the GLCM for direction a
    """
    H, W = q_img.shape
    glcm = np.zeros((levels, levels, len(ANGLES)), dtype=np.float64)

    for a_idx, (dy, dx) in enumerate(ANGLES):
        dy *= distance
        dx *= distance

        # Iterate over all valid pixel pairs
        for y in range(H):
            for x in range(W):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    i = q_img[y,  x]
                    j = q_img[ny, nx]
                    glcm[i, j, a_idx] += 1.0

        # Symmetrise: add transpose so (i→j) and (j→i) are both counted
        glcm[:, :, a_idx] += glcm[:, :, a_idx].T

        # Normalise to probability distribution
        s = glcm[:, :, a_idx].sum()
        if s > 0:
            glcm[:, :, a_idx] /= s

    return glcm


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Haralick feature extraction
#     6 standard descriptors, each capturing a different texture property
# ─────────────────────────────────────────────────────────────────────────────

def _features_from_matrix(P: np.ndarray) -> dict:
    """
    Compute the 6 Haralick features from a single normalised GLCM matrix P.

    All features are defined over the joint probability distribution P(i,j)
    where i and j are grey-level indices.

    Features
    --------
    1. contrast      Σ (i-j)² · P(i,j)
       High when neighbouring pixels differ a lot (rough texture).
       Nucleus: low. Background: very low. Irregular structures: high.

    2. energy        Σ P(i,j)²   [also called Angular Second Moment]
       High when the image is very uniform (few grey-level transitions).
       Perfectly uniform → 1.0. Random texture → close to 0.

    3. homogeneity   Σ P(i,j) / (1 + |i-j|)
       High when grey levels of neighbouring pixels are similar.
       Related to contrast but less sensitive to extreme differences.

    4. entropy       -Σ P(i,j) · log₂(P(i,j)+ε)
       Measures unpredictability of grey-level pairs.
       Uniform image → 0. Complex texture → high value.

    5. correlation   Σ [(i-μᵢ)(j-μⱼ)·P(i,j)] / (σᵢ·σⱼ)
       Measures linear dependency between grey levels of pixel pairs.
       Regular periodic patterns → high. Random → near 0.

    6. dissimilarity  Σ |i-j| · P(i,j)
       Like contrast but linear (not squared) — less sensitive to outliers.
    """
    L     = P.shape[0]
    idx   = np.arange(L, dtype=np.float64)

    # Pre-build index grids for vectorised computation
    I, J  = np.meshgrid(idx, idx, indexing="ij")   # I[i,j]=i, J[i,j]=j

    # 1. Contrast
    contrast = float(np.sum((I - J) ** 2 * P))

    # 2. Energy (Angular Second Moment)
    energy = float(np.sum(P ** 2))

    # 3. Homogeneity
    homogeneity = float(np.sum(P / (1.0 + np.abs(I - J))))

    # 4. Entropy
    eps     = 1e-12   # avoid log(0)
    entropy = float(-np.sum(P * np.log2(P + eps)))

    # 5. Correlation
    mu_i = float(np.sum(I * P))
    mu_j = float(np.sum(J * P))
    si   = float(np.sqrt(np.sum((I - mu_i) ** 2 * P)))
    sj   = float(np.sqrt(np.sum((J - mu_j) ** 2 * P)))
    if si > 1e-10 and sj > 1e-10:
        correlation = float(np.sum((I - mu_i) * (J - mu_j) * P) / (si * sj))
    else:
        correlation = 1.0   # uniform image → perfectly correlated

    # 6. Dissimilarity
    dissimilarity = float(np.sum(np.abs(I - J) * P))

    return {
        "contrast":      round(contrast,      4),
        "energy":        round(energy,         4),
        "homogeneity":   round(homogeneity,    4),
        "entropy":       round(entropy,        4),
        "correlation":   round(correlation,    4),
        "dissimilarity": round(dissimilarity,  4),
    }


def extract_haralick(glcm: np.ndarray) -> tuple:
    """
    Compute Haralick features for each direction, then average.

    Averaging over 4 directions gives a rotation-invariant descriptor —
    the texture profile is the same regardless of how the tissue is oriented
    under the microscope.

    Parameters
    ----------
    glcm : (L, L, 4) array from build_glcm()

    Returns
    -------
    averaged : dict  — mean feature values over 4 directions
    per_angle: list of dicts — one per direction (for detailed display)
    """
    per_angle = []
    for a in range(glcm.shape[2]):
        per_angle.append(_features_from_matrix(glcm[:, :, a]))

    # Average each feature over the 4 directions
    keys     = per_angle[0].keys()
    averaged = {}
    for k in keys:
        vals        = [pa[k] for pa in per_angle]
        averaged[k] = round(float(np.mean(vals)), 4)

    return averaged, per_angle


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Biological texture profile
#     Maps feature values → human-readable label + interpretation
# ─────────────────────────────────────────────────────────────────────────────

def texture_profile(features: dict) -> tuple:
    """
    Assign a biological texture profile based on the Haralick features.

    Decision rules (empirically calibrated for 8-level GLCM):
    ----------------------------------------------------------
    Fine & uniform   : energy > 0.15  and  contrast < 0.10
    Moderately fine  : energy > 0.08  and  contrast < 0.25
    Moderately rough : contrast between 0.25 and 0.60
    Rough / complex  : contrast > 0.60  or  entropy > 3.5

    Returns
    -------
    profile        : short label (e.g. "Texture fine et uniforme")
    profile_detail : biological interpretation sentence
    """
    contrast    = features["contrast"]
    energy      = features["energy"]
    entropy     = features["entropy"]
    homogeneity = features["homogeneity"]

    if energy > 0.15 and contrast < 0.10:
        profile = "Texture fine et uniforme"
        detail  = ("Distribution homogène des niveaux de gris — caractéristique "
                   "du fond intercellulaire ou de noyaux bien définis à chromatine "
                   "condensée.")
    elif energy > 0.08 and contrast < 0.25:
        profile = "Texture modérément fine"
        detail  = ("Transitions douces entre niveaux voisins — typique du "
                   "cytoplasme ou de tissus à structure régulière.")
    elif contrast < 0.60:
        profile = "Texture modérément complexe"
        detail  = ("Hétérogénéité spatiale notable — peut indiquer un mélange "
                   "de types cellulaires ou des artefacts de coloration.")
    else:
        profile = "Texture rugueuse / complexe"
        detail  = ("Fortes variations locales de gris — associées à des "
                   "granules cytoplasmiques, débris cellulaires ou bruit résiduel.")

    if homogeneity > 0.90:
        detail += " Homogénéité élevée confirme la régularité structurelle."

    return profile, detail


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Multi-distance GLCM
#     Running at distances 1 and 2 reveals multi-scale texture structure
# ─────────────────────────────────────────────────────────────────────────────

def _avg_glcm(glcms: list) -> np.ndarray:
    """Average a list of GLCM arrays (one per distance)."""
    result = np.zeros_like(glcms[0])
    for g in glcms:
        result += g
    return result / len(glcms)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  High-level wrapper — the object you import from report_panel
# ─────────────────────────────────────────────────────────────────────────────

class GLCMAnalyzer:
    """
    Complete Module D pipeline in one object.

    Usage
    -----
      ana     = GLCMAnalyzer(levels=8, distances=[1, 2])
      results = ana.run(grayscale_image)

    Results dict
    ------------
      glcm            : (L, L, 4) averaged GLCM array
      features        : dict  — 6 Haralick features (averaged over distances & angles)
      features_per_angle : list of dicts — per-direction breakdown (for debug/display)
      profile         : str   — texture profile label
      profile_detail  : str   — biological interpretation
      levels          : int   — grey levels used
      distances       : list  — distances used
      angle_names     : list  — ["0°", "45°", "90°", "135°"]
    """

    def __init__(self, levels: int = 8, distances: list = None):
        self.levels    = levels
        self.distances = distances if distances is not None else [1, 2]

    def run(self, grayscale: np.ndarray) -> dict:
        """
        Run the full Module D pipeline.

        Parameters
        ----------
        grayscale : (H, W) uint8 — any grayscale image
                    Use state.original_pixels or state.m1_result

        Returns
        -------
        dict with all results (see class docstring)
        """
        # ── Step 1: quantise ──────────────────────────────────────────────────
        q_img = quantise(grayscale, self.levels)

        # ── Step 2: build GLCMs at each distance ──────────────────────────────
        glcms = []
        for d in self.distances:
            glcms.append(build_glcm(q_img, self.levels, distance=d))

        # ── Step 3: average across distances ─────────────────────────────────
        glcm_avg = _avg_glcm(glcms)

        # ── Step 4: extract Haralick features ─────────────────────────────────
        features, per_angle = extract_haralick(glcm_avg)

        # ── Step 5: biological profile ────────────────────────────────────────
        profile, detail = texture_profile(features)

        return {
            "glcm":               glcm_avg,
            "features":           features,
            "features_per_angle": per_angle,
            "profile":            profile,
            "profile_detail":     detail,
            "levels":             self.levels,
            "distances":          self.distances,
            "angle_names":        ANGLE_NAMES,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Context block for M6 AI chat (drop into ai_panel._build_context)
# ─────────────────────────────────────────────────────────────────────────────

def glcm_context_block(results: dict) -> str:
    """
    Format Module D results as text for the AI chat context.

    Usage in ai_panel.py
    --------------------
      from core.glcm import glcm_context_block
      ...
      if hasattr(state, "md_results") and state.md_results:
          lines.append(glcm_context_block(state.md_results))
    """
    if not results:
        return "--- Module D — GLCM texture ---\nNon exécuté.\n"

    f = results["features"]
    lines = [
        "--- Module D — GLCM Texture Analysis (Haralick, from scratch) ---",
        f"Niveaux de gris  : {results['levels']}",
        f"Distances        : {results['distances']}",
        f"Profil texture   : {results['profile']}",
        "",
        "Caractéristiques Haralick (moyennées sur 4 directions) :",
        f"  Contraste      : {f['contrast']}",
        f"  Énergie        : {f['energy']}",
        f"  Homogénéité    : {f['homogeneity']}",
        f"  Entropie       : {f['entropy']}",
        f"  Corrélation    : {f['correlation']}",
        f"  Dissimilarité  : {f['dissimilarity']}",
        "",
        f"Interprétation : {results['profile_detail']}",
    ]
    return "\n".join(lines)