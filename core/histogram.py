import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    return (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)


def compute_histogram(pixels: np.ndarray, bins: int = 256) -> np.ndarray:
    hist = np.zeros(bins, dtype=np.int64)
    scale = bins / 256
    for v in pixels.ravel():
        b = min(int(v * scale), bins - 1)
        hist[b] += 1
    return hist


def compute_cdf(hist: np.ndarray) -> np.ndarray:
    cdf = np.zeros_like(hist)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf


def stretch_contrast(pixels: np.ndarray) -> np.ndarray:
    mn, mx = int(pixels.min()), int(pixels.max())
    if mx == mn:
        return pixels.copy()
    out = np.zeros_like(pixels)
    h, w = pixels.shape
    for y in range(h):
        for x in range(w):
            out[y, x] = round((int(pixels[y, x]) - mn) / (mx - mn) * 255)
    return out


def equalize_histogram(pixels: np.ndarray) -> np.ndarray:
    N = pixels.size
    L = 256
    hist = compute_histogram(pixels, L)
    cdf  = compute_cdf(hist)
    cdf_min = int(next(v for v in cdf if v > 0))
    lut = np.zeros(L, dtype=np.uint8)
    for i in range(L):
        lut[i] = round(((int(cdf[i]) - cdf_min) / (N - cdf_min)) * (L - 1))
    out = np.zeros_like(pixels)
    h, w = pixels.shape
    for y in range(h):
        for x in range(w):
            out[y, x] = lut[pixels[y, x]]
    return out


def threshold_binary(pixels: np.ndarray, seuil: int) -> np.ndarray:
    out = np.zeros_like(pixels)
    h, w = pixels.shape
    for y in range(h):
        for x in range(w):
            out[y, x] = 255 if pixels[y, x] >= seuil else 0
    return out


def image_stats(pixels: np.ndarray) -> dict:
    flat    = pixels.ravel().astype(np.float64)
    mean    = float(np.mean(flat))
    std     = float(np.std(flat))
    mn      = int(pixels.min())
    mx      = int(pixels.max())
    hist    = compute_histogram(pixels, 256)
    total   = pixels.size
    entropy = 0.0
    for c in hist:
        if c > 0:
            p = c / total
            entropy -= p * np.log2(p)
    return {"min": mn, "max": mx, "mean": round(mean, 1),
            "std": round(std, 1), "entropy": round(entropy, 2)}