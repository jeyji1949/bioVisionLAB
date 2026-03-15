import numpy as np


def _pad(img: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(img, ((pad, pad), (pad, pad)), mode="edge")


def _gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    center = ksize // 2
    k = np.zeros((ksize, ksize), dtype=np.float64)
    for y in range(ksize):
        for x in range(ksize):
            dx, dy = x - center, y - center
            k[y, x] = np.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
    return k / k.sum()


def convolve2d(pixels: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w   = pixels.shape
    kh, kw = kernel.shape
    pad    = kh // 2
    padded = _pad(pixels, pad)
    out    = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            out[y, x] = np.sum(padded[y:y+kh, x:x+kw] * kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def mean_filter(pixels: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    return convolve2d(pixels, k)


def gaussian_filter(pixels: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    k = _gaussian_kernel(ksize, sigma)
    return convolve2d(pixels, k)


def median_filter(pixels: np.ndarray, ksize: int = 3) -> np.ndarray:
    h, w   = pixels.shape
    pad    = ksize // 2
    padded = _pad(pixels, pad)
    out    = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            window = padded[y:y+ksize, x:x+ksize].ravel()
            out[y, x] = np.sort(window)[len(window) // 2]
    return out


def laplacian_filter(pixels: np.ndarray) -> np.ndarray:
    k = np.array([[0, -1,  0],
                  [-1,  4, -1],
                  [0, -1,  0]], dtype=np.float64)
    h, w   = pixels.shape
    padded = _pad(pixels, 1)
    out    = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            out[y, x] = abs(np.sum(padded[y:y+3, x:x+3] * k))
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper(pixels: np.ndarray, amount: float = 0.05) -> np.ndarray:
    out = pixels.copy()
    n   = int(pixels.size * amount)
    for _ in range(n):
        y = np.random.randint(0, pixels.shape[0])
        x = np.random.randint(0, pixels.shape[1])
        out[y, x] = 255 if np.random.random() < 0.5 else 0
    return out


def add_gaussian_noise(pixels: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    out = np.zeros_like(pixels, dtype=np.float64)
    h, w = pixels.shape
    for y in range(h):
        for x in range(w):
            u1 = max(np.random.random(), 1e-10)
            u2 = np.random.random()
            z  = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            out[y, x] = pixels[y, x] + z * sigma
    return np.clip(out, 0, 255).astype(np.uint8)


NOISE_FILTER_MAP = {
    "Poivre & Sel": {"recommended": "Médian",   "others": ["Moyenneur", "Gaussien"]},
    "Gaussien":     {"recommended": "Gaussien", "others": ["Moyenneur", "Médian"]},
    "Aucun bruit":  {"recommended": None,       "others": ["Médian", "Gaussien", "Moyenneur", "Laplacien"]},
}