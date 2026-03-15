import numpy as np


def _threshold(pixels: np.ndarray, seuil: int) -> np.ndarray:
    out = np.zeros_like(pixels)
    h, w = pixels.shape
    for y in range(h):
        for x in range(w):
            out[y, x] = 255 if pixels[y, x] >= seuil else 0
    return out


def erode(pixels: np.ndarray) -> np.ndarray:
    h, w = pixels.shape
    out  = np.zeros_like(pixels)
    for y in range(h):
        for x in range(w):
            mn = 255
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    py = max(0, min(h-1, y+ky))
                    px = max(0, min(w-1, x+kx))
                    if pixels[py, px] < mn:
                        mn = pixels[py, px]
            out[y, x] = mn
    return out


def dilate(pixels: np.ndarray) -> np.ndarray:
    h, w = pixels.shape
    out  = np.zeros_like(pixels)
    for y in range(h):
        for x in range(w):
            mx = 0
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    py = max(0, min(h-1, y+ky))
                    px = max(0, min(w-1, x+kx))
                    if pixels[py, px] > mx:
                        mx = pixels[py, px]
            out[y, x] = mx
    return out


def _iter(fn, pixels, n):
    for _ in range(n):
        pixels = fn(pixels)
    return pixels


def opening(pixels: np.ndarray, iterations: int = 1) -> np.ndarray:
    return _iter(dilate, _iter(erode, pixels, iterations), iterations)


def closing(pixels: np.ndarray, iterations: int = 1) -> np.ndarray:
    return _iter(erode, _iter(dilate, pixels, iterations), iterations)


def top_hat(pixels: np.ndarray, iterations: int = 1) -> np.ndarray:
    opened = opening(pixels, iterations)
    diff   = pixels.astype(np.int16) - opened.astype(np.int16)
    return np.clip(diff, 0, 255).astype(np.uint8)


def black_hat(pixels: np.ndarray, iterations: int = 1) -> np.ndarray:
    closed = closing(pixels, iterations)
    diff   = closed.astype(np.int16) - pixels.astype(np.int16)
    return np.clip(diff, 0, 255).astype(np.uint8)


def morpho_gradient(pixels: np.ndarray) -> np.ndarray:
    d = dilate(pixels).astype(np.int16)
    e = erode(pixels).astype(np.int16)
    return np.clip(d - e, 0, 255).astype(np.uint8)


def run_morpho(pixels: np.ndarray, op: str, seuil: int, iterations: int) -> np.ndarray:
    binary = _threshold(pixels, seuil)
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