import numpy as np


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def fft1d(re: np.ndarray, im: np.ndarray, invert: bool = False) -> None:
    n = len(re)
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
    length = 2
    while length <= n:
        sign      = -1.0 if not invert else 1.0
        ang       = sign * 2.0 * np.pi / length
        wr, wi    = np.cos(ang), np.sin(ang)
        for i in range(0, n, length):
            cur_r, cur_i = 1.0, 0.0
            for k in range(length // 2):
                u_r = re[i + k]
                u_i = im[i + k]
                v_r = re[i + k + length//2] * cur_r - im[i + k + length//2] * cur_i
                v_i = re[i + k + length//2] * cur_i + im[i + k + length//2] * cur_r
                re[i + k]             = u_r + v_r
                im[i + k]             = u_i + v_i
                re[i + k + length//2] = u_r - v_r
                im[i + k + length//2] = u_i - v_i
                new_r  = cur_r * wr - cur_i * wi
                cur_i  = cur_r * wi + cur_i * wr
                cur_r  = new_r
        length <<= 1
    if invert:
        re /= n
        im /= n


def fft2d(pixels: np.ndarray):
    h, w = pixels.shape
    W = _next_pow2(w)
    H = _next_pow2(h)
    re = np.zeros((H, W), dtype=np.float64)
    im = np.zeros((H, W), dtype=np.float64)
    re[:h, :w] = pixels.astype(np.float64)
    for y in range(H):
        row_r = re[y].copy(); row_i = im[y].copy()
        fft1d(row_r, row_i, invert=False)
        re[y] = row_r; im[y] = row_i
    for x in range(W):
        col_r = re[:, x].copy(); col_i = im[:, x].copy()
        fft1d(col_r, col_i, invert=False)
        re[:, x] = col_r; im[:, x] = col_i
    return re, im, W, H


def ifft2d(re: np.ndarray, im: np.ndarray, W: int, H: int) -> None:
    for y in range(H):
        row_r = re[y].copy(); row_i = im[y].copy()
        fft1d(row_r, row_i, invert=True)
        re[y] = row_r; im[y] = row_i
    for x in range(W):
        col_r = re[:, x].copy(); col_i = im[:, x].copy()
        fft1d(col_r, col_i, invert=True)
        re[:, x] = col_r; im[:, x] = col_i


def fft_shift(re: np.ndarray, im: np.ndarray, W: int, H: int):
    hw, hh = W // 2, H // 2
    shift_r = np.zeros_like(re)
    shift_i = np.zeros_like(im)
    for y in range(H):
        for x in range(W):
            nx = (x + hw) % W
            ny = (y + hh) % H
            shift_r[ny, nx] = re[y, x]
            shift_i[ny, nx] = im[y, x]
    return shift_r, shift_i


def apply_frequency_mask(re, im, W, H, filter_type, radius):
    cx, cy = W // 2, H // 2
    for y in range(H):
        for x in range(W):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if filter_type == "passe-bas":
                mask = 1.0 if dist <= radius else 0.0
            elif filter_type == "passe-haut":
                mask = 0.0 if dist <= radius else 1.0
            else:
                mask = 1.0 if radius <= dist <= radius * 1.6 else 0.0
            re[y, x] *= mask
            im[y, x] *= mask


def spectrum_image(re, im, W, H, orig_h, orig_w):
    logs = np.zeros((orig_h, orig_w), dtype=np.float64)
    for y in range(orig_h):
        for x in range(orig_w):
            mag = np.sqrt(re[y, x]**2 + im[y, x]**2)
            logs[y, x] = np.log1p(mag)
    mx = logs.max()
    if mx > 0:
        logs = logs / mx * 255
    return logs.astype(np.uint8)