"""
shape_bmp.py
============
64 x 64 black/white bitmaps with EXACTLY balanced pixel counts, plus a
real-time random-pixel sampler for streaming training / testing.

    cir.bmp    circle
    sqr.bmp    square
    trngl.bmp  equilateral triangle (apex UP)

Balance guarantee
-----------------
Every bitmap has exactly  size*size/2  white and  size*size/2  black pixels
(2048 / 2048 for 64x64).  This is achieved by ranking all pixels by a scaled
distance field whose level sets are similar copies of the shape, then taking
the 2048 innermost.  Pixels that tie at the cut level (the single boundary
ring) are broken by a seeded random draw, so the exact balance never distorts
the geometry -- it only dithers the one-pixel edge.

Sampling protocol (real time)
-----------------------------
    draw random integer (x, y) in [0, 64)
    label = 1 if pixel is white else 0
    inputs = (x / 64.0, y / 64.0)

Coordinates: row 0 is the TOP of the displayed image, so (x, y) indexes the
bitmap exactly as it is drawn (imshow origin='upper'). No flips anywhere.
"""

from __future__ import annotations
import os
import numpy as np

SHAPES = ("circle", "square", "triangle")
BMP_NAMES = {"circle": "cir.bmp", "square": "sqr.bmp", "triangle": "trngl.bmp"}
DEFAULT_SIZE = 64


def assets_dir() -> str:
    """<repo>/sena2026/assets, created on demand."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(os.path.dirname(here), "assets")
    os.makedirs(d, exist_ok=True)
    return d


def bmp_path(shape: str) -> str:
    return os.path.join(assets_dir(), BMP_NAMES[shape])


# --------------------------------------------------------------------------- #
#  Scaled distance fields (math coords: origin at centre, y UP)
# --------------------------------------------------------------------------- #
def _shape_field(shape: str, size: int) -> np.ndarray:
    """Field whose level sets are similar copies of the shape, growing outward."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    xm = xx - c
    ym = c - yy                       # row 0 = TOP  ->  math y up
    shape = shape.lower().strip()
    if shape == "circle":
        return np.sqrt(xm ** 2 + ym ** 2)
    if shape == "square":
        return np.maximum(np.abs(xm), np.abs(ym))
    if shape == "triangle":
        # Isosceles, apex UP, base on the bottom edge.
        # Base = height = size  =>  area = size^2 / 2  =>  exactly half the
        # canvas, so the 2048-pixel level set is a COMPLETE triangle (sharp
        # apex, full base).  An equilateral of the same area needs side ~69
        # and overflows a 64x64 frame into a truncated trapezoid.
        c = size / 2.0
        xm = xx + 0.5 - c                 # pixel centres in (-c, c)
        ym = c - (yy + 0.5)               # top ≈ +c, bottom ≈ -c
        # Half-planes of apex (0,+c), BL (-c,-c), BR (+c,-c); boundary at 1.
        f_bottom = -ym / c
        f_right = (2.0 * xm + ym) / c
        f_left = (-2.0 * xm + ym) / c
        return np.maximum(np.maximum(f_left, f_right), f_bottom)
    raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")


def make_balanced_shape(shape: str, size: int = DEFAULT_SIZE, seed: int = 0):
    """Bitmap with exactly size*size/2 white pixels. Returns (img, n_dither)."""
    d = _shape_field(shape, size).ravel()
    n_white = (size * size) // 2
    rng = np.random.default_rng(seed)
    # primary key = distance, secondary key = random  -> symmetric tie handling
    order = np.lexsort((rng.random(d.size), d))
    img = np.zeros(size * size, dtype=np.uint8)
    img[order[:n_white]] = 255
    d_cut = d[order[n_white - 1]]
    n_dither = int(np.count_nonzero(d == d_cut))
    return img.reshape(size, size), n_dither


def balance_stats(img: np.ndarray) -> dict:
    n_white = int(np.count_nonzero(img > 127))
    n_black = int(img.size - n_white)
    return dict(n_white=n_white, n_black=n_black, total=int(img.size),
                balanced=(n_white == n_black), size=int(img.shape[0]))


# --------------------------------------------------------------------------- #
#  BMP file I/O
# --------------------------------------------------------------------------- #
def save_bmp(img: np.ndarray, path: str) -> str:
    from PIL import Image
    Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8)).convert("1") \
         .save(path, format="BMP")
    return path


def load_bmp(path: str) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def write_all_bmps(size: int = DEFAULT_SIZE, seed: int = 0) -> dict:
    """(Re)generate cir.bmp / sqr.bmp / trngl.bmp. Returns {shape: path}."""
    out = {}
    for sh in SHAPES:
        img, _ = make_balanced_shape(sh, size=size, seed=seed)
        out[sh] = save_bmp(img, bmp_path(sh))
    return out


def get_shape_image(shape: str, size: int = DEFAULT_SIZE,
                    seed: int = 0) -> np.ndarray:
    """Load the .bmp, generating (and saving) it the first time."""
    p = bmp_path(shape)
    if os.path.exists(p):
        img = load_bmp(p)
        if img.shape == (size, size):
            return img
    img, _ = make_balanced_shape(shape, size=size, seed=seed)
    save_bmp(img, p)
    return img


# --------------------------------------------------------------------------- #
#  Real-time pixel sampler
# --------------------------------------------------------------------------- #
class PixelStream:
    """Endless stream of random pixel probes: (x/64, y/64) -> {0,1}."""

    def __init__(self, img: np.ndarray, seed: int = 0):
        self.img = img
        self.size = int(img.shape[0])
        self.rng = np.random.default_rng(seed)
        self.n_drawn = 0

    def next_batch(self, n: int):
        """Returns X (n,2) normalized, labels (n,) in {0,1}, and raw x, y."""
        s = self.size
        x = self.rng.integers(0, s, size=n)
        y = self.rng.integers(0, s, size=n)
        lab = (self.img[y, x] > 127).astype(np.int64)
        X = np.stack([x / float(s), y / float(s)], axis=1)
        self.n_drawn += n
        return X, lab, x, y


def grid_inputs(size: int = DEFAULT_SIZE) -> np.ndarray:
    """All pixel centres as normalized (x/size, y/size); row-major (y outer)."""
    yy, xx = np.mgrid[0:size, 0:size]
    return np.stack([xx.ravel() / float(size),
                     yy.ravel() / float(size)], axis=1)


if __name__ == "__main__":
    paths = write_all_bmps()
    for sh in SHAPES:
        img = load_bmp(paths[sh])
        st = balance_stats(img)
        print(f"{sh:9s} {os.path.basename(paths[sh]):10s} "
              f"{st['size']}x{st['size']}  "
              f"white={st['n_white']}  black={st['n_black']}  "
              f"balanced={st['balanced']}")
    img = load_bmp(paths["triangle"])
    print("\ntriangle top rows white count (apex should be UP):",
          [int(np.count_nonzero(img[r] > 127)) for r in (2, 16, 32, 48, 62)])
    st = PixelStream(load_bmp(paths["circle"]), seed=1)
    X, lab, x, y = st.next_batch(8)
    for i in range(8):
        print(f"  (x={x[i]:2d}, y={y[i]:2d}) -> "
              f"({X[i,0]:.4f}, {X[i,1]:.4f})  label={lab[i]}")
