"""
pythagoras_analog.py
====================
Analog MLP 2 -> 30 -> 1 learns

    h = sqrt(a^2 + b^2)     (normalized to [0, 1])

Training images: solid Circle / Square / Triangle.

Geometry is defined in MATH coordinates (origin at image center, y UP).
The image raster is derived by flipping y, so the displayed image and the
reconstructed outline always share the same orientation (no upside-down
triangle).

Size directive:
    All shapes are inscribed in a circle of radius  R = frac * size/2,
    so they fit fully inside the canvas (no clipping) and can be made
    smaller/larger with one knob.

Balanced training data:
    Instead of forcing the shape area to be half the canvas, the training
    sampler draws an EQUAL number of white and black pixels
    (#black == #white in the training set, by construction).
"""

from __future__ import annotations
import numpy as np

SHAPES = ("circle", "square", "triangle")


# --------------------------------------------------------------------------- #
#  Geometry (math coords, y up, origin at center)
# --------------------------------------------------------------------------- #
def _shape_verts(shape: str, R: float) -> np.ndarray | None:
    """Polygon vertices in math coords for square/triangle inscribed in R."""
    if shape == "square":
        s = R / np.sqrt(2.0)          # half-side of square inscribed in R
        return np.array([[-s, -s], [s, -s], [s, s], [-s, s]])
    if shape == "triangle":
        # equilateral, circumradius R, apex UP (math coords)
        ang = np.deg2rad([90.0, 210.0, 330.0])
        return np.stack([R * np.cos(ang), R * np.sin(ang)], axis=1)
    return None  # circle


def _point_in_poly(xm: np.ndarray, ym: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon test."""
    inside = np.zeros(xm.shape, dtype=bool)
    n = len(verts)
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        cond = ((yi > ym) != (yj > ym)) & \
               (xm < (xj - xi) * (ym - yi) / (yj - yi + 1e-300) + xi)
        inside ^= cond
        j = i
    return inside


def make_shape_image(shape: str = "circle", size: int = 128,
                     frac: float = 0.7) -> np.ndarray:
    """Solid B/W shape image. frac = shape circumradius / (size/2)."""
    shape = shape.lower().strip()
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}")
    R = frac * size / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    xm = xx - cx          # math x
    ym = cy - yy          # math y (flip: image row 0 is TOP)
    img = np.zeros((size, size), dtype=np.uint8)
    if shape == "circle":
        img[np.sqrt(xm ** 2 + ym ** 2) <= R] = 255
    else:
        verts = _shape_verts(shape, R)
        img[_point_in_poly(xm, ym, verts)] = 255
    return img


def shape_balance_stats(img: np.ndarray) -> dict:
    n_white = int(np.count_nonzero(img > 127))
    n_black = int(img.size - n_white)
    return dict(
        n_white=n_white, n_black=n_black, total=int(img.size),
        ratio_white=n_white / img.size, balance_err=abs(n_white - n_black),
        size=img.shape[0],
    )


circle_balance_stats = shape_balance_stats  # legacy alias


def shape_boundary_radius(shape: str, theta: np.ndarray, size: int = 128,
                          frac: float = 0.7) -> np.ndarray:
    """Distance (px) from center to shape boundary at each angle (math coords)."""
    shape = shape.lower()
    R = frac * size / 2.0
    if shape == "circle":
        return np.full_like(theta, R, dtype=float)
    verts = _shape_verts(shape, R)
    return np.array([
        _ray_poly_hit(0.0, 0.0, float(np.cos(t)), float(np.sin(t)), verts)
        for t in theta])


def _ray_poly_hit(ox, oy, dx, dy, verts: np.ndarray) -> float:
    """Distance from (ox,oy) along unit dir (dx,dy) to nearest polygon edge."""
    n = len(verts)
    best = 1e9
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        den = dx * (y1 - y2) - dy * (x1 - x2)
        if abs(den) < 1e-12:
            continue
        t = ((x1 - ox) * (y1 - y2) - (y1 - oy) * (x1 - x2)) / den
        u = (dx * (y1 - oy) - dy * (x1 - ox)) / den
        if t > 1e-6 and 0.0 <= u <= 1.0:
            best = min(best, t)
    return float(best if best < 1e8 else 0.0)


# --------------------------------------------------------------------------- #
#  Balanced training data (#white samples == #black samples)
# --------------------------------------------------------------------------- #
def samples_from_shape_image(img: np.ndarray, n_extra: int = 1000,
                             seed: int = 0, max_from_image: int = 4000):
    """(a,b) -> h_norm with STRICTLY equal white/black pixel counts.

    n_per_class = min(#white, #black, max_from_image//2) drawn from each,
    so the pixel training set is balanced black==white by construction,
    independent of shape size. Plus n_extra random (a,b) fillers for
    full-range coverage of the Pythagorean map.
    """
    rng = np.random.default_rng(seed)
    size = img.shape[0]
    cy = cx = (size - 1) / 2.0
    scale = size / 2.0

    yw, xw = np.where(img > 127)
    yb, xb = np.where(img <= 127)
    n_per = int(min(len(xw), len(xb), max_from_image // 2))
    if n_per > 0:
        pw = rng.choice(len(xw), size=n_per, replace=False)
        pb = rng.choice(len(xb), size=n_per, replace=False)
        xs = np.concatenate([xw[pw], xb[pb]])
        ys = np.concatenate([yw[pw], yb[pb]])
    else:
        xs = np.concatenate([xw, xb])
        ys = np.concatenate([yw, yb])

    # math coords -> absolute legs
    a = np.clip(np.abs(xs - cx) / scale, 0, 1)
    b = np.clip(np.abs(cy - ys) / scale, 0, 1)
    a = np.concatenate([a, rng.random(n_extra)])
    b = np.concatenate([b, rng.random(n_extra)])

    h = np.sqrt(a ** 2 + b ** 2) / np.sqrt(2.0)
    X = np.stack([a, b], axis=1).astype(np.float64)
    y = h.astype(np.float64).reshape(-1, 1)
    idx = rng.permutation(len(X))
    return X[idx], y[idx], n_per


samples_from_circle_image = samples_from_shape_image  # legacy alias


# --------------------------------------------------------------------------- #
#  2 -> 30 -> 1 analog MLP
# --------------------------------------------------------------------------- #
class AnalogMLP2301:
    """Continuous (analog) MLP: 2-30-1, tanh hidden, sigmoid output."""

    def __init__(self, seed: int = 0, lr: float = 0.05):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W1 = rng.normal(0, np.sqrt(2 / 2), size=(30, 2))
        self.b1 = np.zeros(30)
        self.W2 = rng.normal(0, np.sqrt(2 / 30), size=(1, 30))
        self.b2 = np.zeros(1)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1.T + self.b1
        h = np.tanh(z1)
        z2 = h @ self.W2.T + self.b2
        y = self._sigmoid(z2)
        return y, h, z1

    def predict(self, X: np.ndarray) -> np.ndarray:
        y, _, _ = self.forward(X)
        return y

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 400,
              batch: int = 64, callback=None):
        rng = np.random.default_rng(0)
        n = len(X)
        hist = []
        for ep in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                xb = X[idx[s:s + batch]]
                yb = y[idx[s:s + batch]]
                yhat, h, z1 = self.forward(xb)
                dy = (yhat - yb) / len(xb)
                dz2 = dy * yhat * (1 - yhat)
                dW2 = dz2.T @ h
                db2 = dz2.sum(axis=0)
                dh = dz2 @ self.W2
                dz1 = dh * (1 - h ** 2)
                dW1 = dz1.T @ xb
                db1 = dz1.sum(axis=0)
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
            yhat, _, _ = self.forward(X)
            mse = float(np.mean((yhat - y) ** 2))
            hist.append(mse)
            if callback and (ep % 25 == 0 or ep == epochs - 1):
                callback(ep, mse)
        return hist

    def reconstruct_shape(self, shape: str = "circle", n_theta: int = 360,
                          size: int = 128, frac: float = 0.7):
        """Reconstruct outline via predicted hypot along boundary rays.

        All in math coords (y up) -- consistent with make_shape_image display.
        Returns x, y, x_true, y_true, R_hat, R, theta, rmse(normalized).
        """
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        R = shape_boundary_radius(shape, theta, size=size, frac=frac)
        scale = size / 2.0
        a = np.clip(R * np.abs(np.cos(theta)) / scale, 0, 1)
        b = np.clip(R * np.abs(np.sin(theta)) / scale, 0, 1)
        h_norm = self.predict(np.stack([a, b], axis=1)).ravel()
        R_hat = h_norm * np.sqrt(2.0) * scale
        x = R_hat * np.cos(theta)
        y = R_hat * np.sin(theta)
        x_true = R * np.cos(theta)
        y_true = R * np.sin(theta)
        rmse = float(np.sqrt(np.mean((R_hat - R) ** 2))) / scale
        return x, y, x_true, y_true, R_hat, R, theta, rmse

    def reconstruct_circle(self, n_theta: int = 360, r_ref: float = 1.0):
        """Legacy unit-circle reconstruction."""
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        a = np.abs(np.cos(theta))
        b = np.abs(np.sin(theta))
        h_norm = self.predict(np.stack([a, b], axis=1)).ravel()
        r_hat = h_norm * np.sqrt(2.0) * r_ref
        return r_hat * np.cos(theta), r_hat * np.sin(theta), r_hat, theta
