"""Point-query wrapper for a randomized PhysiCell MISO instance.

The optimizer interacts through noisy local queries, while full fields and tumor
metadata remain internal to the benchmark for training labels and post-hoc
scoring.
"""
import json, os, glob
import numpy as np
import scipy.io as sio

SOURCE_NAMES = ["oxygen", "chemoattractant", "therapeutic"]  # 0:min  1:max  2:max at tumor


def _load_field(inst_dir):
    """Return (xs, ys, fields[S, ny, nx]) from the final microenvironment .mat of an instance."""
    mats = sorted(glob.glob(os.path.join(inst_dir, "out", "*microenvironment0.mat")))
    # prefer 'final', else last output
    final = [m for m in mats if "final" in os.path.basename(m)]
    path = final[0] if final else mats[-1]
    A = sio.loadmat(path)["multiscale_microenvironment"]  # (7, Nvox): x,y,z,vol,ox,chemo,ther
    x, y = A[0], A[1]
    xs, ys = np.unique(x), np.unique(y)
    nx, ny = len(xs), len(ys)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    fields = np.full((3, ny, nx), np.nan, dtype=np.float64)
    for s in range(3):
        vals = A[4 + s]
        for k in range(A.shape[1]):
            fields[s, yi[y[k]], xi[x[k]]] = vals[k]
    return xs, ys, fields


def _bilinear(xs, ys, grid, x, y):
    """Bilinear interpolation of a 2D grid (ny,nx) at point (x,y), clamped to domain."""
    x = float(np.clip(x, xs[0], xs[-1]))
    y = float(np.clip(y, ys[0], ys[-1]))
    ix = np.searchsorted(xs, x) - 1
    iy = np.searchsorted(ys, y) - 1
    ix = int(np.clip(ix, 0, len(xs) - 2))
    iy = int(np.clip(iy, 0, len(ys) - 2))
    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    g = grid
    return ((1 - tx) * (1 - ty) * g[iy, ix] + tx * (1 - ty) * g[iy, ix + 1]
            + (1 - tx) * ty * g[iy + 1, ix] + tx * ty * g[iy + 1, ix + 1])


class PhysiCellMISOOracle:
    """One hidden instance. Public API exposes ONLY noisy point queries + post-hoc evaluation."""

    def __init__(self, inst_dir, query_seed=0, rotate=False, source_perm=None):
        self._inst_dir = inst_dir
        self._xs, self._ys, self._fields = _load_field(inst_dir)   # PRIVATE
        with open(os.path.join(inst_dir, "theta.json")) as fh:
            theta = json.load(fh)
        self._x_star = np.array(theta["x_star"], dtype=np.float64)  # PRIVATE (hidden)
        self._tumor_radius = float(theta["tumor_radius"])
        self.domain = (float(self._xs[0]), float(self._xs[-1]),
                       float(self._ys[0]), float(self._ys[-1]))
        self.n_sources = 3

        # per-instance measurement model (varying reliability across sources) — derived from a seed,
        # NOT exposed. Field std used to scale noise so SNR is meaningful per source.
        rng = np.random.default_rng(query_seed)
        fstd = np.array([np.nanstd(self._fields[s]) + 1e-9 for s in range(3)])
        self._noise_sigma = fstd * rng.uniform(0.05, 0.30, size=3)   # source-specific noise
        self._bias = fstd * rng.uniform(-0.05, 0.05, size=3)
        self._dropout = rng.uniform(0.0, 0.10, size=3)              # prob of a garbage reading
        self.source_cost = rng.uniform(1.0, 5.0, size=3)           # public: cost is allowed to be known

        # optional test-time distribution shift wrappers (do NOT leak x_star)
        self._rotate = rotate
        if rotate:
            ang = rng.uniform(0, 2 * np.pi)
            self._R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        self._source_perm = np.array(source_perm) if source_perm is not None else np.arange(3)
        self._rng = rng
        self.n_queries = 0

    # ---------------- public API ----------------
    def query(self, x, source, t=None):
        """Noisy local biomarker value at location x for given source. Point-query only."""
        self.n_queries += 1
        src = int(self._source_perm[int(source)])   # external source id -> internal channel
        p = np.array([x[0], x[1]], dtype=np.float64)
        if self._rotate:                              # rotate query into the hidden field frame
            p = self._R @ p
        val = _bilinear(self._xs, self._ys, self._fields[src], p[0], p[1])
        if self._rng.uniform() < self._dropout[src]:
            val = val + self._rng.normal(0, 5 * self._noise_sigma[src])  # occasional garbage
        return float(val + self._bias[src] + self._rng.normal(0, self._noise_sigma[src]))

    def evaluate_recommendation(self, x_hat):
        """Post-hoc scoring with the hidden x_star (only after the optimizer stops)."""
        x_hat = np.array(x_hat, dtype=np.float64)
        if self._rotate:
            x_hat = self._R @ x_hat                  # map recommendation back to hidden frame
        d = float(np.linalg.norm(x_hat - self._x_star))
        return dict(distance=d, detected=bool(d <= self._tumor_radius),
                    tumor_radius=self._tumor_radius, n_queries=self.n_queries)

    # ---------------- privileged (training/harness ONLY) ----------------
    def privileged_label(self):
        """x_star in the EXTERNAL (possibly rotated) query frame — for the pretraining loss only."""
        xs = self._x_star
        if self._rotate:
            xs = self._R.T @ xs
        return xs.copy()

    def privileged_full_field(self):
        """Full field — ONLY for the explicitly-labeled 'full-field privileged baseline'."""
        return self._xs.copy(), self._ys.copy(), self._fields.copy()


def list_instances(split_dir):
    mf = os.path.join(split_dir, "manifest.txt")
    return [l.strip() for l in open(mf) if l.strip()]
