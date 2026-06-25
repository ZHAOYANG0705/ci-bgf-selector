"""Validate PhysiCell MISO instances: each channel's extremum must sit near the hidden tumor center,
and channels must be correlated-but-not-redundant (different shapes)."""
import sys, json, os, numpy as np
from pathlib import Path
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "miso").is_dir())
sys.path.insert(0, str(REPO_ROOT))
from miso.physicell_oracle import _load_field, list_instances

EXTREMA = {0: "min(oxygen)", 1: "max(chemo)", 2: "max(ther)"}

def extremum_loc(xs, ys, grid, want_max):
    g = grid.copy()
    idx = np.nanargmax(g) if want_max else np.nanargmin(g)
    iy, ix = np.unravel_index(idx, g.shape)
    return np.array([xs[ix], ys[iy]])


def validate_packed_fields(split_dir):
    packed = os.path.join(split_dir, "fields.npz")
    d = np.load(packed)
    required = ["fields", "masks", "x_star", "tumor_radius", "xs", "ys"]
    missing = [k for k in required if k not in d.files]
    if missing:
        raise ValueError(f"{packed} is missing keys: {missing}")

    fields = d["fields"]
    masks = d["masks"]
    cores = d["cores"] if "cores" in d.files else None
    rims = d["rims"] if "rims" in d.files else None
    x_star = d["x_star"]
    radius = d["tumor_radius"]
    xs, ys = d["xs"], d["ys"]

    if fields.ndim != 4 or fields.shape[1] != 3:
        raise ValueError(f"expected fields shape (n, 3, ny, nx), got {fields.shape}")
    n, _, ny, nx = fields.shape
    if masks.shape != (n, ny, nx):
        raise ValueError("mask shape does not match fields")
    if cores is not None and cores.shape != (n, ny, nx):
        raise ValueError("core shape does not match fields")
    if rims is not None and rims.shape != (n, ny, nx):
        raise ValueError("rim shape does not match fields")
    if x_star.shape != (n, 2) or radius.shape != (n,):
        raise ValueError("x_star or tumor_radius has an unexpected shape")
    if len(xs) != nx or len(ys) != ny:
        raise ValueError("grid axes do not match field dimensions")
    if np.isnan(fields).any():
        raise ValueError("fields contain NaNs")
    if np.any(masks.reshape(n, -1).sum(axis=1) == 0):
        raise ValueError("at least one instance has an empty tumor mask")

    locs = []
    for F in fields:
        locs.append([
            extremum_loc(xs, ys, F[0], False),
            extremum_loc(xs, ys, F[1], True),
            extremum_loc(xs, ys, F[2], True),
        ])
    locs = np.array(locs)
    dists = np.linalg.norm(locs - x_star[:, None, :], axis=2)
    print(f"validated packed fields: {packed}")
    print(f"instances={n} fields={fields.shape} grid={ny}x{nx}")
    msg = f"mask_frac={masks.mean():.4f}"
    if cores is not None and rims is not None:
        msg += f" core_frac={cores.mean():.4f} rim_frac={rims.mean():.4f}"
    print(msg)
    print(
        "mean extremum->x_star distance (micron): "
        f"ox_min={dists[:, 0].mean():.1f} "
        f"chemo_max={dists[:, 1].mean():.1f} "
        f"ther_max={dists[:, 2].mean():.1f}"
    )


def main(split_dir):
    if os.path.exists(os.path.join(split_dir, "fields.npz")) and not os.path.exists(os.path.join(split_dir, "manifest.txt")):
        validate_packed_fields(split_dir)
        return

    insts = list_instances(split_dir)
    print(f"{'inst':<12}{'x_star':>16} | dist(ox_min, chemo_max, ther_max) to x_star | cross-corr ox/chemo,ox/ther")
    dists_all = []
    for d in insts:
        xs, ys, F = _load_field(d)
        x_star = np.array(json.load(open(os.path.join(d, "theta.json")))["x_star"])
        locs = [extremum_loc(xs, ys, F[0], False),
                extremum_loc(xs, ys, F[1], True),
                extremum_loc(xs, ys, F[2], True)]
        dd = [float(np.linalg.norm(l - x_star)) for l in locs]
        dists_all.append(dd)
        # correlation between channels (flattened) — want correlated but not identical
        ox, ch, th = F[0].ravel(), F[1].ravel(), F[2].ravel()
        c1 = np.corrcoef(ox, ch)[0, 1]; c2 = np.corrcoef(ox, th)[0, 1]
        print(f"{os.path.basename(d):<12}({x_star[0]:6.0f},{x_star[1]:6.0f}) |"
              f"  {dd[0]:7.1f} {dd[1]:7.1f} {dd[2]:7.1f}  |  {c1:+.2f} {c2:+.2f}")
    dists_all = np.array(dists_all)
    print(f"\nmean extremum->x_star distance (micron): ox_min={dists_all[:,0].mean():.1f} "
          f"chemo_max={dists_all[:,1].mean():.1f} ther_max={dists_all[:,2].mean():.1f}")
    print("(grid spacing dx=20 micron; tumor_radius ~120-250, so <~tumor_radius means extremum is ON the tumor)")

if __name__ == "__main__":
    default = REPO_ROOT / "data" / "instances" / "train"
    main(sys.argv[1] if len(sys.argv) > 1 else str(default))
