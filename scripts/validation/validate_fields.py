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

def main(split_dir):
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
