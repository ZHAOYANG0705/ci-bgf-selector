"""Validate emergent-growth instances: did the tumor GROW (field footprint >> seed),
and do the 3 channels keep the expected compartment structure (oxygen sink + therapeutic
peak near centre, chemoattractant peak at the rim)? Reads the final PhysiCell .mat.
Pure read + small-grid numpy (no simulation)."""
import sys, os, json, glob
import numpy as np
import scipy.io as sio

from pathlib import Path
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "miso").is_dir())
sys.path.insert(0, str(REPO_ROOT))
from miso.physicell_oracle import _load_field   # (xs, ys, fields[3,ny,nx]) = [ox,chemo,ther]

CH = ["oxygen", "chemoattractant", "therapeutic"]


def offset_of(xs, ys, grid, kind):
    """location of the extremum (argmin for oxygen, argmax for secreted) and the field footprint."""
    g = np.array(grid, dtype=float)
    flat = g.copy()
    if kind == "min":
        idx = np.unravel_index(np.nanargmin(flat), flat.shape)
    else:
        idx = np.unravel_index(np.nanargmax(flat), flat.shape)
    yi, xi = idx
    return np.array([xs[xi], ys[yi]])


def main():
    root = sys.argv[1]                       # e.g. instances_grow_smoke/test
    insts = sorted(glob.glob(os.path.join(root, "inst_*")))
    print(f"# validating {len(insts)} instances under {root}\n")
    for d in insts:
        try:
            th = json.load(open(os.path.join(d, "theta.json")))
            xstar = np.array(th["x_star"])
            xs, ys, F = _load_field(d)        # F[0]=ox, F[1]=chemo, F[2]=ther
            ox, chemo, ther = F[0], F[1], F[2]
            # extrema offsets from the hidden centre
            ox_off  = np.linalg.norm(offset_of(xs, ys, ox,    "min") - xstar)
            ch_off  = np.linalg.norm(offset_of(xs, ys, chemo, "max") - xstar)
            th_off  = np.linalg.norm(offset_of(xs, ys, ther,  "max") - xstar)
            # tumor footprint proxy: spatial std of the oxygen-depletion region (deep sink voxels)
            dep = ox < (np.nanmean(ox) - 0.5 * np.nanstd(ox))
            XX, YY = np.meshgrid(xs, ys)
            if dep.sum() > 3:
                rx = XX[dep] - xstar[0]; ry = YY[dep] - xstar[1]
                footprint = float(np.sqrt((rx**2 + ry**2).mean()))
                ncells_px = int(dep.sum())
            else:
                footprint, ncells_px = float("nan"), 0
            seed = th.get("tumor_seed_radius", float("nan"))
            print(f"{os.path.basename(d)}: seed_r={seed:.0f}  footprint~{footprint:6.1f}um  "
                  f"deplVox={ncells_px:4d} | offsets(um) ox={ox_off:5.1f} ther={th_off:5.1f} chemo={ch_off:6.1f}  "
                  f"| ox[min,max]=[{np.nanmin(ox):.2f},{np.nanmax(ox):.2f}] "
                  f"chemo.max={np.nanmax(chemo):.3f} ther.max={np.nanmax(ther):.3f}")
        except Exception as e:
            print(f"{os.path.basename(d)}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
