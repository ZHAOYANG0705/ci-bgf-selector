"""Cache all instance fields of a split into one npz for fast selector training/evaluation."""
import sys, os, json, glob, argparse
from pathlib import Path
import numpy as np
import scipy.io as sio

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "miso").is_dir())
ROOT = os.environ.get("MISO_DATA_ROOT", str(REPO_ROOT / "data" / "instances"))


def load_field_fast(inst_dir):
    mats = sorted(glob.glob(os.path.join(inst_dir, "out", "*microenvironment0.mat")))
    final = [m for m in mats if "final" in os.path.basename(m)]
    path = final[0] if final else mats[-1]
    A = sio.loadmat(path)["multiscale_microenvironment"]  # (7, Nvox)
    x, y = A[0], A[1]
    xs, ys = np.unique(x), np.unique(y)
    nx, ny = len(xs), len(ys)
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    ix = np.round((x - xs[0]) / dx).astype(int)
    iy = np.round((y - ys[0]) / dy).astype(int)
    F = np.full((3, ny, nx), np.nan, np.float32)
    for s in range(3):
        F[s, iy, ix] = A[4 + s]
    return xs.astype(np.float32), ys.astype(np.float32), F


def load_gt_mask(inst_dir, xs, ys, reach=18.0):
    """GT tumor occupancy mask on the (ys,nx) grid: voxel within `reach` um of any cancer cell."""
    f = sorted(glob.glob(os.path.join(inst_dir, "out", "*cells.mat")))
    f = [m for m in f if "final" in os.path.basename(m)] or f
    M = sio.loadmat(f[-1]); key = [k for k in M if not k.startswith("__")][0]
    A = M[key]
    cx, cy = A[1], A[2]                              # cell x,y (row0=ID,1=x,2=y,3=z)
    X, Y = np.meshgrid(xs, ys)
    m = np.zeros(X.shape, bool)
    r2 = reach * reach
    for px, py in zip(cx, cy):
        m |= ((X - px) ** 2 + (Y - py) ** 2) <= r2
    centroid = np.array([cx.mean(), cy.mean()], np.float32) if len(cx) else np.array([0, 0], np.float32)
    return m.astype(np.uint8), centroid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    split_dir = os.path.join(args.root, args.split)
    insts = [l.strip() for l in open(os.path.join(split_dir, "manifest.txt")) if l.strip()]
    fields, masks, cores, rims, xstar, radius = [], [], [], [], [], []
    xs = ys = None
    ok = 0
    for d in insts:
        try:
            xs, ys, F = load_field_fast(d)
            if np.isnan(F).any():
                print("WARN nan in", d); continue
            m, centroid = load_gt_mask(d, xs, ys)
            if m.sum() < 4:
                print("WARN empty mask", d); continue
            th = json.load(open(os.path.join(d, "theta.json")))
            # compartments: core = tumor & hypoxic (O2<thr); rim = tumor & ~core
            thr = float(th.get("hypoxia_threshold", 15.0))
            core = (m.astype(bool) & (F[0] < thr)).astype(np.uint8)
            rim = (m.astype(bool) & ~core.astype(bool)).astype(np.uint8)
            fields.append(F); masks.append(m); cores.append(core); rims.append(rim)
            xstar.append(centroid); radius.append(th["tumor_radius"]); ok += 1
        except Exception as e:
            print("SKIP", d, e)
    out = os.path.join(split_dir, "cache.npz")
    np.savez_compressed(out, fields=np.stack(fields), masks=np.stack(masks),
                        cores=np.stack(cores), rims=np.stack(rims),
                        x_star=np.array(xstar, np.float32),
                        tumor_radius=np.array(radius, np.float32), xs=xs, ys=ys)
    print(f"cached {ok}/{len(insts)} {args.split} -> {out}  fields {np.stack(fields).shape} "
          f"core_frac={np.stack(cores).mean():.3f} rim_frac={np.stack(rims).mean():.3f}")


if __name__ == "__main__":
    main()
