"""Train the BGF selector on per-source advantage-label shards."""
import os, sys, glob, argparse, numpy as np, torch, torch.nn as nn
from pathlib import Path
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "miso").is_dir())
sys.path.insert(0, str(REPO_ROOT))
from miso.selector_swarm import BGFSelectorNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)        # glob
    ap.add_argument("--mode", default="rel", choices=["rel", "abs"])
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug_trans", type=float, default=0.0,
                    help="if >0, augment each sample with a random global translation in [-w,w]^2 (um); "
                         "tests whether augmentation can replace relative encoding for an abs model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    files = sorted(glob.glob(args.shards)); assert files, f"no shards {args.shards}"
    b1 = np.concatenate([np.load(f)["b1"] for f in files]); b2 = np.concatenate([np.load(f)["b2"] for f in files])
    mk = np.concatenate([np.load(f)["mask"] for f in files]); y = np.concatenate([np.load(f)["y"] for f in files])
    print(f"loaded {len(y)} samples from {len(files)} shards, pos_rate={y.mean():.3f}", flush=True)
    b1 = torch.tensor(b1); b2 = torch.tensor(b2); mk = torch.tensor(mk); y = torch.tensor(y)
    n = len(y); idx = torch.randperm(n)
    b1, b2, mk, y = b1[idx], b2[idx], mk[idx], y[idx]
    model = BGFSelectorNet(mode=args.mode).to(DEV)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    pr = float(y.mean()); pw = min(max((1 - pr) / max(pr, 1e-3), 1.0), 20.0)
    print(f"pos_rate={pr:.3f} pos_weight={pw:.1f}", flush=True)
    def lossf(o, t):  # weighted BCE (o is already sigmoid prob)
        eps = 1e-6; o = o.clamp(eps, 1 - eps)
        return -(pw * t * o.log() + (1 - t) * (1 - o).log()).mean()
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.3)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n); tot = 0
        for i in range(0, n, args.bs):
            j = perm[i:i + args.bs]
            bb1, bb2 = b1[j], b2[j]
            if args.aug_trans > 0:                         # random global translation per sample
                t = (torch.rand(len(j), 1, 2) * 2 - 1) * args.aug_trans
                bb1 = bb1.clone(); bb2 = bb2.clone()
                bb1[..., :2] += t; bb2[..., :2] += t       # shift obs coords and candidate coords together
            o = model(bb1.to(DEV), bb2.to(DEV), mk[j].to(DEV)).squeeze(-1)
            loss = lossf(o, y[j].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(j)
        sched.step()
        if (ep + 1) % 5 == 0: print(f"epoch {ep+1}: loss={tot/n:.4f}", flush=True)
    torch.save(model.state_dict(), args.out); print("saved ->", args.out)

if __name__ == "__main__":
    main()
