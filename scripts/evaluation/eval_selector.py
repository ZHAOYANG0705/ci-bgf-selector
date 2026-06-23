"""Detection-rate evaluation with per-instance records and iterations-to-detection."""
import os, sys, json, argparse, numpy as np, torch
from pathlib import Path
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "miso").is_dir())
sys.path.insert(0, str(REPO_ROOT))
from miso.selector_swarm import BGFEnv, run_trial, load_cache, BGFSelectorNet, KSRC

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--selector", default=""); ap.add_argument("--mode", default="rel", choices=["rel", "abs"])
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=250)
    ap.add_argument("--trials", type=int, default=8); ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--radius", type=float, default=60.0); ap.add_argument("--G", type=int, default=150)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--region", default="all", choices=["all", "left", "right"])
    ap.add_argument("--policies", default="random,fusion,learned")
    ap.add_argument("--base", default="depso", choices=["depso", "gsa"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cache = load_cache(args.root, args.split); xstar = cache["x_star"].numpy()
    R = None if args.radius < 0 else args.radius
    pols = args.policies.split(",")
    sel = None
    if "learned" in pols and args.selector:
        sel = BGFSelectorNet(mode=args.mode); sel.load_state_dict(torch.load(args.selector, map_location="cpu")); sel.eval()
    # per-instance success counts and step sums
    cnt = {p: 0 for p in pols}; steps = {p: 0 for p in pols}; per_inst = {p: [] for p in pols}; tot = 0
    Ninst = cache["fields"].shape[0]
    for ii in range(args.start, min(args.end, Ninst)):
        if args.region == "left" and xstar[ii, 0] >= -40: continue
        if args.region == "right" and xstar[ii, 0] <= 40: continue
        env = BGFEnv(cache["fields"][ii], cache["masks"][ii], cache["xs"], cache["ys"], noise=args.noise,
                     x_star=cache["x_star"][ii], success_radius=R)
        inst_succ = {p: 0 for p in pols}
        for t in range(args.trials):
            tot += 1
            for p in pols:
                pol = args.mode if p == "learned" else p
                ok, st = run_trial(env, sel if p == "learned" else None, pol, N=args.N, G=args.G,
                                   seed=7919 * ii + 13 * t, base=args.base, return_steps=True)
                if ok: cnt[p] += 1; inst_succ[p] += 1
                steps[p] += st
        for p in pols: per_inst[p].append(inst_succ[p])
    json.dump({"counts": cnt, "total": tot, "trials": args.trials, "base": args.base,
               "mean_steps": {p: steps[p] / max(tot, 1) for p in pols},
               "per_instance": per_inst, "n_inst": len(per_inst[pols[0]])}, open(args.out, "w"))
    print(f"{args.out}: " + " ".join(f"{p}={cnt[p]/max(tot,1):.3f}" for p in pols))

if __name__ == "__main__":
    main()
