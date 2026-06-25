"""Per-source advantage labels for the BGF selector.
At recorded states along random WP rollouts, evaluate EACH source by K continuation rollouts
(first step = that source, then random) -> per-source success value q_s in [0,1]. Each (state,
candidate-next-state-of-source-s, q_s) becomes a training sample. The index range can be sharded."""
import os, sys, argparse, numpy as np, torch
from pathlib import Path
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "miso").is_dir())
sys.path.insert(0, str(REPO_ROOT))
from miso.selector_swarm import BGFEnv, load_fields, wp_candidates, _build_sample, KSRC

W_INI, W_END, C, N = 0.9, 0.4, 1.5, 20


def wstep(it, G): return (W_INI - W_END) * (G - it) / G + W_END
def kstep(it, G): return max(3, 20 - int(it * 12 / G))


def continue_success(env, x, v, it0, G, vmax, g):
    """Continue WP random-source policy from (x,v) at iteration it0; return success bool."""
    for it in range(it0, G):
        vals = [env.query(x, s) for s in range(KSRC)]
        if env.in_tumor(x).any(): return True
        cx, cv = wp_candidates(x, v, vals, wstep(it, G), kstep(it, G), C, torch.rand(1, generator=g), vmax)
        s = int(torch.randint(0, KSRC, (1,), generator=g))
        x = torch.max(torch.min(cx[s], env.hi), env.lo); v = cv[s]
    return bool(env.in_tumor(x).any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--split", default="train")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=25)
    ap.add_argument("--region", default="all", choices=["all", "left", "right"])
    ap.add_argument("--G", type=int, default=150); ap.add_argument("--every", type=int, default=12)
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--n_base", type=int, default=2)
    ap.add_argument("--radius", type=float, default=-1.0); ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--noise", type=float, default=0.5); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = load_fields(args.root, args.split); xstar = data["x_star"]
    R = None if args.radius < 0 else args.radius
    B1, B2, MK, Y = [], [], [], []
    for ii in range(args.start, min(args.end, data["fields"].shape[0])):
        if args.region == "left" and xstar[ii, 0] >= -40: continue
        if args.region == "right" and xstar[ii, 0] <= 40: continue
        env = BGFEnv(data["fields"][ii], data["masks"][ii], data["xs"], data["ys"],
                     noise=args.noise, x_star=data["x_star"][ii], success_radius=R)
        span = env.hi - env.lo; vmax = span * 0.10
        for b in range(args.n_base):
            g = torch.Generator().manual_seed(1000 * ii + b)
            x = env.lo + span * 0.08 + torch.rand(N, 2, generator=g) * span * 0.12; v = torch.zeros(2)
            obs_xy, obs_v = [], []
            for it in range(args.G):
                vals = [env.query(x, s) for s in range(KSRC)]
                if env.in_tumor(x).any(): break
                obs_xy.append(x.clone()); obs_v.append(torch.stack(vals, -1))
                cx, cv = wp_candidates(x, v, vals, wstep(it, args.G), kstep(it, args.G), C, torch.rand(1, generator=g), vmax)
                if it % args.every == 0 and it < args.G - 2:
                    b1, mask = _build_sample(obs_xy, obs_v, args.M, g)
                    for s in range(KSRC):                       # per-source value via K continuations
                        succ = 0
                        for kk in range(args.K):
                            gc = torch.Generator().manual_seed(7 * (1000 * ii + b) + 131 * it + 17 * s + kk)
                            succ += continue_success(env, cx[s].clone(), cv[s].clone(), it + 1, args.G, vmax, gc)
                        B1.append(b1); B2.append(cx[s].clone()); MK.append(mask); Y.append(succ / args.K)
                s = int(torch.randint(0, KSRC, (1,), generator=g))   # advance main rollout (random)
                x = torch.max(torch.min(cx[s], env.hi), env.lo); v = cv[s]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not B1: print("WARN no samples", args.out); return
    np.savez_compressed(args.out, b1=torch.stack(B1).numpy(), b2=torch.stack(B2).numpy(),
                        mask=torch.stack(MK).numpy(), y=np.array(Y, np.float32))
    print(f"shard {args.out}: {len(Y)} samples, mean q={np.mean(Y):.3f}")


if __name__ == "__main__":
    main()
