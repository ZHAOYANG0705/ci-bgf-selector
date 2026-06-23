"""Aggregate advantage re-eval: per-policy detection mean+/-SD over seeds, selector iters, and the
paired selector-random bootstrap (instance-resampled) for the main set."""
import json, glob, sys
import numpy as np

POLS = ["learned", "fusion", "random"]
LAB = {"learned": "selector", "fusion": "fusion", "random": "random"}

files = sorted(glob.glob(sys.argv[1]))
boot = len(sys.argv) > 2 and sys.argv[2] == "boot"
if not files:
    print("no files", sys.argv[1]); sys.exit(0)
rates = {}; steps = {}; per_inst = {}
ninst = None
for f in files:
    d = json.load(open(f)); tot = max(d["total"], 1); ninst = d.get("n_inst")
    for p, c in d["counts"].items():
        rates.setdefault(p, []).append(c / tot)
    for p, s in d.get("mean_steps", {}).items():
        steps.setdefault(p, []).append(s)
    for p, arr in d.get("per_instance", {}).items():
        per_inst.setdefault(p, []).append(np.array(arr, float))
print(f"# {len(files)} seeds, {ninst} instances")
for p in POLS:
    if p in rates:
        a = np.array(rates[p]); it = np.mean(steps.get(p, [float('nan')]))
        print(f"  {LAB.get(p,p):<14} {a.mean():.3f} +/- {a.std():.3f}   iters={it:.1f}")
if boot and "learned" in per_inst and "random" in per_inst:
    # pool per-instance success-fraction across seeds, resample instances
    L = np.concatenate(per_inst["learned"]); R = np.concatenate(per_inst["random"])
    diff = (L - R) / 8.0   # 8 trials/instance
    rng = np.random.default_rng(0)
    bs = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(2000)]
    print(f"  selector-random paired bootstrap: +{diff.mean():.3f}  95% CI [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
