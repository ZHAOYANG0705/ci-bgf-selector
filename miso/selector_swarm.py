"""Swarm tumor targeting with per-iteration biological-gradient-field selection.

At each iteration a uniformly actuated swarm observes noisy local values from
each available BGF. The movement operator constructs one candidate swarm
configuration per source. A learned selector scores those source-conditioned
candidates, and the swarm commits to the highest-scoring move. Success is
measured by whether any agent reaches the tumor region within the query budget.
"""
import os, argparse, numpy as np, torch, torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KSRC = 3


# ----------------------------- environment (vectorized over agents) -----------------------------
class BGFEnv:
    """One hidden PhysiCell instance: query k biomarker sources (noisy) + tumor membership test."""
    def __init__(self, fields, mask, xs, ys, noise=0.5, rng=None, x_star=None, success_radius=None):
        self.F = fields                       # (3,ny,nx) torch
        self.mask = mask                      # (ny,nx) torch {0,1}
        self.x_star = x_star                  # (2,) tumor center (harness only)
        self.success_radius = success_radius  # if set: success = within radius of x_star (else mask membership)
        self.xs, self.ys = xs, ys
        self.x0, self.dx = float(xs[0]), float(xs[1] - xs[0])
        self.y0, self.dy = float(ys[0]), float(ys[1] - ys[0])
        self.nx, self.ny = len(xs), len(ys)
        self.lo = torch.tensor([self.x0, self.y0]); self.hi = torch.tensor([float(xs[-1]), float(ys[-1])])
        # per-source value normalization (z-score) so sources are comparable; noise as frac of std
        self.mu = self.F.reshape(3, -1).mean(1); self.sd = self.F.reshape(3, -1).std(1) + 1e-6
        self.noise = noise; self.rng = rng or np.random

    def _bilinear(self, pts, s):
        gx = ((pts[:, 0] - self.x0) / self.dx).clamp(0, self.nx - 1.001)
        gy = ((pts[:, 1] - self.y0) / self.dy).clamp(0, self.ny - 1.001)
        ix = gx.floor().long(); iy = gy.floor().long(); tx = gx - ix.float(); ty = gy - iy.float()
        ix1 = (ix + 1).clamp(max=self.nx - 1); iy1 = (iy + 1).clamp(max=self.ny - 1)
        g = self.F[s]
        v = ((1 - tx) * (1 - ty) * g[iy, ix] + tx * (1 - ty) * g[iy, ix1]
             + (1 - tx) * ty * g[iy1, ix] + tx * ty * g[iy1, ix1])
        return v

    def query(self, pts, s):
        """noisy normalized value of source s at agent positions pts (Na,2). Lower = closer to tumor
        for oxygen; we flip so that LOWER fitness = better for ALL sources (minimization)."""
        v = (self._bilinear(pts, s) - self.mu[s]) / self.sd[s]
        v = v + torch.randn(v.shape) * self.noise
        # oxygen (s=0) is a MIN at tumor -> already 'lower=better'; chemo/ther are MAX at tumor -> negate
        return v if s == 0 else -v

    def in_tumor(self, pts):
        if self.success_radius is not None and self.x_star is not None:
            return (pts - self.x_star).norm(dim=-1) <= self.success_radius
        gx = ((pts[:, 0] - self.x0) / self.dx).round().long().clamp(0, self.nx - 1)
        gy = ((pts[:, 1] - self.y0) / self.dy).round().long().clamp(0, self.ny - 1)
        return self.mask[gy, gx] > 0.5


# ----------------------------- GBR grid aggregation -----------------------------
class GBR:
    """Vectorized grid-based resampling: online weighted mean of each source per grid cell."""
    def __init__(self, env, n=40):
        self.n = n
        self.minx, self.maxx = env.x0, float(env.xs[-1]); self.miny, self.maxy = env.y0, float(env.ys[-1])
        self.dx = (self.maxx - self.minx) / n; self.dy = (self.maxy - self.miny) / n
        self.W = torch.zeros(n * n)                 # accumulated weight per cell
        self.V = torch.zeros(n * n, KSRC)           # weighted-mean value per cell per source
        cx = self.minx + (torch.arange(n) + 0.5) * self.dx
        cy = self.miny + (torch.arange(n) + 0.5) * self.dy
        gx, gy = torch.meshgrid(cx, cy, indexing="ij")
        self.centers = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)   # (n*n,2)

    def update(self, pts, vals_per_src):
        ix = ((pts[:, 0] - self.minx) / self.dx).clamp(0, self.n - 1).long()
        iy = ((pts[:, 1] - self.miny) / self.dy).clamp(0, self.n - 1).long()
        flat = ix * self.n + iy
        ccx = self.minx + (ix + 0.5) * self.dx; ccy = self.miny + (iy + 0.5) * self.dy
        w = torch.exp(-1.414 * torch.hypot(pts[:, 0] - ccx, pts[:, 1] - ccy) * self.n / (self.maxx - self.minx))
        sw = torch.zeros(self.n * self.n).scatter_add(0, flat, w)
        vstack = torch.stack(vals_per_src, -1)      # (Na, KSRC)
        swv = torch.zeros(self.n * self.n, KSRC).scatter_add(0, flat[:, None].expand(-1, KSRC), w[:, None] * vstack)
        Wn = self.W + sw
        self.V = torch.where(Wn[:, None] > 0, (self.V * self.W[:, None] + swv) / Wn[:, None].clamp(min=1e-9), self.V)
        self.W = Wn

    def samplelist(self, max_pts=400):
        nz = (self.W > 0).nonzero().squeeze(-1)
        if nz.numel() == 0:
            return torch.zeros(1, 2 + KSRC + 1)
        nz = nz[:max_pts]
        return torch.cat([self.centers[nz], self.V[nz], self.W[nz, None]], -1)   # (m,2+KSRC+1)


# ----------------------------- selectors -----------------------------
class BGFSelectorNet(nn.Module):
    """PointNet-style selector. mode='abs' uses absolute coords; mode='rel' uses coords relative to
    the swarm centroid (translation-invariant). Branch1 over GBR samples, Branch2 over agent positions."""
    def __init__(self, k=KSRC, mode="rel", hidden=128, dim=2):
        super().__init__()
        self.mode = mode; self.k = k; self.dim = dim
        b1_in = ((dim + 1) if mode == "rel" else dim) + k + 1   # [coords(+radius), k values, weight]
        b2_in = dim
        self.b1 = nn.Sequential(nn.Linear(b1_in, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 512), nn.ReLU())
        self.b2 = nn.Sequential(nn.Linear(b2_in, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 512), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def encode(self, sample, agents):
        # sample (B,m,dim+k+1), agents (B,Na,dim)
        a = agents.mean(1, keepdim=True)                       # swarm centroid (B,1,dim)
        coords = sample[..., :self.dim]; vals = sample[..., self.dim:]
        if self.mode == "rel":
            rc = coords - a; rn = rc.norm(dim=-1, keepdim=True)
            s1 = torch.cat([rc, rn, vals], -1)
            s2 = agents - a
        else:
            s1 = torch.cat([coords, vals], -1); s2 = agents
        return s1, s2

    def forward(self, sample, agents, mask=None):
        s1, s2 = self.encode(sample, agents)
        h1 = self.b1(s1)
        if mask is not None:                                   # ignore padded sample rows in maxpool
            h1 = h1.masked_fill(~mask.bool().unsqueeze(-1), -1e9)
        h1 = h1.max(1).values
        h2 = self.b2(s2).max(1).values
        return self.head(torch.cat([h1, h2], -1))             # (B,1)


# ----------------------------- DEPSO step -----------------------------
def depso_theta(x, fit, k):
    """Elite-weighted center: top-k agents by fitness (lower=better)."""
    idx = torch.argsort(fit)[:k]; ev = fit[idx]
    fmax, fmin = ev.max(), ev.min()
    W = torch.exp((fmax - ev) / (fmax - fmin + 1e-9)); W = W / W.sum()
    return (W[:, None] * x[idx]).sum(0)


def _build_sample(obs_xy, obs_v, M, g):
    """Raw observation buffer -> padded selector Branch1 input (M, 2+K+1): [x,y, v0..vK-1, w=1]. NO GBR."""
    XY = torch.cat(obs_xy, 0); V = torch.cat(obs_v, 0)        # (T,d),(T,K)
    T = XY.shape[0]; d = XY.shape[1]
    if T > M:
        idx = torch.randperm(T, generator=g)[:M]; XY = XY[idx]; V = V[idx]; T = M
    b1 = torch.zeros(M, d + KSRC + 1); mask = torch.zeros(M)
    b1[:T, :d] = XY; b1[:T, d:d + KSRC] = V; b1[:T, d + KSRC] = 1.0; mask[:T] = 1
    return b1, mask


def _inject(env, N, g):
    """FIXED (locked) injection region: a fixed corner box; shared zero initial velocity (WP)."""
    span = env.hi - env.lo
    d = env.lo.numel()
    x = env.lo + span * 0.08 + torch.rand(N, d, generator=g) * span * 0.12   # fixed corner box
    v = torch.zeros(d)                                                        # WP: single shared velocity
    return x, v, span


def wp_candidates(x, v, vals, w, k, c, r, vmax):
    """WP (weak-priority): ALL particles move by the SAME velocity = weakest agent's pull toward the
    elite center theta. Returns per-source (cand_x[N,2], cand_v[2]). Pure directional block-translation,
    no spreading -> detection depends entirely on choosing the right source's direction."""
    cand_x, cand_v = [], []
    for s in range(KSRC):
        theta = depso_theta(x, vals[s], k)              # elite-weighted center for source s
        weak = int(torch.argmax(vals[s]))               # worst-fitness agent
        vv = (w * v + c * r * (theta - x[weak])).clamp(-vmax, vmax)   # shared (2,) velocity
        cand_v.append(vv); cand_x.append(x + vv[None])  # rigid block translation
    return cand_x, cand_v


def gsa_theta(x, fit):
    """GSA (Rashedi 2009) gravitational mass-weighted centre, used as the WP attractor. For
    minimisation (lower fit = better, the oracle's orientation), gravitational mass
    m_i=(f_i-f_worst)/(f_best-f_worst) makes the best agent heaviest and the worst massless;
    M_i=m_i/sum_j m_j. The attractor is the mass-weighted centre theta=sum_i M_i x_i over ALL
    agents -- the gravitational centre of mass a GSA swarm is collectively drawn toward. This is
    the GSA analogue of the elite-weighted centre (all-agent linear masses vs top-k
    exponential weights). Only the (correct) GSA mass law is used; the per-agent force/velocity
    update is not needed under uniform-field actuation."""
    fbest = fit.min(); fworst = fit.max()
    denom = fbest - fworst
    if float(denom.abs()) < 1e-12:                       # degenerate (all-equal) -> uniform masses
        M = torch.ones_like(fit) / fit.numel()
    else:
        m = (fit - fworst) / denom                       # best -> 1, worst -> 0
        ssum = m.sum()
        M = m / ssum if float(ssum) > 1e-12 else torch.ones_like(fit) / fit.numel()
    return (M[:, None] * x).sum(0)                        # (2,) mass-weighted centre


def gsa_candidates(x, v, vals, w, k, c, r, vmax):
    """WP-GSA: identical weak-priority uniform-field shared step as wp_candidates, but the
    per-source attractor is the GSA mass-weighted centre (gsa_theta) instead of the
    elite-weighted centre. Drop-in signature (k unused: GSA weights all agents)."""
    cand_x, cand_v = [], []
    for s in range(KSRC):
        theta = gsa_theta(x, vals[s])                    # GSA mass-weighted centre for source s
        weak = int(torch.argmax(vals[s]))                # worst-fitness agent (same weak-priority rule)
        vv = (w * v + c * r * (theta - x[weak])).clamp(-vmax, vmax)
        cand_v.append(vv); cand_x.append(x + vv[None])   # rigid block translation
    return cand_x, cand_v


def rollout_record(env, N=20, G=150, c=1.5, w_ini=0.9, w_end=0.4, M=256, seed=0, every=2):
    """Random-policy WP rollout; records (raw-obs sample[M], next_x, mask) per step and a success label."""
    g = torch.Generator().manual_seed(seed)
    x, v, span = _inject(env, N, g); vmax = span * 0.10
    obs_xy, obs_v, recs, success = [], [], [], False
    for it in range(G):
        vals = [env.query(x, s) for s in range(KSRC)]
        if env.in_tumor(x).any(): success = True; break
        obs_xy.append(x.clone()); obs_v.append(torch.stack(vals, -1))
        w = (w_ini - w_end) * (G - it) / G + w_end; k = max(3, 20 - int(it * 12 / G))
        r = torch.rand(1, generator=g)
        cand_x, cand_v = wp_candidates(x, v, vals, w, k, c, r, vmax)
        s = int(torch.randint(0, KSRC, (1,), generator=g))               # random source
        nx = torch.max(torch.min(cand_x[s], env.hi), env.lo)
        if it % every == 0:
            b1, mask = _build_sample(obs_xy, obs_v, M, g)
            recs.append([b1, nx.clone(), mask])
        x = nx; v = cand_v[s]
    return recs, (1.0 if success else 0.0)


def run_trial(env, selector, policy, N=20, G=150, c=1.5, w_ini=0.9, w_end=0.4, vmax=None, seed=0, M=256,
              base="depso", return_steps=False):
    """One WP detection trial.

    policy in {'rel','abs','random','fusion'}. 'fusion' is a non-learned
    variant that averages the oriented sources into one objective and takes one
    WP step.
    return_steps -> (success_bool, iters_to_detection [=G if never])."""
    g = torch.Generator().manual_seed(seed)
    x, v, span = _inject(env, N, g); vmax = span * 0.10 if vmax is None else vmax
    obs_xy, obs_v = [], []
    def done(succ, t): return (succ, t) if return_steps else succ
    for it in range(G):
        vals = [env.query(x, s) for s in range(KSRC)]
        if env.in_tumor(x).any():
            return done(True, it)
        obs_xy.append(x.clone()); obs_v.append(torch.stack(vals, -1))
        w = (w_ini - w_end) * (G - it) / G + w_end
        k = max(3, 20 - int(it * 12 / G))
        r = torch.rand(1, generator=g)
        if policy == "fusion":                            # combine sources into one objective, one WP step
            vfused = torch.stack(vals, -1).mean(-1)       # (N,), all sources oriented lower=better
            theta = (gsa_theta(x, vfused) if base == "gsa" else depso_theta(x, vfused, k)); weak = int(torch.argmax(vfused))
            vv = (w * v + c * r * (theta - x[weak])).clamp(-vmax, vmax)
            x = torch.max(torch.min(x + vv[None], env.hi), env.lo); v = vv
            continue
        cand_x, cand_v = (gsa_candidates if base == "gsa" else wp_candidates)(x, v, vals, w, k, c, r, vmax)
        if policy in ("rel", "abs"):
            b1, mask = _build_sample(obs_xy, obs_v, M, g)
            b1 = b1.unsqueeze(0); mask = mask.unsqueeze(0)
            with torch.no_grad():
                scores = [float(selector(b1, cand_x[s].unsqueeze(0), mask)) for s in range(KSRC)]
            sidx = int(np.argmax(scores))
        elif policy == "random":
            sidx = int(torch.randint(0, KSRC, (1,), generator=g))
        else:
            raise ValueError(f"unknown release policy: {policy}")
        x = torch.max(torch.min(cand_x[sidx], env.hi), env.lo); v = cand_v[sidx]
    return done(bool(env.in_tumor(x).any()), G)


# ----------------------------- detection-rate evaluation -----------------------------
def eval_detection(cache, selectors, policies, n_inst=120, trials=8, noise=0.5, seed0=0):
    res = {p: 0 for p in policies}; tot = 0
    for ii in range(min(n_inst, cache["fields"].shape[0])):
        env = BGFEnv(cache["fields"][ii], cache["masks"][ii], cache["xs"], cache["ys"], noise=noise)
        for t in range(trials):
            tot += 1
            for p in policies:
                sel = selectors.get(p)
                if eval_one(env, sel, p, seed=seed0 + 1000 * ii + t):
                    res[p] += 1
    return {p: res[p] / tot for p in policies}


def eval_one(env, sel, p, seed):
    return run_trial(env, sel, p, seed=seed)


def load_cache(root, split):
    d = np.load(os.path.join(root, split, "cache.npz"))
    return dict(fields=torch.tensor(d["fields"], dtype=torch.float32),
                masks=torch.tensor(d["masks"], dtype=torch.float32),
                x_star=torch.tensor(d["x_star"], dtype=torch.float32),
                xs=torch.tensor(d["xs"], dtype=torch.float32), ys=torch.tensor(d["ys"], dtype=torch.float32))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "instances"))
    ap.add_argument("--n_inst", type=int, default=60); ap.add_argument("--trials", type=int, default=6)
    args = ap.parse_args()
    cache = load_cache(args.root, "test")
    # Stage-1 sanity: untrained selectors vs non-learned policies (baseline detection rates)
    sels = {"rel": BGFSelectorNet(mode="rel"), "abs": BGFSelectorNet(mode="abs")}
    pol = ["random", "fusion", "rel", "abs"]
    r = eval_detection(cache, sels, pol, n_inst=args.n_inst, trials=args.trials)
    print("STAGE-1 detection rate (untrained selectors):")
    for p in pol: print(f"  {p:<8} {r[p]:.3f}")
