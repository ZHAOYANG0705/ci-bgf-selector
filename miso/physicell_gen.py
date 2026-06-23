"""Randomized PhysiCell instance generator for the MISO benchmark.

Each instance is an independently seeded tumor-bearing tissue with a random
tumor center/radius and random per-substrate physics. The optimizer is therefore
evaluated on a distribution of tissues rather than a fixed field.
We make the cancer cell INDUCE 3 biomarker channels (all extremal at the tumor, different shapes):
  - oxygen        : tumor CONSUMES  -> local MINIMUM at tumor
  - chemoattractant: tumor SECRETES -> local MAXIMUM at tumor (sharp, small D)
  - therapeutic    : tumor SECRETES -> local MAXIMUM at tumor (broad, large D, slow decay)
All three share argmin/argmax location = tumor center = x_star. The metadata in
theta.json is used by the benchmark harness for labeling and scoring, not as a
model input.
"""
import argparse, json, os, copy
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get(
    "MISO_PHYSICELL_CONFIG",
    str(REPO_ROOT / "physicell" / "miso_biomarker_fields" / "config" / "PhysiCell_settings.xml"),
)
ROOT = os.environ.get("MISO_DATA_ROOT", str(REPO_ROOT / "data" / "instances"))
DOMAIN = (-750.0, 750.0)  # x,y min/max (square)
DOMAIN3 = 540.0           # 3D half-extent (compact cube; voxels dominate cost, so this stays tractable)

# split -> (seed_base, center_halfrange, extra_shift_flag)
SPLITS = {
    "train": dict(seed_base=0,        center_hr=400.0, shift=False),
    "val":   dict(seed_base=1_000_000, center_hr=400.0, shift=False),
    # test: wider tumor placement + later we also apply rotation/source-perm in the oracle
    "test":  dict(seed_base=2_000_000, center_hr=520.0, shift=True),
}


# Two generating distributions f_A and f_B (for cross-distribution / sim-to-sim transfer).
# f_A = training distribution; f_B = shifted "different patient cohort / different biology" (test only).
REGIMES = {
    "A": dict(radius=(150, 280), rough=(0.20, 0.45), oxD=(7e4, 1.3e5), oxdec=(0.05, 0.15), oxup=(9, 18),
              hyp=(12, 18), sec=(5, 9), chD=(800, 2500), chdec=(0.05, 0.15), chs=(2, 10),
              thD=(1.5e4, 4e4), thdec=(0.03, 0.08), ths=(2, 10)),
    "B": dict(radius=(280, 420), rough=(0.45, 0.65), oxD=(1.3e5, 2.2e5), oxdec=(0.15, 0.30), oxup=(5, 9),
              hyp=(8, 12), sec=(2, 5), chD=(2500, 5000), chdec=(0.15, 0.30), chs=(0.5, 2),
              thD=(4e4, 7e4), thdec=(0.08, 0.15), ths=(0.5, 2)),
    # C: small & sharp tumors, slow O2 diffusion (steep gradients), strong sharp biomarkers
    "C": dict(radius=(100, 160), rough=(0.10, 0.25), oxD=(4e4, 7e4), oxdec=(0.05, 0.12), oxup=(12, 22),
              hyp=(14, 20), sec=(6, 12), chD=(400, 900), chdec=(0.05, 0.12), chs=(6, 14),
              thD=(1e4, 2e4), thdec=(0.03, 0.06), ths=(6, 14)),
    # D: very irregular / multifocal tumors, high decay
    "D": dict(radius=(180, 300), rough=(0.50, 0.70), oxD=(8e4, 1.4e5), oxdec=(0.08, 0.20), oxup=(8, 16),
              hyp=(10, 16), sec=(3, 7), chD=(1000, 3000), chdec=(0.08, 0.18), chs=(3, 8),
              thD=(2e4, 5e4), thdec=(0.05, 0.10), ths=(3, 8)),
    # E: secretion-imbalanced (weak chemo / strong therapeutic), different diffusion
    "E": dict(radius=(150, 280), rough=(0.25, 0.50), oxD=(7e4, 1.3e5), oxdec=(0.05, 0.15), oxup=(9, 18),
              hyp=(12, 18), sec=(4, 9), chD=(1500, 4000), chdec=(0.10, 0.20), chs=(0.3, 1.5),
              thD=(1.5e4, 4e4), thdec=(0.03, 0.08), ths=(6, 14)),
}


def sample_theta(rng, split, regime="A", growth=False, seed_radius=40.0, max_time=600.0, dim3=False):
    cfg = SPLITS[split]; hr = cfg["center_hr"]; g = REGIMES[regime]
    U = lambda ab: float(rng.uniform(*ab))
    th = dict(
        tumor_center_x=float(rng.uniform(-hr, hr)),
        tumor_center_y=float(rng.uniform(-hr, hr)),
        tumor_radius=U(g["radius"]),
        tumor_shape_seed=int(rng.integers(1, 2**31 - 1)),
        tumor_roughness=U(g["rough"]),
        random_seed=int(rng.integers(0, 2**31 - 1)),
        ox_D=U(g["oxD"]), ox_decay=U(g["oxdec"]), ox_uptake=U(g["oxup"]),
        hypoxia_threshold=U(g["hyp"]), biomarker_secretion=U(g["sec"]),
        chemo_D=U(g["chD"]), chemo_decay=U(g["chdec"]), chemo_secrete=U(g["chs"]),
        ther_D=U(g["thD"]), ther_decay=U(g["thdec"]), ther_secrete=U(g["ths"]),
        max_time=max_time, split=split, regime=regime,
        growth_mode=(1 if growth else 0), tumor_seed_radius=float(seed_radius),
        tumor_center_z=0.0, dim3=(1 if dim3 else 0),
    )
    # keep tumor fully inside the domain with margin.
    if dim3:
        th["tumor_radius"] = float(rng.uniform(165.0, 235.0))  # large enough to form a hypoxic core in 3D
        lim = DOMAIN3 - (th["tumor_radius"] + 60.0)
        th["tumor_center_x"] = float(rng.uniform(-lim, lim))   # place directly inside the 3D cube
        th["tumor_center_y"] = float(rng.uniform(-lim, lim))
        th["tumor_center_z"] = float(rng.uniform(-lim, lim))
        return th
    if growth:
        lim = 300.0                                    # emergent tumors grow large -> place well inside
    else:
        lim = DOMAIN[1] - (th["tumor_radius"] + 60.0)
    th["tumor_center_x"] = float(np.clip(th["tumor_center_x"], -lim, lim))
    th["tumor_center_y"] = float(np.clip(th["tumor_center_y"], -lim, lim))
    return th


def _set(root, path, value):
    el = root.find(path)
    if el is None:
        raise KeyError(f"XML path not found: {path}")
    el.text = str(value)


def _set_substrate_rate(root, cell_name, sub_name, field, value):
    """cell_definitions/cell_definition[@name=cell]/phenotype/secretion/substrate[@name=sub]/field"""
    for cd in root.iter("cell_definition"):
        if cd.get("name") == cell_name:
            sec = cd.find("./phenotype/secretion")
            for sub in sec.findall("substrate"):
                if sub.get("name") == sub_name:
                    sub.find(field).text = str(value)
                    return
    raise KeyError(f"{cell_name}/{sub_name}/{field} not found")


def write_instance(theta, out_dir, base_tree):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "out"), exist_ok=True)
    tree = copy.deepcopy(base_tree)
    r = tree.getroot()

    # --- run control ---
    _set(r, "./overall/max_time", theta["max_time"])
    _set(r, "./options/random_seed", theta["random_seed"])
    _set(r, "./save/folder", os.path.join(out_dir, "out"))
    _set(r, "./save/full_data/interval", theta["max_time"])   # only final field
    _set(r, "./save/SVG/enable", "false")
    _set(r, "./save/SVG/interval", theta["max_time"])

    # --- 3D domain (compact cube) when requested; default stays 2D ---
    if theta.get("dim3"):
        _set(r, "./domain/use_2D", "false")
        for ax in ("x", "y", "z"):
            _set(r, f"./domain/{ax}_min", -DOMAIN3)
            _set(r, f"./domain/{ax}_max", DOMAIN3)

    # --- hidden tumor placement ---
    _set(r, ".//user_parameters/tumor_center_x", theta["tumor_center_x"])
    _set(r, ".//user_parameters/tumor_center_y", theta["tumor_center_y"])
    _set(r, ".//user_parameters/tumor_center_z", theta.get("tumor_center_z", 0.0))
    _set(r, ".//user_parameters/tumor_radius", theta["tumor_radius"])
    _set(r, ".//user_parameters/tumor_shape_seed", theta["tumor_shape_seed"])
    _set(r, ".//user_parameters/tumor_roughness", theta["tumor_roughness"])
    _set(r, ".//user_parameters/hypoxia_threshold", theta["hypoxia_threshold"])
    _set(r, ".//user_parameters/biomarker_secretion", theta["biomarker_secretion"])
    _set(r, ".//user_parameters/growth_mode", theta.get("growth_mode", 0))
    _set(r, ".//user_parameters/tumor_seed_radius", theta.get("tumor_seed_radius", 40.0))
    _set(r, ".//user_parameters/number_of_injected_cells", 0)   # no biorobots -> channels purely tumor-induced
    _set(r, ".//user_parameters/therapy_activation_time", 1e12)

    # --- substrate physics ---
    for sub, D, decay in [("oxygen", theta["ox_D"], theta["ox_decay"]),
                          ("chemoattractant", theta["chemo_D"], theta["chemo_decay"]),
                          ("therapeutic", theta["ther_D"], theta["ther_decay"])]:
        for var in r.iter("variable"):
            if var.get("name") == sub:
                var.find("./physical_parameter_set/diffusion_coefficient").text = str(D)
                var.find("./physical_parameter_set/decay_rate").text = str(decay)

    # --- make cancer cell INDUCE all three channels ---
    _set_substrate_rate(r, "cancer cell", "oxygen", "uptake_rate", theta["ox_uptake"])
    _set_substrate_rate(r, "cancer cell", "chemoattractant", "secretion_rate", theta["chemo_secrete"])
    _set_substrate_rate(r, "cancer cell", "chemoattractant", "secretion_target", 1.0)
    _set_substrate_rate(r, "cancer cell", "therapeutic", "secretion_rate", theta["ther_secrete"])
    _set_substrate_rate(r, "cancer cell", "therapeutic", "secretion_target", 1.0)

    cfg_path = os.path.join(out_dir, "config.xml")
    tree.write(cfg_path)
    # PRIVATE label file (x_star + theta): harness/training only, NEVER model input
    with open(os.path.join(out_dir, "theta.json"), "w") as fh:
        json.dump(dict(theta, x_star=[theta["tumor_center_x"], theta["tumor_center_y"]]), fh, indent=2)
    return cfg_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--regime", default="A", choices=list(REGIMES))
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--growth", action="store_true",
                    help="emergent-growth instances: seed a small cluster, let morphology grow (growth_mode=1)")
    ap.add_argument("--max_time", type=float, default=600.0, help="PhysiCell sim minutes (growth needs more)")
    ap.add_argument("--seed_radius", type=float, default=40.0, help="initial seed radius for --growth")
    ap.add_argument("--dim3", action="store_true", help="3D benchmark: stamp an irregular sphere in a 3D domain")
    args = ap.parse_args()

    base_tree = ET.parse(BASE)
    seed0 = SPLITS[args.split]["seed_base"] + list(REGIMES).index(args.regime) * 7_000_000
    if args.growth:
        seed0 += 5_000_000   # distinct RNG stream for the growth cohort
    if args.dim3:
        seed0 += 11_000_000  # distinct RNG stream for the 3D cohort
    rng = np.random.default_rng(seed0)
    split_dir = os.path.join(args.root, args.split)
    os.makedirs(split_dir, exist_ok=True)
    manifest = os.path.join(split_dir, "manifest.txt")
    with open(manifest, "w") as mf:
        for i in range(args.n):
            theta = sample_theta(rng, args.split, regime=args.regime,
                                 growth=args.growth, seed_radius=args.seed_radius, max_time=args.max_time,
                                 dim3=args.dim3)
            out_dir = os.path.join(split_dir, f"inst_{i:05d}")
            write_instance(theta, out_dir, base_tree)
            mf.write(out_dir + "\n")
    print(f"wrote {args.n} {args.split} instance configs -> {split_dir}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
