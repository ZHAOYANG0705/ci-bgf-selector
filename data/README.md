# Released data

This directory contains the released data used by the main selector experiments.

```text
data/
├── instances/
│   ├── train/fields.npz  # 1500 PhysiCell tumor instances
│   ├── val/fields.npz    # 250 PhysiCell tumor instances
│   └── test/fields.npz   # 250 PhysiCell tumor instances
└── shards_adv/
    └── adv_*.npz         # per-source advantage-label training shards
```

Each split file stores the simulated BGF fields, tumor masks, tumor centers, tumor radii, and grid axes.
The selector never receives the tumor centers or masks as input; they are used for labels, scoring, and
reproducibility checks.

The `shards_adv` files are derived from the training split and can be passed directly to
`scripts/training/train_selector.py`.
