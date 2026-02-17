#!/usr/bin/env python
"""
subsample_mudata.py

Subsample a MuData file to a target number of cells, stratified by one or more obs keys.
Outputs a new MuData alongside the source with a suffix: SUBSAMPLED_N_<n>.h5mu.
Prints the original and resulting MuData summaries and progress updates.

Usage:
  python subsample_mudata.py \
      --input path/to/data.h5mu \
      --n_cells 5000 \
      --stratify_on broad_cell_type medium_cell_type
"""

import argparse
from pathlib import Path
from typing import List
import numpy as np
import mudata as mu
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Subsample a MuData with stratification")
    ap.add_argument("--input", required=True, help="Path to input .h5mu file")
    ap.add_argument("--n_cells", type=int, required=True, help="Total cells to keep")
    ap.add_argument(
        "--stratify_on",
        nargs="+",
        default=[],
        help="Obs columns to stratify on (comma-joined key). If empty, uniform sampling.",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"[LOAD] Reading MuData from {input_path}")
    mdata = mu.read_h5mu(input_path)
    print("[LOAD] Original MuData:")
    print(mdata)
    total_cells = mdata.n_obs

    strat_keys = args.stratify_on
    if strat_keys:
        missing = [k for k in strat_keys if k not in mdata.obs.columns]
        if missing:
            raise ValueError(f"Stratify keys missing in obs: {missing}")
        group_labels = mdata.obs[strat_keys].astype(str).agg("|".join, axis=1)
    else:
        group_labels = pd.Series(["all"] * total_cells, index=mdata.obs.index)

    # determine quota per group
    group_counts = group_labels.value_counts()
    print(f"[INFO] Found {len(group_counts)} strata:")
    for g, c in group_counts.items():
        print(f"        {g}: {c}")

    # allocate proportional samples per group, then fix rounding by distributing remainder
    desired = (group_counts / group_counts.sum() * args.n_cells).astype(int)
    remainder = args.n_cells - desired.sum()
    if remainder > 0:
        # give remainder to the largest groups first
        for g in group_counts.sort_values(ascending=False).index[:remainder]:
            desired[g] += 1

    selected_indices = []
    for g, need in desired.items():
        available = group_labels[group_labels == g].index.to_numpy()
        take = min(need, len(available))
        choice = rng.choice(available, size=take, replace=False)
        selected_indices.append(choice)
    selected_indices = np.concatenate(selected_indices)
    print(f"[SAMPLE] Requested {args.n_cells}, selected {len(selected_indices)} cells")

    # slice MuData
    mdata_sub = mdata[selected_indices, :].copy()
    print("[RESULT] Subsampled MuData:")
    print(mdata_sub)

    # write output
    suffix = f"SUBSAMPLED_N_{len(selected_indices)}.h5mu"
    output_path = input_path.with_name(input_path.stem + "_" + suffix)
    print(f"[WRITE] Writing subsample to {output_path}")
    mdata_sub.write_h5mu(output_path)


if __name__ == "__main__":
    main()
