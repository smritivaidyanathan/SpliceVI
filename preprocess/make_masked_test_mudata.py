# top of file, add
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import mudata as mu
from scipy import sparse
from joblib import Parallel, delayed

def mask_file(input_path: str, mask_fraction: float, seed: int, n_jobs: int = 1):
    assert 0.0 < mask_fraction < 1.0, "mask_fraction must be in (0, 1)"
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    fracpct = int(round(mask_fraction * 100))
    out_path = in_path.with_name(f"MASKED_{fracpct}_PERCENT_{in_path.name}")  # same dir, prefixed file

    INPUT_PATH  = str(in_path)
    MASK_FRACTION = mask_fraction
    OUTPUT_PATH = str(out_path)
    SEED = seed

    # 1) Load MuData and extract the splicing AnnData
    print(f"Loading MuData from {INPUT_PATH}", flush=True)
    mdata = mu.read_h5mu(INPUT_PATH)
    ad = mdata["splicing"]
    n_obs, n_junc = ad.n_obs, ad.n_vars
    print(f"Splicing AnnData: {n_obs} cells x {n_junc} junctions\n", flush=True)

    # 2) Pull out layers
    print("Pull out the four layers we will modify", flush=True)
    jr_layer  = ad.layers["junc_ratio"]
    cj_layer  = ad.layers["cell_by_junction_matrix"]
    cc_layer  = ad.layers["cell_by_cluster_matrix"]
    psi_layer = ad.layers["psi_mask"]

    # Convert to arrays for easy indexing
    print("Convert to arrays for easy indexing", flush=True)
    jr_arr  = jr_layer.toarray()  if sparse.issparse(jr_layer)  else jr_layer.copy()
    cj_arr  = cj_layer.toarray()  if sparse.issparse(cj_layer)  else cj_layer.copy()
    cc_arr  = cc_layer.toarray()  if sparse.issparse(cc_layer)  else cc_layer.copy()
    psi_arr = psi_layer.toarray() if sparse.issparse(psi_layer) else psi_layer.copy()
    psi_arr = psi_arr.astype(bool)  # ensure boolean mask

    # Keep a copy of the original junc_ratio before any masking
    print("Keep a copy of the original junc_ratio before any masking", flush=True)
    jr_original = jr_arr.copy()

    # Prepare arrays to store outputs
    print("Prepare arrays for outputs", flush=True)
    orig_jr = np.zeros_like(jr_arr)              # original junc_ratio at masked positions
    masked_bin = np.zeros_like(psi_arr, bool)    # NEW: binary mask of masked junctions

    # Event IDs to map junctions → ATSEs
    print("Event IDs to map junctions → ATSEs", flush=True)
    event_ids = ad.var["event_id"].values  # length = n_junc

    # 3) Loop over each cell and mask a fraction of its observed ATSEs
    print("Loop over each cell and mask a fraction of its observed ATSEs", flush=True)
    from tqdm.auto import tqdm

    def process_cell(i: int):
        rng = np.random.RandomState(SEED + i)

        obs_juncs = np.nonzero(psi_arr[i])[0]
        if obs_juncs.size == 0:
            return set(), set(), None

        obs_events = np.unique(event_ids[obs_juncs])
        k = max(1, int(len(obs_events) * MASK_FRACTION))
        mask_events = rng.choice(obs_events, size=k, replace=False)

        masked_atse = set(mask_events.tolist())
        junc_mask = np.isin(event_ids, mask_events)
        masked_junc = set(np.where(junc_mask)[0].tolist())

        example = None
        if mask_events.size > 0:
            atse_ex = mask_events[0]
            junc_indices = np.where(event_ids == atse_ex)[0]
            if junc_indices.size > 0:
                junc_ex = junc_indices[0]
                example = (i, atse_ex, junc_ex)

        orig_jr[i, junc_mask] = jr_arr[i, junc_mask]
        jr_arr [i, junc_mask] = 0
        cj_arr [i, junc_mask] = 0
        cc_arr [i, junc_mask] = 0
        psi_arr[i, junc_mask] = False
        masked_bin[i, junc_mask] = True

        return masked_atse, masked_junc, example

    iterator = tqdm(range(n_obs), desc="Masking cells", unit="cell")
    if n_jobs == 1:
        results = [process_cell(i) for i in iterator]
    else:
        print(f"Using {n_jobs} parallel jobs", flush=True)
        results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(process_cell)(i) for i in iterator)

    masked_atse_set = set()
    masked_junc_set = set()
    example_record  = None  # (cell_idx, atse_id, junc_idx)
    for atse, juncs, ex in results:
        masked_atse_set.update(atse)
        masked_junc_set.update(juncs)
        if example_record is None and ex is not None:
            example_record = ex

    # 4) Write masked data back into the AnnData layers
    print("Write masked data back into the AnnData layers", flush=True)
    ad.layers["junc_ratio_masked_original"] = sparse.csr_matrix(orig_jr)
    ad.layers["junc_ratio"]                 = sparse.csr_matrix(jr_arr)
    ad.layers["cell_by_junction_matrix"]    = sparse.csr_matrix(cj_arr)
    ad.layers["cell_by_cluster_matrix"]     = sparse.csr_matrix(cc_arr)
    ad.layers["psi_mask"]                   = sparse.csr_matrix(psi_arr)
    ad.layers["junc_ratio_masked_bin_mask"] = sparse.csr_matrix(masked_bin.astype(np.uint8))  # NEW

    # 5) Summary
    print(f"Total unique ATSEs masked: {len(masked_atse_set)}", flush=True)
    print(f"Total unique junctions masked: {len(masked_junc_set)}", flush=True)

    if example_record:
        cell_i, atse_id, junc_i = example_record
        print("\nRepresentative example:", flush=True)
        print(f"  Cell index:        {cell_i}", flush=True)
        print(f"  ATSE event ID:     {atse_id}", flush=True)
        print(f"  Junction index:    {junc_i}", flush=True)
        print(f"  Original junc_ratio at that position: {jr_original[cell_i, junc_i]}", flush=True)
        print(f"  After masking, junc_ratio:              {jr_arr[cell_i, junc_i]}", flush=True)
        print(f"  Stored in 'junc_ratio_masked_original': {orig_jr[cell_i, junc_i]}", flush=True)
        print(f"  Binary mask 'junc_ratio_masked_bin_mask': {int(masked_bin[cell_i, junc_i])}", flush=True)

    # 6) Save
    print(f"\nWriting masked MuData to {OUTPUT_PATH}", flush=True)
    mdata.write_h5mu(OUTPUT_PATH)
    print("Write complete.\n", flush=True)

    # 7) Final MuData summary
    print("=== Final MuData Summary ===", flush=True)
    print(mdata)
    print("Modalities:", list(mdata.mod.keys()))
    sp_mod = mdata["splicing"]
    print(f"Splicing layers after masking: {list(sp_mod.layers.keys())}", flush=True)
    print(f"Splicing shape: {sp_mod.n_obs} cells x {sp_mod.n_vars} junctions", flush=True)

def parse_args():
    p = argparse.ArgumentParser(description="Mask a fraction of ATSEs in splicing modality.")
    p.add_argument("--inputs", nargs="+", required=True, help="One or more .h5mu paths")
    p.add_argument("--fractions", nargs="+", type=float, required=True, help="Mask fractions, e.g. 0.1 0.25 0.5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel threads for masking (default=1)")
    return p.parse_args()

def main():
    args = parse_args()
    for f in args.inputs:
        for frac in args.fractions:
            print(f"[submit] {f} at frac={frac}")
            mask_file(f, frac, args.seed, n_jobs=args.n_jobs)

if __name__ == "__main__":
    main()
