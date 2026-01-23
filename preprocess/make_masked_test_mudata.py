import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import mudata as mu
from scipy import sparse
from tqdm.auto import tqdm
from collections import defaultdict


def mask_file(input_path: str, mask_fraction: float, seed: int, batch_size: int):
    assert 0.0 < mask_fraction < 1.0, "mask_fraction must be in (0, 1)"
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    fracpct = int(round(mask_fraction * 100))
    out_path = in_path.with_name(f"MASKED_{fracpct}_PERCENT_{in_path.name}")

    INPUT_PATH  = str(in_path)
    MASK_FRACTION = mask_fraction
    OUTPUT_PATH = str(out_path)
    SEED = seed

    # 1) Load MuData
    print(f"Loading MuData from {INPUT_PATH}", flush=True)
    mdata = mu.read_h5mu(INPUT_PATH)
    ad = mdata["splicing"]
    n_obs, n_junc = ad.n_obs, ad.n_vars
    print(f"Splicing AnnData: {n_obs} cells x {n_junc} junctions\n", flush=True)

    # 2) Pull out layers as sparse; we will densify in batches
    print("Extracting layers (sparse)...", flush=True)
    jr_layer  = ad.layers["junc_ratio"]
    cj_layer  = ad.layers["cell_by_junction_matrix"]
    cc_layer  = ad.layers["cell_by_cluster_matrix"]
    psi_layer = ad.layers["psi_mask"]

    # Handle both sparse and dense inputs
    jr_csr  = jr_layer.tocsr() if sparse.issparse(jr_layer) else sparse.csr_matrix(jr_layer)
    cj_csr  = cj_layer.tocsr() if sparse.issparse(cj_layer) else sparse.csr_matrix(cj_layer)
    cc_csr  = cc_layer.tocsr() if sparse.issparse(cc_layer) else sparse.csr_matrix(cc_layer)
    psi_csr = psi_layer.tocsr().astype(bool) if sparse.issparse(psi_layer) else sparse.csr_matrix(psi_layer.astype(bool))

    # Pre-map junctions to Event IDs for O(1) lookup
    event_ids = ad.var["event_id"].values
    event_to_junc_indices = defaultdict(list)
    for idx, eid in enumerate(event_ids):
        event_to_junc_indices[eid].append(idx)

    rng = np.random.RandomState(SEED)
    masked_atse_set = set()
    masked_junc_set = set()
    example_record  = None 
    example_value = None

    jr_chunks = []
    cj_chunks = []
    cc_chunks = []
    psi_chunks = []
    orig_jr_chunks = []
    masked_bin_chunks = []

    # 3) Loop over batches of cells, densify each batch only
    batch_iter = tqdm(range(0, n_obs, batch_size), desc="Batches", unit="batch")
    for start in batch_iter:
        end = min(start + batch_size, n_obs)
        rows = slice(start, end)

        jr_batch  = jr_csr[rows, :].toarray()
        cj_batch  = cj_csr[rows, :].toarray()
        cc_batch  = cc_csr[rows, :].toarray()
        psi_batch = psi_csr[rows, :].toarray().astype(bool)

        orig_jr_batch = np.zeros_like(jr_batch)
        masked_bin_batch = np.zeros_like(psi_batch, bool)

        for local_idx, cell_idx in enumerate(
            tqdm(range(start, end), desc="Cells", unit="cell", leave=False, position=1)
        ):
            obs_juncs = np.nonzero(psi_batch[local_idx])[0]
            if obs_juncs.size == 0:
                continue

            obs_events = np.unique(event_ids[obs_juncs])
            k = max(1, int(len(obs_events) * MASK_FRACTION))
            mask_events = rng.choice(obs_events, size=k, replace=False)

            curr_mask_cols = []
            for eid in mask_events:
                curr_mask_cols.extend(event_to_junc_indices[eid])

            if not curr_mask_cols:
                continue

            target_cols = np.array(curr_mask_cols, dtype=int)

            masked_atse_set.update(mask_events)
            masked_junc_set.update(curr_mask_cols)

            if example_record is None and mask_events.size > 0:
                atse_ex = mask_events[0]
                junc_ex = event_to_junc_indices[atse_ex][0]
                example_record = (cell_idx, atse_ex, junc_ex)
                example_value = jr_batch[local_idx, junc_ex]

            orig_jr_batch[local_idx, target_cols] = jr_batch[local_idx, target_cols]
            jr_batch [local_idx, target_cols] = 0
            cj_batch [local_idx, target_cols] = 0
            cc_batch [local_idx, target_cols] = 0
            psi_batch[local_idx, target_cols] = False
            masked_bin_batch[local_idx, target_cols] = True

        # Append sparse batches to be stacked after processing
        jr_chunks.append(sparse.csr_matrix(jr_batch))
        cj_chunks.append(sparse.csr_matrix(cj_batch))
        cc_chunks.append(sparse.csr_matrix(cc_batch))
        psi_chunks.append(sparse.csr_matrix(psi_batch))
        orig_jr_chunks.append(sparse.csr_matrix(orig_jr_batch))
        masked_bin_chunks.append(sparse.csr_matrix(masked_bin_batch.astype(np.uint8)))

    # 4) Write back
    print("Converting back to sparse and saving...", flush=True)
    ad.layers["junc_ratio_masked_original"] = sparse.vstack(orig_jr_chunks, format="csr")
    ad.layers["junc_ratio"]                 = sparse.vstack(jr_chunks, format="csr")
    ad.layers["cell_by_junction_matrix"]    = sparse.vstack(cj_chunks, format="csr")
    ad.layers["cell_by_cluster_matrix"]     = sparse.vstack(cc_chunks, format="csr")
    ad.layers["psi_mask"]                   = sparse.vstack(psi_chunks, format="csr")
    ad.layers["junc_ratio_masked_bin_mask"] = sparse.vstack(masked_bin_chunks, format="csr")

    # 5) Summary
    print(f"Total unique ATSEs masked: {len(masked_atse_set)}")
    print(f"Total unique junctions masked: {len(masked_junc_set)}")

    if example_record:
        cell_i, atse_id, junc_i = example_record
        print(f"\nExample Cell {cell_i} | ATSE {atse_id} | Junction {junc_i}")
        if example_value is not None:
            print(f"  Original junc_ratio: {example_value}")

    print(f"\nWriting to {OUTPUT_PATH}")
    mdata.write_h5mu(OUTPUT_PATH)
    print("Done.\n")

def parse_args():
    p = argparse.ArgumentParser(description="Mask a fraction of ATSEs in splicing modality.")
    p.add_argument("--inputs", nargs="+", required=True, help="One or more .h5mu paths")
    p.add_argument("--fractions", nargs="+", type=float, required=True, help="Mask fractions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256, help="Number of cells to densify per batch")
    return p.parse_args()

def main():
    args = parse_args()
    for f in args.inputs:
        for frac in args.fractions:
            mask_file(f, frac, args.seed, args.batch_size)

if __name__ == "__main__":
    main()
