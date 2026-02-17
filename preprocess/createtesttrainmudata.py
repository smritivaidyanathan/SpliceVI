#!/usr/bin/env python
"""
Create a combined MuData from GE + ATSE AnnData, with optional stratified
train/test split.

Key behaviors:
- If MAX_JUNCTIONS_PER_ATSE is None, no junction filtering is performed.
- If filtering is performed, the output filename is suffixed with the
  max-junctions value (unless already present).
- Train/test filenames are derived from the combined filename:
  train_{train_pct}_{test_pct}_{combined_name}.h5mu
  test_{test_pct}_{train_pct}_{combined_name}.h5mu
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from scipy.sparse import csr_matrix, issparse
from sklearn.model_selection import train_test_split

# ---------------------------
# Defaults (match prior behavior)
# ---------------------------
DEFAULT_ROOT_PATH = (
    "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/"
    "HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/102025/"
)
DEFAULT_ATSE_BASENAME = "model_ready_aligned_splicing_data_20251009_023419.h5ad"
DEFAULT_GE_BASENAME = "model_ready_gene_expression_data_20251009_023419.h5ad"
DEFAULT_OUTPUT_BASENAME = (
    "model_ready_combined_gene_expression_aligned_splicing_data_20251009_023419.h5mu"
)
DEFAULT_TEST_SIZE = 0.30
DEFAULT_RANDOM_STATE = 42
DEFAULT_STRATIFY_COLS = ["broad_cell_type", "age"]
DEFAULT_MAX_JUNCTIONS_PER_ATSE = None


# ---------------------------
# Helpers
# ---------------------------

def str_to_bool(val: Union[str, bool]) -> bool:
    if isinstance(val, bool):
        return val
    v = str(val).strip().lower()
    if v in {"true", "t", "1", "yes", "y"}:
        return True
    if v in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {val}")


def none_or_int(val: Union[str, int, None]) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    v = str(val).strip().lower()
    if v in {"none", "null", ""}:
        return None
    return int(v)


def resolve_path(path_str: str, root_path: Optional[str]) -> Path:
    path = Path(path_str)
    if not path.is_absolute() and root_path:
        path = Path(root_path) / path
    return path


def maybe_append_maxjunc_suffix(base_name: str, max_juncs: Optional[int]) -> str:
    if max_juncs is None:
        return base_name

    # Avoid double-appending if the user already included it
    lowered = base_name.lower()
    if "maxjunc" in lowered or "max_junc" in lowered:
        return base_name

    return f"{base_name}_maxjunc_per_atse_{max_juncs}"


def compute_age_numeric(obs: pd.DataFrame) -> pd.Series:
    s = obs["age"]

    if is_integer_dtype(s):
        return s.astype("Int64")

    extracted = (
        s.astype("string")
        .str.strip()
        .str.extract(r"(?i)^\s*(\d+)\s*m?\s*$")[0]
    )
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def compute_age_group_and_label(age_numeric: pd.Series) -> Tuple[pd.Series, pd.Series]:
    # Use bins for human data if many distinct ages
    unique_age_count = age_numeric.dropna().nunique()
    if unique_age_count > 10:
        print("[STRATIFY] Detected many distinct ages; using age bins.")
        years = age_numeric.astype("Float64")
        age_group = pd.cut(
            years,
            bins=[0, 35, 65, np.inf],
            labels=["young", "medium", "old"],
            include_lowest=True,
            right=False,
        )
        age_label = age_group.astype("string").fillna("NA")
        return age_group.astype("string"), age_label

    # Otherwise: keep age_group empty (as before) and use raw numeric age for labels
    age_group = pd.Series(pd.NA, index=age_numeric.index, dtype="string")
    age_label = age_numeric.astype("Int64").astype("string").fillna("NA")
    return age_group, age_label


def build_stratify_combo(
    mdata: mu.MuData,
    stratify_cols: List[str],
    age_label: pd.Series,
) -> pd.Series:
    if not stratify_cols:
        return pd.Series(index=mdata.obs_names, dtype="string")

    labels = []
    for col in stratify_cols:
        if col in {"age", "age_numeric", "age_group"}:
            if col == "age_group":
                labels.append(mdata.obs["age_group"].astype("string").fillna("NA"))
            elif col == "age_numeric":
                labels.append(
                    mdata.obs["age_numeric"].astype("Int64").astype("string").fillna("NA")
                )
            else:
                labels.append(age_label)
        else:
            if col not in mdata.obs.columns:
                raise KeyError(
                    f"Stratify column '{col}' not found in mdata.obs columns."
                )
            labels.append(mdata.obs[col].astype("string").fillna("NA"))

    combo = labels[0].copy()
    for lbl in labels[1:]:
        combo = (combo + "|" + lbl).astype("string")
    return combo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create combined MuData and optional train/test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--root_path",
        default=DEFAULT_ROOT_PATH,
        help="Root directory used to resolve relative paths.",
    )
    parser.add_argument(
        "--atse_data_path",
        default=DEFAULT_ATSE_BASENAME,
        help="ATSE AnnData path (absolute or relative to root_path).",
    )
    parser.add_argument(
        "--ge_data_path",
        default=DEFAULT_GE_BASENAME,
        help="Gene-expression AnnData path (absolute or relative to root_path).",
    )
    parser.add_argument(
        "--output_mudata_path",
        default=DEFAULT_OUTPUT_BASENAME,
        help="Output combined MuData path (absolute or relative to root_path).",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Test set fraction (e.g., 0.30).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for the split.",
    )
    parser.add_argument(
        "--stratify_cols",
        nargs="+",
        default=DEFAULT_STRATIFY_COLS,
        help=(
            "Obs columns to stratify on. Use 'age' to enable age binning logic. "
            "Set to 'none' to disable stratification."
        ),
    )
    parser.add_argument(
        "--do_split",
        type=str_to_bool,
        default=True,
        help="Whether to create train/test splits (true/false).",
    )
    parser.add_argument(
        "--max_junctions_per_atse",
        type=none_or_int,
        default=DEFAULT_MAX_JUNCTIONS_PER_ATSE,
        help=(
            "Maximum junctions per ATSE event. "
            "Use None to skip filtering."
        ),
    )

    return parser.parse_args()


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    args = parse_args()

    # Resolve paths
    atse_path = resolve_path(args.atse_data_path, args.root_path)
    ge_path = resolve_path(args.ge_data_path, args.root_path)
    output_path = resolve_path(args.output_mudata_path, args.root_path)

    # Normalize output extension
    if output_path.suffix != ".h5mu":
        output_path = output_path.with_suffix(".h5mu")

    # Append max-junction suffix if filtering
    combined_base = output_path.stem
    combined_base = maybe_append_maxjunc_suffix(combined_base, args.max_junctions_per_atse)
    if combined_base != output_path.stem:
        output_path = output_path.with_name(f"{combined_base}{output_path.suffix}")

    # Train/test output names (derived later if split requested)
    test_pct = int(round(args.test_size * 100))
    train_pct = 100 - test_pct

    print("=" * 70)
    print("[CONFIG] ATSE data path           :", atse_path)
    print("[CONFIG] GE data path             :", ge_path)
    print("[CONFIG] Combined MuData output   :", output_path)
    print("[CONFIG] Do train/test split      :", args.do_split)
    print("[CONFIG] Test size                :", args.test_size)
    print("[CONFIG] Random state             :", args.random_state)
    print("[CONFIG] Stratify columns         :", args.stratify_cols)
    print("[CONFIG] Max junctions per ATSE   :", args.max_junctions_per_atse)
    print("=" * 70, flush=True)

    # ---------------------------
    # Load modalities
    # ---------------------------
    print("[LOAD] Reading ATSE AnnData...", flush=True)
    atse = ad.read_h5ad(atse_path)
    print("[LOAD] Reading GE AnnData...", flush=True)
    ge = ad.read_h5ad(ge_path)

    print("[LOAD] ATSE:", atse, flush=True)
    print("[LOAD] GE  :", ge, flush=True)

    # ---------------------------
    # Rescale GE length-norm and recompute library size
    # ---------------------------
    print("[PREP] Rescaling GE length_norm and computing library size...", flush=True)
    ge.layers["length_norm"] = (
        ge.layers["length_norm"] * np.median(ge.var["mean_transcript_length"])
    )
    ge.layers["length_norm"].data = np.floor(ge.layers["length_norm"].data)
    ge.obsm["X_library_size"] = ge.layers["length_norm"].sum(axis=1)

    # ---------------------------
    # Parse numeric age
    # ---------------------------
    print("[PREP] Parsing age to numeric...", flush=True)
    for adata in (atse, ge):
        if "age" not in adata.obs.columns:
            raise KeyError("Missing 'age' column in obs.")
        adata.obs["age_numeric"] = compute_age_numeric(adata.obs)
        bad = adata.obs.index[
            adata.obs["age"].notna() & adata.obs["age_numeric"].isna()
        ]
        if len(bad) > 0:
            print(f"[WARN] {len(bad)} 'age' values could not be parsed.")

    # ---------------------------
    # Create var metadata
    # ---------------------------
    print("[PREP] Creating var metadata...", flush=True)
    ge.var["ID"] = ge.var["gene_id"]
    ge.var["modality"] = "Gene_Expression"

    atse.var["ID"] = atse.var["junction_id"]
    atse.var["modality"] = "Splicing"

    # ---------------------------
    # Optional ATSE junction filtering
    # ---------------------------
    if args.max_junctions_per_atse is None:
        print("[FILTER] No junction filtering requested (MAX_JUNCTIONS_PER_ATSE=None).")
    else:
        print("[FILTER] Filtering ATSE by max junctions per event...", flush=True)
        evt = atse.var["event_id"]
        n_junc_before = atse.n_vars
        n_event_before = evt.nunique()

        evt_counts = evt.value_counts()
        per_junc_evt_size = evt.map(evt_counts)
        keep_junctions = (per_junc_evt_size <= args.max_junctions_per_atse).to_numpy()

        print(f"[ATSE] Junctions before: {n_junc_before}")
        print(
            f"[ATSE] Junctions after <= {args.max_junctions_per_atse} per ATSE: "
            f"{int(keep_junctions.sum())}"
        )
        print(f"[ATSE] ATSEs (unique event_id) before: {n_event_before}")

        atse = atse[:, keep_junctions].copy()

        n_event_after = atse.var["event_id"].nunique()
        print(f"[ATSE] ATSEs (unique event_id) after:  {n_event_after}")

    # ---------------------------
    # Compute splicing layers
    # ---------------------------
    print("[PREP] Computing splicing junc_ratio and psi_mask...", flush=True)
    junc = atse.layers["cell_by_junction_matrix"]
    clus = atse.layers["cell_by_cluster_matrix"]

    if not issparse(junc):
        junc = csr_matrix(junc)
    if not issparse(clus):
        clus = csr_matrix(clus)

    mask = clus.copy()
    mask.data = np.ones_like(mask.data, dtype=np.uint8)
    atse.layers["psi_mask"] = mask

    cj = junc.toarray()
    cc = clus.toarray()
    ratio = np.divide(cj, cc, out=np.zeros_like(cj, float), where=(cc != 0))
    atse.layers["junc_ratio"] = ratio

    # ---------------------------
    # Assemble MuData with shared obs columns
    # ---------------------------
    print("[BUILD] Building combined MuData...", flush=True)
    all_cols = ge.obs.columns.union(atse.obs.columns)
    df = pd.DataFrame(index=ge.obs_names, columns=all_cols)

    for col in all_cols:
        if col in ge.obs:
            df[col] = ge.obs[col]
        else:
            df[col] = atse.obs[col]

    ge.obs = df
    atse.obs = df

    mdata = mu.MuData({"rna": ge, "splicing": atse})
    mdata.obsm["X_library_size"] = ge.obsm["X_library_size"]
    mdata.obs = df

    print("[BUILD] MuData:", mdata, flush=True)

    # ---------------------------
    # Compute age_group/age_label (mirrors prior behavior)
    # ---------------------------
    print("[STRATIFY] Computing age_group and age labels...", flush=True)
    age_numeric = mdata.obs["age_numeric"].astype("Float64")
    age_group, age_label = compute_age_group_and_label(age_numeric)
    mdata.obs["age_group"] = age_group

    # ---------------------------
    # Stratified train/test split
    # ---------------------------
    if args.do_split:
        stratify_cols = [c for c in args.stratify_cols if c.lower() not in {"none", "null"}]
        print("[SPLIT] Stratify columns:", stratify_cols, flush=True)

        combo = build_stratify_combo(mdata, stratify_cols, age_label)
        if stratify_cols:
            counts = combo.value_counts()
            rare_mask = combo.isin(counts[counts < 2].index)
            combo_collapsed = combo.mask(rare_mask, "OTHER")
        else:
            combo_collapsed = None

        cells = mdata.obs_names.to_numpy()

        if stratify_cols and (combo_collapsed == "OTHER").sum() == 1:
            print("[SPLIT] Found a single-sample strata; forcing it into train.")
            only_other = cells[combo_collapsed == "OTHER"]
            keep_mask = combo_collapsed != "OTHER"

            train_rest, test_rest = train_test_split(
                cells[keep_mask],
                test_size=args.test_size,
                random_state=args.random_state,
                stratify=combo_collapsed[keep_mask],
            )
            train_cells = np.concatenate([only_other, train_rest])
            test_cells = test_rest
        else:
            train_cells, test_cells = train_test_split(
                cells,
                test_size=args.test_size,
                random_state=args.random_state,
                stratify=combo_collapsed,
            )

        mdata_train = mdata[train_cells, :].copy()
        mdata_test = mdata[test_cells, :].copy()

        train_path = output_path.with_name(
            f"train_{train_pct}_{test_pct}_{combined_base}{output_path.suffix}"
        )
        test_path = output_path.with_name(
            f"test_{test_pct}_{train_pct}_{combined_base}{output_path.suffix}"
        )

        print(
            f"[SPLIT] Total: {len(cells)} | Train: {mdata_train.n_obs} | "
            f"Test: {mdata_test.n_obs} (target test ratio {args.test_size:.2f}; "
            f"actual {mdata_test.n_obs/len(cells):.3f})",
            flush=True,
        )
        print("[WRITE] Train MuData:", train_path, flush=True)
        print(mdata_train)
        print("[WRITE] Test  MuData:", test_path, flush=True)
        print(mdata_test)

        mdata_train.write(train_path)
        mdata_test.write(test_path)

    else:
        print("[SPLIT] Skipping train/test split (do_split=false).", flush=True)

    # ---------------------------
    # Write combined MuData
    # ---------------------------
    print("[WRITE] Combined MuData:", output_path, flush=True)
    print(mdata)
    mdata.write(output_path)

    print("[DONE] Completed MuData creation.")


if __name__ == "__main__":
    main()
