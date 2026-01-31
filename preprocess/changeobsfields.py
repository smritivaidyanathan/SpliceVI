"""Replace obs metadata in MuData files using a template AnnData.

This script loads a template .h5ad (AnnData) file and a list of .h5mu
(MuData) files. For each MuData, it overwrites:
  1) the top-level mdata.obs, and
  2) the modality-level obs for 'splicing' and 'rna' (if present),
with the obs table from the template (aligned by cell/barcode index).

The updated MuData is written next to the original file with the suffix
"_UPDATEDOBS" inserted before the extension.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import anndata as ad
import mudata as mu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace obs fields in MuData files using a template AnnData"
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Path to template .h5ad file whose obs will be copied",
    )
    parser.add_argument(
        "--mudata-paths",
        nargs="+",
        required=True,
        help="One or more .h5mu files to update",
    )
    parser.add_argument(
        "--tag",
        default="_UPDATEDOBS",
        help="Tag inserted before file extension when writing output",
    )
    parser.add_argument(
        "--skip-cell-check",
        action="store_true",
        help="Skip strict cell-name equality check between template and MuData",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help=(
            "Allow template to contain extra cells; will update only the intersection "
            "in MuData order. Errors if MuData has cells not in the template."
        ),
    )
    return parser.parse_args()


def load_template(template_path: str) -> ad.AnnData:
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Template AnnData not found: {path}")
    print(f"[TEMPLATE] Loading AnnData from {path}")
    tmpl = ad.read_h5ad(path)
    print(f"[TEMPLATE] Loaded obs shape: {tmpl.obs.shape}")
    return tmpl


def align_obs(
    template_obs,
    target_index: Iterable[str],
    ctx: str,
    allow_subset: bool,
    skip_check: bool,
):
    """Return template obs rows aligned to target_index.

    If allow_subset is True, extra cells in the template are ignored, but any
    cells present in the target and missing from the template still raise.
    """
    if skip_check:
        return template_obs.loc[target_index]

    tmpl_set = set(template_obs.index)
    tgt_set = set(target_index)

    missing_in_template = tgt_set - tmpl_set
    extra_in_template = tmpl_set - tgt_set

    if missing_in_template:
        msg = [f"Cell mismatch in {ctx}:", f"  Missing in template: {len(missing_in_template)}"]
        raise ValueError("\n".join(msg))

    if not allow_subset and extra_in_template:
        msg = [f"Cell mismatch in {ctx}:", f"  Extra in template: {len(extra_in_template)}"]
        raise ValueError("\n".join(msg))

    if extra_in_template:
        print(
            f"[INFO] {ctx}: template has {len(extra_in_template)} extra cells; "
            "updating intersection only."
        )

    # Preserve target order while discarding template-only cells
    target_in_template = [idx for idx in target_index if idx in tmpl_set]
    return template_obs.loc[target_in_template]


def make_output_path(in_path: Path, tag: str) -> Path:
    suffix = "".join(in_path.suffixes) or in_path.suffix
    return in_path.with_name(f"{in_path.stem}{tag}{suffix}")


def overwrite_obs_for_modality(
    mdata: mu.MuData,
    modality: str,
    template_obs,
    allow_subset: bool,
    skip_check: bool,
) -> None:
    if modality not in mdata.mod:
        print(f"[WARN] Modality '{modality}' not found; skipping.")
        return

    adata = mdata.mod[modality]
    aligned_obs = align_obs(template_obs, adata.obs_names, f"{modality} modality", allow_subset, skip_check)
    adata.obs = aligned_obs.copy()
    print(f"[OK] Updated obs for modality '{modality}' with {adata.n_obs} cells.")


def process_mudata(
    mdata_path: str,
    template_obs,
    tag: str,
    allow_subset: bool,
    skip_check: bool,
) -> Tuple[Path, Path]:
    in_path = Path(mdata_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    print(f"\n[LOAD] Reading MuData: {in_path}")
    mdata = mu.read_h5mu(in_path)
    print(mdata)
    print(mdata.obs)
    print(f"[INFO] Cells: {mdata.n_obs}; Modalities: {list(mdata.mod.keys())}")

    aligned_obs = align_obs(template_obs, mdata.obs_names, "MuData top-level", allow_subset, skip_check)
    mdata.obs = aligned_obs.copy()
    print("[OK] Updated top-level mdata.obs")

    for modality in ("splicing", "rna"):
        overwrite_obs_for_modality(mdata, modality, template_obs, allow_subset, skip_check)

    out_path = make_output_path(in_path, tag)
    print(f"[SAVE] Writing updated MuData to {out_path}")
    print(mdata)
    print(mdata.obs)
    mdata.write_h5mu(out_path)
    return in_path, out_path


def main() -> None:
    args = parse_args()
    template = load_template(args.template)
    template_obs = template.obs.copy()

    print("[START] Processing MuData files...")
    for m_path in args.mudata_paths:
        process_mudata(m_path, template_obs, args.tag, args.allow_subset, args.skip_cell_check)
    print("\n[DONE] All files processed.")


if __name__ == "__main__":
    main()
