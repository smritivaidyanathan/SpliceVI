# %% [markdown]
# # Notebook for MuData Creation from GE and ATSE AnnData
#
# This notebook:
# 1. Reads and inspects ATSE and gene expression AnnData files.
# 2. Fixes NaNs in the splicing data and adds a numeric age column.
# 3. Creates modality-specific `.var` metadata.
# 4. Builds a MuData object containing only the shared `.obs` columns.
# 5. Splits into stratified train/test by (cell type, age) and writes out the results.

# %% [markdown]
# ## 0. Set Paths and Configuration

# ROOT_PATH          = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/"
# ATSE_DATA_PATH     = ROOT_PATH + "model_ready_aligned_splicing_data_20251009_024406.h5ad"
# GE_DATA_PATH       = ROOT_PATH + "model_ready_gene_expression_data_20251009_024406.h5ad"
# OUTPUT_MUDATA_PATH = ROOT_PATH + "model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5ad"

ROOT_PATH          = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/HUMAN_SPLICING_FOUNDATION/MODEL_INPUT/102025/"
ATSE_DATA_PATH     = ROOT_PATH + "model_ready_aligned_splicing_data_20251009_023419.h5ad"
GE_DATA_PATH       = ROOT_PATH + "model_ready_gene_expression_data_20251009_023419.h5ad"
OUTPUT_MUDATA_PATH = ROOT_PATH + "model_ready_combined_gene_expression_aligned_splicing_data_20251009_023419.h5mu"

print("ATSE data path:", ATSE_DATA_PATH)
print("GE data path:  ", GE_DATA_PATH)
print("Output MuData path:", OUTPUT_MUDATA_PATH)

MAX_JUNCTIONS_PER_ATSE = 100000000000

print(f"MAX_JUNCTIONS_PER_ATSE: {MAX_JUNCTIONS_PER_ATSE}")

# %% [markdown]
# ## 1. Imports

# %%
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix, issparse
import mudata as mu
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## 2. Load and Preprocess AnnData

# %%
# Load modalities
atse = ad.read_h5ad(ATSE_DATA_PATH)
ge   = ad.read_h5ad(GE_DATA_PATH)

print(atse, flush = True)
print(ge , flush = True)

# Rescale GE length‐norm and recompute library size
ge.layers["length_norm"] = (
    ge.layers["length_norm"] * np.median(ge.var["mean_transcript_length"])
)
ge.layers["length_norm"].data = np.floor(ge.layers["length_norm"].data)
ge.obsm["X_library_size"] = ge.layers["length_norm"].sum(axis=1)


# %% [markdown]
# ## 3. Parse Numeric Age

# %%
import pandas as pd
from pandas.api.types import is_integer_dtype

for adata in (atse, ge):
    s = adata.obs["age"]

    if is_integer_dtype(s):
        # Already integer dtype → keep as-is (use nullable Int64 to preserve NAs)
        adata.obs["age_numeric"] = s.astype("Int64")
    else:
        # Convert to string, strip, extract leading digits (optionally followed by 'm')
        # Matches '12m', '12 m', '12', ' 12M ', etc.
        extracted = (
            s.astype("string")           # handles mixed/object dtypes and NaNs
             .str.strip()
             .str.extract(r'(?i)^\s*(\d+)\s*m?\s*$')[0]
        )

        adata.obs["age_numeric"] = pd.to_numeric(extracted, errors="coerce").astype("Int64")

        # Optional: flag rows that failed to parse (had a value but became NA)
        bad = adata.obs.index[adata.obs["age"].notna() & adata.obs["age_numeric"].isna()]
        if len(bad) > 0:
            print(f"[warn] {len(bad)} 'age' values could not be parsed in this AnnData.")

# %% [markdown]
# ## 4. Create `.var` Metadata

# %%
ge.var["ID"]       = ge.var["gene_id"]
ge.var["modality"] = "Gene_Expression"

atse.var["ID"]       = atse.var["junction_id"]
atse.var["modality"] = "Splicing"

# 2b. Subset ATSE to junctions whose ATSE (event_id) has ≤ MAX_JUNCTIONS_PER_ATSE junctions
print(atse)
# %%
import pandas as pd

evt = atse.var["event_id"]                 # one event_id per junction (var)
n_junc_before  = atse.n_vars
n_event_before = evt.nunique()

# Count how many junctions belong to each event_id
evt_counts = evt.value_counts()            # index: event_id, value: num junctions in that ATSE
per_junc_evt_size = evt.map(evt_counts)    # align count to each junction

# Keep junctions only if their ATSE has ≤ threshold junctions
keep_junctions = (per_junc_evt_size <= MAX_JUNCTIONS_PER_ATSE).to_numpy()

print(f"[ATSE] Junctions before: {n_junc_before}")
print(f"[ATSE] Junctions after ≤{MAX_JUNCTIONS_PER_ATSE} per ATSE: {int(keep_junctions.sum())}")
print(f"[ATSE] ATSEs (unique event_id) before: {n_event_before}")

# Apply subset
atse = atse[:, keep_junctions].copy()

# Recompute ATSE count after subsetting
n_event_after = atse.var["event_id"].nunique()
print(f"[ATSE] ATSEs (unique event_id) after:  {n_event_after}")


print(atse)

# %% [markdown]
# ## 5. Compute Splicing `junc_ratio` and `psi_mask`

# %%
# Ensure sparse then compute ratio
junc = atse.layers["cell_by_junction_matrix"]
clus = atse.layers["cell_by_cluster_matrix"]

if not issparse(junc):
    junc = csr_matrix(junc)
if not issparse(clus):
    clus = csr_matrix(clus)

# Build mask (1 where cluster > 0)
mask = clus.copy()
mask.data = np.ones_like(mask.data, dtype=np.uint8)
atse.layers["psi_mask"] = mask

# Compute junc_ratio = junc / clus (NaN→0)
cj = junc.toarray()
cc = clus.toarray()
ratio = np.divide(cj, cc, out=np.zeros_like(cj, float), where=(cc != 0))
atse.layers["junc_ratio"] = ratio

# %% [markdown]
# ## 6. Assemble MuData with Shared `.obs` Columns

import pandas as pd

# 1) Compute the union of obs‐columns
all_cols = ge.obs.columns.union(atse.obs.columns)
print(all_cols)

# 2) Build a new DataFrame, preferring the 'rna' values when a column exists in both
df = pd.DataFrame(index=ge.obs_names, columns=all_cols)

for col in all_cols:
    if col in ge.obs:
        df[col] = ge.obs[col]
    else:
        df[col] = atse.obs[col]

# 3) Assign it back to each AnnData and to mdata
ge.obs = df
atse.obs = df

mdata = mu.MuData({"rna": ge, "splicing": atse})
mdata.obsm["X_library_size"] = ge.obsm["X_library_size"]
mdata.obs = df

print(mdata)

# %% [markdown]
# ## 7. Stratified Train/Test Split by (Cell Type, Age)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

TEST_SIZE    = 0.30
RANDOM_STATE = 42

# 1) Make age_group (fixed cutoffs). If age_numeric is in months, use (mdata.obs["age_numeric"]/12)
years = mdata.obs["age_numeric"].astype("Float64")
mdata.obs["age_group"] = pd.cut(
    years, bins=[0, 35, 65, np.inf], labels=["young", "medium", "old"],
    include_lowest=True, right=False
)

# 2) Build combined strata and collapse rare ones
ct   = mdata.obs["broad_cell_type"].astype("string").fillna("NA")
ageg = mdata.obs["age_group"].astype("string").fillna("NA")
combo = (ct + "|" + ageg).astype("string")

counts = combo.value_counts()
rare_mask = combo.isin(counts[counts < 2].index)
combo_collapsed = combo.mask(rare_mask, "OTHER")

# Edge case: if OTHER has only 1 sample total, force it to train and stratify the rest
cells = mdata.obs_names.to_numpy()
if (combo_collapsed == "OTHER").sum() == 1:
    only_other = cells[combo_collapsed == "OTHER"]
    keep_mask  = combo_collapsed != "OTHER"

    train_rest, test_rest = train_test_split(
        cells[keep_mask],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=combo_collapsed[keep_mask],
    )
    train_cells = np.concatenate([only_other, train_rest])
    test_cells  = test_rest
else:
    train_cells, test_cells = train_test_split(
        cells,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=combo_collapsed,  # single call, robust
    )

# Slice
mdata_train = mdata[train_cells, :].copy()
mdata_test  = mdata[test_cells,  :].copy()

print(f"Total: {len(cells)} | Train: {mdata_train.n_obs} | Test: {mdata_test.n_obs} "
      f"(target test ratio {TEST_SIZE:.2f}; actual {mdata_test.n_obs/len(cells):.3f})")


# %% [markdown]
# ## 8. Write Out Results

# %%
mdata_train.write(ROOT_PATH + "train_70_30_model_ready_aligned_splicing_data_20251009_023419.h5mu")
mdata_test.write( ROOT_PATH + "test_30_70_model_ready_aligned_splicing_data_20251009_023419.h5mu")
mdata.write(OUTPUT_MUDATA_PATH)

print(
    f"Training cells: {mdata_train.n_obs} | "
    f"Testing cells: {mdata_test.n_obs}"
)
