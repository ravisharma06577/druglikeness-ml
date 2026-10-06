"""Step 7: curate the pool and compute all feature sets (descriptors, Morgan bits/counts, MACCS)."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from dl_common import *

log = []
def note(m):
    print(m, flush=True); log.append(m)

df = pd.read_csv(f"{RAW}/pool_raw.csv")
cnt = lambda d: ", ".join(f"{g}={int((d.group == g).sum())}" for g in ["approved", "phase23", "other"])
note("Start: " + cnt(df))

mols = [standardize(s) for s in df.smiles]
df["mol"] = mols
df = df[df.mol.notna()].copy()
note("After parsing/standardising: " + cnt(df))
df = df[df.mol.apply(passes_filters)].copy()
note("After element and size filters: " + cnt(df))
df["key"] = df.mol.apply(lambda m: Chem.MolToSmiles(m, isomericSmiles=False))
prio = {"approved": 0, "phase23": 1, "other": 2}
df = df.assign(_p=df.group.map(prio)).sort_values("_p").drop_duplicates("key", keep="first").drop(columns="_p")
note("After removing duplicate structures (priority approved > phase 2-3 > other): " + cnt(df))
df = df.reset_index(drop=True)
df["can_smiles"] = df.mol.apply(Chem.MolToSmiles)
df["scaffold"] = df.mol.apply(lambda m: MurckoScaffold.MurckoScaffoldSmiles(mol=m))

smiles = df.can_smiles.tolist()
chunks = [smiles[i:i + 500] for i in range(0, len(smiles), 500)]
note(f"Computing features for {len(smiles)} compounds in {len(chunks)} chunks ...")
res = Parallel(n_jobs=-1, verbose=5)(delayed(featurize_chunk)(c) for c in chunks)
D10 = np.vstack([r[0] for r in res]); DALL = np.vstack([r[1] for r in res])
FB = np.vstack([r[2] for r in res]); FC = np.vstack([r[3] for r in res]); MK = np.vstack([r[4] for r in res])

meta = df[["molregno", "chembl_id", "group", "max_phase", "first_approval", "first_doc_year", "can_smiles", "scaffold", "key"]].copy()
for j, name in enumerate(DESC10):
    meta[name] = D10[:, j]
meta.to_csv(f"{CLEAN}/pool_meta.csv", index=False)
np.save(f"{CLEAN}/pool_desc_all.npy", DALL)
np.save(f"{CLEAN}/pool_morgan_bits.npy", FB)
np.save(f"{CLEAN}/pool_morgan_counts.npy", FC)
np.save(f"{CLEAN}/pool_maccs.npy", MK)
pd.Series(RDKIT_NAMES).to_csv(f"{CLEAN}/pool_desc_all_names.csv", index=False, header=False)
note(f"Saved pool: {len(meta)} compounds; RDKit descriptors {DALL.shape[1]}, Morgan bits {FB.shape[1]}, counts {FC.shape[1]}, MACCS {MK.shape[1]}")
open(f"{RES}/pool_cleaning_log.txt", "w").write("\n".join(log))
