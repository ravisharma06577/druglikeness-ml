"""Step 6: extract approved, phase 2-3 (never approved) and a large pool of other small molecules from ChEMBL."""
import os, sqlite3
import numpy as np, pandas as pd
from dl_common import DB, RAW

N_OTHER = int(os.environ.get("DL_N_OTHER", "20000"))
con = sqlite3.connect(DB)
cols = pd.read_sql("PRAGMA table_info(molecule_dictionary)", con)["name"].tolist()
assert "max_phase" in cols and "first_approval" in cols, "expected columns missing"

ids = lambda cond: pd.read_sql(
    "SELECT md.molregno FROM molecule_dictionary md JOIN compound_structures cs ON md.molregno = cs.molregno "
    f"WHERE md.molecule_type = 'Small molecule' AND cs.canonical_smiles IS NOT NULL AND {cond}", con).molregno.to_numpy()
approved = ids("md.max_phase = 4")
phase23 = ids("md.max_phase IN (2, 3)")
others_all = ids("(md.max_phase IS NULL OR md.max_phase = 0)")
print(f"Small molecules with a structure: approved={len(approved)}, phase 2-3={len(phase23)}, no recorded phase={len(others_all)}")

rng = np.random.default_rng(42)
others = rng.choice(others_all, size=min(N_OTHER, len(others_all)), replace=False)
groups = {**{int(i): "approved" for i in approved}, **{int(i): "phase23" for i in phase23}, **{int(i): "other" for i in others}}

cur = con.cursor()
cur.execute("DROP TABLE IF EXISTS temp.sel")
cur.execute("CREATE TEMP TABLE sel(molregno INTEGER PRIMARY KEY)")
cur.executemany("INSERT INTO sel VALUES (?)", [(i,) for i in groups])
df = pd.read_sql("""
SELECT md.molregno, md.chembl_id, md.max_phase, md.first_approval, cs.canonical_smiles AS smiles
FROM sel JOIN molecule_dictionary md ON sel.molregno = md.molregno
JOIN compound_structures cs ON md.molregno = cs.molregno""", con)
yr = pd.read_sql("""
SELECT cr.molregno, MIN(d.year) AS first_doc_year
FROM sel JOIN compound_records cr ON sel.molregno = cr.molregno
JOIN docs d ON cr.doc_id = d.doc_id WHERE d.year IS NOT NULL GROUP BY cr.molregno""", con)
df = df.merge(yr, on="molregno", how="left")
df["group"] = df.molregno.map(groups)
df = df.drop_duplicates("molregno")
df.to_csv(f"{RAW}/pool_raw.csv", index=False)
print("\nPool saved:", RAW + "/pool_raw.csv")
print(df.groupby("group").agg(n=("molregno", "size"), with_year=("first_doc_year", lambda s: int(s.notna().sum())),
                              median_year=("first_doc_year", "median")))
