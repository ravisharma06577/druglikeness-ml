import os, sqlite3
import pandas as pd

DB = os.path.expanduser("~/.data/chembl/37/chembl_37.db")
OUT = os.path.expanduser("~/Drug_Likeness_ML/Original_Data")
con = sqlite3.connect(DB)

cols = pd.read_sql("PRAGMA table_info(molecule_dictionary)", con)["name"].tolist()
print("molecule_dictionary columns:", cols)
assert "max_phase" in cols, "max_phase column not found - send me the line above"

print("\nCompounds by max_phase:")
print(pd.read_sql("SELECT max_phase, COUNT(*) AS n FROM molecule_dictionary GROUP BY max_phase ORDER BY max_phase", con))

base = """
SELECT md.chembl_id, md.max_phase, cs.canonical_smiles AS smiles
FROM molecule_dictionary md
JOIN compound_structures cs ON md.molregno = cs.molregno
WHERE md.molecule_type = 'Small molecule'
  AND cs.canonical_smiles IS NOT NULL
"""
approved = pd.read_sql(base + " AND md.max_phase = 4", con)
others = pd.read_sql(base + " AND (md.max_phase IS NULL OR md.max_phase = 0)", con)
print("\nApproved small molecules:", len(approved))
print("Other small molecules (max_phase NULL or 0):", len(others))

approved["label"] = 1
others = others.sample(n=min(8000, len(others)), random_state=42)
others["label"] = 0
approved.to_csv(os.path.join(OUT, "approved_raw.csv"), index=False)
others.to_csv(os.path.join(OUT, "others_sample_raw.csv"), index=False)
print("\nSaved two files in", OUT)
