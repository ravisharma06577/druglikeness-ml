import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
BASE = os.path.expanduser("~/Drug_Likeness_ML")
RAW, CLEAN, RES, FIG = (f"{BASE}/{d}" for d in ["Original_Data", "Clean_Data", "Results", "Figures"])
log = []
def note(msg):
    print(msg); log.append(msg)

ap = pd.read_csv(f"{RAW}/approved_raw.csv")
ot = pd.read_csv(f"{RAW}/others_sample_raw.csv")
df = pd.concat([ap, ot], ignore_index=True)
note(f"Start: approved={int((df.label==1).sum())}, others={int((df.label==0).sum())}")

chooser = rdMolStandardize.LargestFragmentChooser()
uncharger = rdMolStandardize.Uncharger()
def standardize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = chooser.choose(m)
    m = uncharger.uncharge(m)
    return m

df["mol"] = df.smiles.apply(standardize)
df = df[df.mol.notna()].copy()
note(f"After parsing/standardising (largest fragment, neutralised): approved={int((df.label==1).sum())}, others={int((df.label==0).sum())}")

ALLOWED = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
def ok(m):
    atoms = list(m.GetAtoms())
    has_c = any(a.GetAtomicNum() == 6 for a in atoms)
    only_allowed = all(a.GetAtomicNum() in ALLOWED for a in atoms)
    return has_c and only_allowed and 10 <= m.GetNumHeavyAtoms() <= 70
df = df[df.mol.apply(ok)].copy()
note(f"After filters (organic atoms H,C,N,O,F,P,S,Cl,Br,I; 10-70 heavy atoms): approved={int((df.label==1).sum())}, others={int((df.label==0).sum())}")

df["key"] = df.mol.apply(lambda m: Chem.MolToSmiles(m, isomericSmiles=False))
df = df.sort_values("label", ascending=False).drop_duplicates("key", keep="first")
note(f"After removing duplicate structures (ignoring stereo; approved kept if in both classes): approved={int((df.label==1).sum())}, others={int((df.label==0).sum())}")

n_pos = int((df.label == 1).sum())
neg = df[df.label == 0]
neg = neg.sample(n=min(n_pos, len(neg)), random_state=42)
df = pd.concat([df[df.label == 1], neg]).sample(frac=1, random_state=42).reset_index(drop=True)
note(f"Final balanced set: approved={int((df.label==1).sum())}, others={int((df.label==0).sum())}, total={len(df)}")

desc_fns = {
    "MW": Descriptors.MolWt, "LogP": Descriptors.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA, "HBD": rdMolDescriptors.CalcNumHBD,
    "HBA": rdMolDescriptors.CalcNumHBA, "RotB": rdMolDescriptors.CalcNumRotatableBonds,
    "Rings": rdMolDescriptors.CalcNumRings, "AroRings": rdMolDescriptors.CalcNumAromaticRings,
    "Fsp3": rdMolDescriptors.CalcFractionCSP3, "HeavyAtoms": lambda m: m.GetNumHeavyAtoms(),
}
D = pd.DataFrame([{k: f(m) for k, f in desc_fns.items()} for m in df.mol])
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
FP = np.array([gen.GetFingerprintAsNumPy(m) for m in df.mol], dtype=np.uint8)
df["scaffold"] = df.mol.apply(lambda m: MurckoScaffold.MurckoScaffoldSmiles(mol=m))
df["can_smiles"] = df.mol.apply(Chem.MolToSmiles)

out = pd.concat([df[["chembl_id", "can_smiles", "label", "scaffold"]], D], axis=1)
out.to_csv(f"{CLEAN}/clean_compounds.csv", index=False)
np.save(f"{CLEAN}/X_fp.npy", FP)
note(f"Saved clean_compounds.csv ({out.shape[0]} rows, {out.shape[1]} columns) and X_fp.npy {FP.shape}")
open(f"{RES}/cleaning_log.txt", "w").write("\n".join(log))

print("\nMedian descriptor values by class (1 = approved, 0 = other):")
print(out.groupby("label")[list(desc_fns)].median().round(2).T)
print("\nNumber of distinct scaffolds:", out.scaffold.nunique())

plt.figure(figsize=(5, 4))
out.label.map({1: "Approved", 0: "Other"}).value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"])
plt.ylabel("Compounds"); plt.title("Class distribution"); plt.tight_layout()
plt.savefig(f"{FIG}/fig_class_distribution.png", dpi=300); plt.close()

for col in ["MW", "LogP"]:
    plt.figure(figsize=(6, 4))
    plt.hist(out[out.label == 1][col], bins=40, alpha=0.6, label="Approved")
    plt.hist(out[out.label == 0][col], bins=40, alpha=0.6, label="Other")
    plt.xlabel(col); plt.ylabel("Compounds"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{FIG}/fig_dist_{col}.png", dpi=300); plt.close()

corr = D.corr()
plt.figure(figsize=(7, 6))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(corr)), corr.columns, rotation=90); plt.yticks(range(len(corr)), corr.columns)
plt.colorbar(label="Pearson r"); plt.tight_layout()
plt.savefig(f"{FIG}/fig_descriptor_correlation.png", dpi=300); plt.close()
print("Figures saved in", FIG)
