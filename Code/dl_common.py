"""Shared helpers for the extended analyses (scripts 06-10)."""
import os, warnings
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

BASE = os.path.expanduser(os.environ.get("DL_BASE", "~/Drug_Likeness_ML"))
DB = os.path.expanduser(os.environ.get("DL_DB", "~/.data/chembl/37/chembl_37.db"))
RAW, CLEAN, RES, FIG = (f"{BASE}/{d}" for d in ["Original_Data", "Clean_Data", "Results", "Figures"])
for d in (RAW, CLEAN, RES, FIG):
    os.makedirs(d, exist_ok=True)

ALLOWED = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
DESC10 = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings", "AroRings", "Fsp3", "HeavyAtoms"]
MATCH_COLS = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "HeavyAtoms"]


def standardize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = rdMolStandardize.LargestFragmentChooser().choose(m)
    return rdMolStandardize.Uncharger().uncharge(m)


def passes_filters(m):
    atoms = list(m.GetAtoms())
    return (any(a.GetAtomicNum() == 6 for a in atoms)
            and all(a.GetAtomicNum() in ALLOWED for a in atoms)
            and 10 <= m.GetNumHeavyAtoms() <= 70)


def desc10(m):
    return [Descriptors.MolWt(m), Descriptors.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
            rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
            rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcFractionCSP3(m),
            m.GetNumHeavyAtoms()]


RDKIT_NAMES = [n for n, _ in Descriptors._descList]


def featurize_chunk(smiles_list):
    """Features for a list of already-standardised SMILES (runs inside worker processes)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    D10, DALL, FB, FC, MK = [], [], [], [], []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        D10.append(desc10(m))
        dd = Descriptors.CalcMolDescriptors(m, missingVal=np.nan)
        DALL.append([dd.get(n, np.nan) for n in RDKIT_NAMES])
        FB.append(gen.GetFingerprintAsNumPy(m).astype(np.uint8))
        FC.append(np.clip(gen.GetCountFingerprintAsNumPy(m), 0, 255).astype(np.uint8))
        MK.append(np.frombuffer(MACCSkeys.GenMACCSKeys(m).ToBitString().encode(), dtype=np.uint8) - 48)
    dall = np.array(DALL, dtype=np.float64)
    dall[~np.isfinite(dall)] = np.nan
    dall = np.clip(dall, -1e9, 1e9).astype(np.float32)
    return (np.array(D10, dtype=np.float32), dall, np.array(FB), np.array(FC), np.array(MK, dtype=np.uint8))


def make_models(n_jobs=-1):
    """Fixed hyperparameters taken from the primary tuned analysis (script 03, 'both' feature set)."""
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "LogReg": make_pipeline(imp(), StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)),
        "SVM": make_pipeline(imp(), StandardScaler(), SVC(C=1, gamma="scale")),
        "RandomForest": make_pipeline(imp(), RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=n_jobs)),
        "GradBoost": HistGradientBoostingClassifier(learning_rate=0.1, random_state=42),
    }


def get_score(est, X):
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X)


def make_groups(df):
    sc = df.scaffold.fillna("")
    return np.where(sc == "", "acyclic_" + df.chembl_id.astype(str), sc)


def bootstrap_auc(y, s, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(B, n))
    vals = []
    for i in idx:
        if len(set(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_pool():
    meta = pd.read_csv(f"{CLEAN}/pool_meta.csv")
    feats = {"desc_all": np.load(f"{CLEAN}/pool_desc_all.npy"),
             "morgan_bits": np.load(f"{CLEAN}/pool_morgan_bits.npy"),
             "morgan_counts": np.load(f"{CLEAN}/pool_morgan_counts.npy"),
             "maccs": np.load(f"{CLEAN}/pool_maccs.npy")}
    return meta, feats


def build_dataset(meta, kind, seed):
    """Return (positive_index, negative_index) as positional row indices of `meta`."""
    rng = np.random.default_rng(seed)
    pos = meta.index[meta.group == "approved"].to_numpy()
    others = meta.index[meta.group == "other"].to_numpy()
    if kind == "random":
        n = min(len(pos), len(others))
        return pos, rng.choice(others, size=n, replace=False)
    if kind == "phase23":
        p23 = meta.index[meta.group == "phase23"].to_numpy()
        n = min(len(pos), len(p23))
        return rng.choice(pos, size=n, replace=False), rng.choice(p23, size=n, replace=False)
    if kind == "property_matched":
        Z = (meta[MATCH_COLS] - meta[MATCH_COLS].mean()) / meta[MATCH_COLS].std()
        Zo = Z.loc[others].values
        k = min(200, len(others))
        nn = NearestNeighbors(n_neighbors=k).fit(Zo)
        _, ind = nn.kneighbors(Z.loc[pos].values)
        used, pos_kept, chosen = set(), [], []
        for i in rng.permutation(len(pos)):
            cand = [others[j] for j in ind[i] if others[j] not in used]
            if not cand:
                continue
            pick = int(rng.choice(cand[:3]))
            used.add(pick); pos_kept.append(pos[i]); chosen.append(pick)
        return np.array(pos_kept), np.array(chosen)
    if kind == "era_matched":
        yr = meta.first_doc_year
        by_year = {}
        for i in others:
            if pd.notna(yr[i]):
                by_year.setdefault(int(yr[i]), []).append(i)
        pos_y = [i for i in pos if pd.notna(yr[i])]
        used, pos_kept, chosen = set(), [], []
        for i in rng.permutation(pos_y):
            y0 = int(yr[i])
            for tol in range(0, 4):
                cand = [c for yy in range(y0 - tol, y0 + tol + 1) for c in by_year.get(yy, []) if c not in used]
                if cand:
                    pick = int(rng.choice(cand))
                    used.add(pick); pos_kept.append(i); chosen.append(pick)
                    break
        return np.array(pos_kept), np.array(chosen)
    raise ValueError(kind)


def feature_matrix(meta, feats, idx, name):
    idx = np.asarray(idx)
    if name == "desc10":
        return meta.loc[idx, DESC10].values.astype(np.float32)
    if name == "morgan_bits":
        return feats["morgan_bits"][idx].astype(np.float32)
    if name == "both":
        return np.hstack([meta.loc[idx, DESC10].values.astype(np.float32), feats["morgan_bits"][idx].astype(np.float32)])
    if name == "rdkit_all":
        return feats["desc_all"][idx]
    if name == "morgan_counts":
        return feats["morgan_counts"][idx].astype(np.float32)
    if name == "maccs":
        return feats["maccs"][idx].astype(np.float32)
    if name == "rdkit_all+counts":
        return np.hstack([feats["desc_all"][idx], feats["morgan_counts"][idx].astype(np.float32)])
    raise ValueError(name)


def dataset_frame(meta, pos_idx, neg_idx):
    idx = np.concatenate([pos_idx, neg_idx])
    y = np.concatenate([np.ones(len(pos_idx), int), np.zeros(len(neg_idx), int)])
    return idx, y


def split_dataset(groups, y, kind, seed):
    """Scaffold-disjoint train/test split (GroupShuffleSplit). For matched designs (rows 0..n-1 are the
    positives and rows n..2n-1 their matched negatives), any training compound whose matched partner lies in
    the test set is dropped: otherwise a near-copy of a test compound sits in the training set with the
    opposite label and the model 'anti-learns' (AUC below 0.5)."""
    from sklearn.model_selection import GroupShuffleSplit
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(np.zeros(len(y)), y, groups))
    if kind in ("property_matched", "era_matched"):
        n = len(y) // 2
        partner = np.concatenate([np.arange(n, 2 * n), np.arange(0, n)])
        in_test = np.zeros(len(y), bool); in_test[te] = True
        tr = tr[~in_test[partner[tr]]]
    return tr, te
