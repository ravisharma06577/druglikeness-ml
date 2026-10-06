import os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/Drug_Likeness_ML")
CLEAN, RES = f"{BASE}/Clean_Data", f"{BASE}/Results"
df = pd.read_csv(f"{CLEAN}/clean_compounds.csv")
FP = np.load(f"{CLEAN}/X_fp.npy").astype(np.float32)
DESC = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings", "AroRings", "Fsp3", "HeavyAtoms"]
XD = df[DESC].values.astype(np.float32)
y = df.label.values
groups = np.where(df.scaffold.fillna("") == "", "acyclic_" + df.chembl_id, df.scaffold.fillna(""))

feature_sets = {"descriptors": XD, "fingerprints": FP, "both": np.hstack([XD, FP])}
models = {
    "LogReg": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
               {"logisticregression__C": [0.01, 0.1, 1, 10]}),
    "SVM": (make_pipeline(StandardScaler(), SVC(kernel="rbf")),
            {"svc__C": [1, 10], "svc__gamma": ["scale"]}),
    "RandomForest": (RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=1),
                     {"max_depth": [None, 20], "min_samples_leaf": [1, 2]}),
    "GradBoost": (HistGradientBoostingClassifier(random_state=42),
                  {"learning_rate": [0.05, 0.1], "max_depth": [None, 6]}),
}
def get_score(est, X):
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X)

rows = []
for seed in [1, 2, 3, 4, 5]:
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(XD, y, groups))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fs, X in feature_sets.items():
        for name, (est, grid) in models.items():
            gs = GridSearchCV(est, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            gs.fit(X[tr], y[tr], groups=groups[tr])
            rows.append(dict(seed=seed, features=fs, model=name,
                             test_auc=roc_auc_score(y[te], get_score(gs.best_estimator_, X[te]))))
    print("scaffold split", seed, "done", flush=True)
res = pd.DataFrame(rows)
res.to_csv(f"{RES}/repeated_splits.csv", index=False)
summ = res.groupby(["features", "model"]).test_auc.agg(["mean", "std", "min", "max"]).round(3).sort_values("mean", ascending=False)
summ.to_csv(f"{RES}/repeated_splits_summary.csv")
print("\n=== TEST ROC-AUC OVER 5 DIFFERENT SCAFFOLD SPLITS ===")
print(summ.to_string())

print("\n=== LABEL-SHUFFLING CONTROL (random forest, descriptors + fingerprints) ===")
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(XD, y, groups))
Xb = feature_sets["both"]
aucs = []
for k in range(5):
    ys = y[tr].copy()
    np.random.default_rng(k).shuffle(ys)
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1).fit(Xb[tr], ys)
    aucs.append(roc_auc_score(y[te], rf.predict_proba(Xb[te])[:, 1]))
print("Test AUC with shuffled training labels:", np.round(aucs, 3), "mean", round(float(np.mean(aucs)), 3))
