import os, time, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve)
warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/Drug_Likeness_ML")
CLEAN, RES, FIG = f"{BASE}/Clean_Data", f"{BASE}/Results", f"{BASE}/Figures"
os.makedirs(f"{RES}/models", exist_ok=True)

df = pd.read_csv(f"{CLEAN}/clean_compounds.csv")
FP = np.load(f"{CLEAN}/X_fp.npy").astype(np.float32)
DESC = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings", "AroRings", "Fsp3", "HeavyAtoms"]
XD = df[DESC].values.astype(np.float32)
y = df.label.values

# acyclic molecules have no scaffold: give each its own group
groups = np.where(df.scaffold.fillna("") == "", "acyclic_" + df.chembl_id, df.scaffold.fillna(""))
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(XD, y, groups))
assert not (set(groups[tr]) & set(groups[te])), "scaffold leakage between train and test"
print(f"Train: {len(tr)} compounds ({y[tr].mean():.2%} approved), {len(set(groups[tr]))} scaffold groups")
print(f"Test : {len(te)} compounds ({y[te].mean():.2%} approved), {len(set(groups[te]))} scaffold groups")

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
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def get_score(est, X):
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X)

rows, preds, curves = [], [], {}
for fs, X in feature_sets.items():
    for name, (est, grid) in models.items():
        t0 = time.time()
        gs = GridSearchCV(est, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
        gs.fit(X[tr], y[tr], groups=groups[tr])
        s = get_score(gs.best_estimator_, X[te])
        p = gs.best_estimator_.predict(X[te])
        rows.append(dict(features=fs, model=name, best_params=str(gs.best_params_),
                         cv_auc=round(gs.best_score_, 4),
                         accuracy=accuracy_score(y[te], p), precision=precision_score(y[te], p),
                         recall=recall_score(y[te], p), f1=f1_score(y[te], p),
                         roc_auc=roc_auc_score(y[te], s), pr_auc=average_precision_score(y[te], s)))
        preds.append(pd.DataFrame(dict(chembl_id=df.chembl_id.values[te], features=fs, model=name,
                                       y_true=y[te], score=s, pred=p)))
        curves[(fs, name)] = roc_curve(y[te], s)[:2]
        joblib.dump(gs.best_estimator_, f"{RES}/models/{fs}_{name}.joblib")
        print(f"{fs:13s} {name:13s} CV AUC={gs.best_score_:.3f}  test AUC={rows[-1]['roc_auc']:.3f}  ({time.time()-t0:.0f}s)")

# Lipinski rule-of-five baseline on the same test set
T = df.iloc[te]
rule = ((T.MW <= 500) & (T.LogP <= 5) & (T.HBD <= 5) & (T.HBA <= 10)).astype(int).values
rows.append(dict(features="rule", model="Lipinski_Ro5", best_params="-", cv_auc=np.nan,
                 accuracy=accuracy_score(y[te], rule), precision=precision_score(y[te], rule),
                 recall=recall_score(y[te], rule), f1=f1_score(y[te], rule), roc_auc=np.nan, pr_auc=np.nan))

res = pd.DataFrame(rows)
res.to_csv(f"{RES}/model_results.csv", index=False)
pd.concat(preds).to_csv(f"{RES}/test_predictions.csv", index=False)
np.save(f"{RES}/test_index.npy", te)
np.save(f"{RES}/train_index.npy", tr)

show = res.drop(columns="best_params").copy()
for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
    show[c] = show[c].round(3)
print("\n=== TEST SET RESULTS ===")
print(show.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, fs in zip(axes, feature_sets):
    for name in models:
        fpr, tpr = curves[(fs, name)]
        auc = res[(res.features == fs) & (res.model == name)].roc_auc.iloc[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(fs); ax.set_xlabel("False positive rate"); ax.legend(fontsize=8)
axes[0].set_ylabel("True positive rate")
plt.tight_layout(); plt.savefig(f"{FIG}/fig_roc_curves.png", dpi=300); plt.close()

pv = res[res.features != "rule"].pivot(index="model", columns="features", values="roc_auc")
pv.plot(kind="bar", figsize=(7, 4.5)); plt.ylabel("Test ROC-AUC"); plt.xticks(rotation=0)
plt.ylim(0.5, 1.0); plt.tight_layout(); plt.savefig(f"{FIG}/fig_model_comparison.png", dpi=300); plt.close()
print("\nSaved results to", RES, "and figures to", FIG)
