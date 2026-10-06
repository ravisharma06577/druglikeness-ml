"""Step 9: established baselines, time-based split, era check, deployment prevalence, calibration."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import QED, Crippen
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from dl_common import *

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
meta, feats = load_pool()
pos, neg = build_dataset(meta, "random", 42)
idx, y = dataset_frame(meta, pos, neg)
df = meta.loc[idx].reset_index(drop=True)
groups = make_groups(df)
tr, te = split_dataset(groups, y, "random", 42)
Xb = feature_matrix(meta, feats, idx, "morgan_bits"); Xd = feature_matrix(meta, feats, idx, "desc10"); Xboth = feature_matrix(meta, feats, idx, "both")
print(f"Reference dataset: {len(df)} compounds ({int(y.sum())} approved); train {len(tr)}, test {len(te)}\n", flush=True)

# ---------- (a) baselines ----------
mols = [Chem.MolFromSmiles(s) for s in df.can_smiles]
res = []
def add_auc(name, s):
    a, lo, hi = bootstrap_auc(y[te], s); res.append(dict(method=name, metric="ROC-AUC", value=a, ci_low=lo, ci_high=hi))
rf = make_models()["RandomForest"].fit(Xboth[tr], y[tr]); s_rf = get_score(rf, Xboth[te])
add_auc("Random forest, descriptors + fingerprints (primary model)", s_rf)
add_auc("QED score", np.array([QED.qed(mols[i]) for i in te]))
lr2 = make_pipeline(StandardScaler(), LogisticRegression()).fit(df.loc[tr, ["MW", "LogP"]], y[tr])
add_auc("Logistic regression on molecular weight and logP only", lr2.predict_proba(df.loc[te, ["MW", "LogP"]])[:, 1])
yr = df.first_doc_year.copy(); yr = yr.fillna(yr.iloc[tr].median())
lry = LogisticRegression().fit(yr.iloc[tr].to_frame(), y[tr])
add_auc("First-report year alone (era shortcut)", lry.predict_proba(yr.iloc[te].to_frame())[:, 1])
def tanimoto(A, B):
    A = A.astype(np.float32); B = B.astype(np.float32); inter = A @ B.T
    return inter / np.maximum(A.sum(1)[:, None] + B.sum(1)[None, :] - inter, 1)
S = tanimoto(Xb[te], Xb[tr])
ptr = y[tr] == 1
add_auc("Nearest-neighbour similarity (to training approved minus to training other)", S[:, ptr].max(1) - S[:, ~ptr].max(1))
auc_tab = pd.DataFrame(res)

def rule_row(name, pred):
    p = pred.astype(int)
    return dict(method=name, accuracy=accuracy_score(y[te], p), precision=precision_score(y[te], p, zero_division=0),
                recall=recall_score(y[te], p), f1=f1_score(y[te], p))
T = df.iloc[te]
ro5 = ((T.MW <= 500) & (T.LogP <= 5) & (T.HBD <= 5) & (T.HBA <= 10)).values
veber = ((T.RotB <= 10) & (T.TPSA <= 140)).values
natoms = np.array([Chem.AddHs(mols[i]).GetNumAtoms() for i in te]); mr = np.array([Crippen.MolMR(mols[i]) for i in te])
ghose = ((T.MW >= 160) & (T.MW <= 480) & (T.LogP >= -0.4) & (T.LogP <= 5.6) & (natoms >= 20) & (natoms <= 70) & (mr >= 40) & (mr <= 130)).values
q = np.array([QED.qed(mols[i]) for i in te])
rules = pd.DataFrame([rule_row("Lipinski rule of five", ro5), rule_row("Veber rules", veber), rule_row("Ghose filter", ghose), rule_row("QED >= 0.5", q >= 0.5)])
print("=== (a) BASELINES: ROC-AUC with 95% bootstrap CI (test set) ==="); print(auc_tab.round(3).to_string(index=False))
print("\n=== (a) RULE-BASED FILTERS (test set) ==="); print(rules.round(3).to_string(index=False))
auc_tab.to_csv(f"{RES}/baselines_auc.csv", index=False); rules.to_csv(f"{RES}/baselines_rules.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 3.5))
t = auc_tab.iloc[::-1]
ax.barh(range(len(t)), t.value, xerr=[t.value - t.ci_low, t.ci_high - t.value], color="#4C72B0", capsize=3)
ax.set_yticks(range(len(t))); ax.set_yticklabels([m.split(",")[0].split(" (")[0][:48] for m in t.method], fontsize=8)
ax.axvline(0.5, color="k", ls="--", lw=0.8); ax.set_xlabel("Test ROC-AUC"); ax.set_xlim(0.3, 1.0)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_baselines.png", dpi=300); plt.close()

# ---------- (b) time-based split ----------
ok = df.first_doc_year.notna().values
cut = np.percentile(df.first_doc_year[ok], 80)
tr_t = np.where(ok & (df.first_doc_year.values <= cut))[0]; te_t = np.where(ok & (df.first_doc_year.values > cut))[0]
print(f"\n=== (b) TIME-BASED SPLIT: train on compounds first reported up to {int(cut)} ({len(tr_t)}), test on later ({len(te_t)}; {y[te_t].mean():.1%} approved) ===", flush=True)
rows = []
for fs, X in [("desc10", Xd), ("morgan_bits", Xb), ("both", Xboth)]:
    for name, est in make_models().items():
        est.fit(X[tr_t], y[tr_t]); s = get_score(est, X[te_t])
        a, lo, hi = bootstrap_auc(y[te_t], s)
        rows.append(dict(features=fs, model=name, n_train=len(tr_t), n_test=len(te_t), cutoff_year=int(cut), test_auc=a, ci_low=lo, ci_high=hi))
tt = pd.DataFrame(rows); tt.to_csv(f"{RES}/time_split_results.csv", index=False); print(tt.round(3).to_string(index=False))

# ---------- (c) era check ----------
yrs_te = df.first_doc_year.values[te]; okte = ~np.isnan(yrs_te)
rho, p = spearmanr(s_rf[okte], yrs_te[okte])
med = np.nanmedian(yrs_te)
print(f"\n=== (c) ERA CHECK (primary random-forest model, scaffold-split test set) ===")
print(f"Spearman correlation between predicted score and first-report year: rho = {rho:.3f} (p = {p:.3g})")
for lab, m in [(f"first reported up to {int(med)}", okte & (yrs_te <= med)), (f"first reported after {int(med)}", okte & (yrs_te > med))]:
    if m.sum() > 20 and len(set(y[te][m])) == 2:
        print(f"ROC-AUC, compounds {lab} (n = {int(m.sum())}): {roc_auc_score(y[te][m], s_rf[m]):.3f}")

# ---------- (d) deployment prevalence and enrichment ----------
fpr, tpr, thr = roc_curve(y[te], s_rf)
rows = []
for pi in [0.5, 0.05, 0.01, 0.0017]:
    for target in [0.2, 0.5, 0.8]:
        k = np.argmax(tpr >= target)
        prec = tpr[k] * pi / (tpr[k] * pi + fpr[k] * (1 - pi))
        rows.append(dict(prevalence=pi, recall=round(float(tpr[k]), 3), false_positive_rate=round(float(fpr[k]), 3), precision=round(float(prec), 4), enrichment=round(float(prec / pi), 2)))
prev = pd.DataFrame(rows); prev.to_csv(f"{RES}/deployment_prevalence.csv", index=False)
print("\n=== (d) EXPECTED PRECISION AT REALISTIC PREVALENCE (primary model; 0.0017 = approx. share of approved drugs among ChEMBL small molecules) ===")
print(prev.to_string(index=False))
order = np.argsort(-s_rf)
for frac in [0.01, 0.05, 0.10]:
    k = max(1, int(round(frac * len(order)))); hit = y[te][order[:k]].mean()
    print(f"Balanced test set: top {int(frac*100)}% of scores contain {hit:.1%} approved (enrichment {hit / y[te].mean():.2f}x)")

# ---------- (e) calibration ----------
def ece(y_, p_, bins=10):
    edges = np.linspace(0, 1, bins + 1); tot = 0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p_ >= a) & (p_ < b) if b < 1 else (p_ >= a) & (p_ <= b)
        if m.sum(): tot += m.sum() / len(p_) * abs(y_[m].mean() - p_[m].mean())
    return tot
cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(Xboth[tr], y[tr], groups[tr]))
cal = CalibratedClassifierCV(make_models()["RandomForest"], method="sigmoid", cv=cv).fit(Xboth[tr], y[tr])
p_cal = cal.predict_proba(Xboth[te])[:, 1]
print("\n=== (e) CALIBRATION (random forest, descriptors + fingerprints) ===")
print(f"Uncalibrated: Brier {brier_score_loss(y[te], s_rf):.3f}, expected calibration error {ece(y[te], s_rf):.3f}")
print(f"Sigmoid-calibrated (fitted by grouped cross-validation on training data): Brier {brier_score_loss(y[te], p_cal):.3f}, expected calibration error {ece(y[te], p_cal):.3f}")
pd.DataFrame([dict(version="uncalibrated", brier=brier_score_loss(y[te], s_rf), ece=ece(y[te], s_rf)),
              dict(version="sigmoid-calibrated", brier=brier_score_loss(y[te], p_cal), ece=ece(y[te], p_cal))]).to_csv(f"{RES}/calibration_results.csv", index=False)
fig, ax = plt.subplots(figsize=(4.5, 4.2))
for lab, pp in [("Uncalibrated", s_rf), ("Sigmoid-calibrated", p_cal)]:
    bins = np.linspace(0, 1, 11); mid, obs = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (pp >= a) & (pp < b) if b < 1 else (pp >= a) & (pp <= b)
        if m.sum() >= 5: mid.append(pp[m].mean()); obs.append(y[te][m].mean())
    ax.plot(mid, obs, marker="o", label=lab)
ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction approved"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_calibration.png", dpi=300); plt.close()
print("\nSaved baselines_auc.csv, baselines_rules.csv, time_split_results.csv, deployment_prevalence.csv, calibration_results.csv, fig_baselines.png, fig_calibration.png")
