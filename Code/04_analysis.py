import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

BASE = os.path.expanduser("~/Drug_Likeness_ML")
CLEAN, RES, FIG = f"{BASE}/Clean_Data", f"{BASE}/Results", f"{BASE}/Figures"
df = pd.read_csv(f"{CLEAN}/clean_compounds.csv")
FP = np.load(f"{CLEAN}/X_fp.npy").astype(np.float32)
DESC = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings", "AroRings", "Fsp3", "HeavyAtoms"]
XD = df[DESC].values.astype(np.float32)
y = df.label.values
te, tr = np.load(f"{RES}/test_index.npy"), np.load(f"{RES}/train_index.npy")
y_te, y_tr = y[te], y[tr]
pred = pd.read_csv(f"{RES}/test_predictions.csv")
scores = {(f, m): g.score.values for (f, m), g in pred.groupby(["features", "model"])}

# ---------- 1. Bootstrap confidence intervals ----------
rng = np.random.default_rng(42)
B, n = 1000, len(te)
boot_idx = rng.integers(0, n, size=(B, n))
def boot_auc(s):
    return np.array([roc_auc_score(y_te[i], s[i]) for i in boot_idx])
boots, rows = {}, []
for (f, m), s in scores.items():
    b = boot_auc(s); boots[(f, m)] = b
    rows.append(dict(features=f, model=m, test_auc=roc_auc_score(y_te, s),
                     ci_low=np.percentile(b, 2.5), ci_high=np.percentile(b, 97.5)))
ci = pd.DataFrame(rows).sort_values("test_auc", ascending=False).round(3)
ci.to_csv(f"{RES}/bootstrap_auc.csv", index=False)
print("=== TEST ROC-AUC WITH 95% BOOTSTRAP CI ===")
print(ci.to_string(index=False))
print("\n=== PAIRED DIFFERENCES (same bootstrap samples) ===")
for a, b_, label in [(("both", "RandomForest"), ("fingerprints", "RandomForest"), "RF both - RF fingerprints"),
                     (("fingerprints", "RandomForest"), ("descriptors", "RandomForest"), "RF fingerprints - RF descriptors"),
                     (("both", "RandomForest"), ("both", "SVM"), "RF both - SVM both")]:
    d = boots[a] - boots[b_]
    print(f"{label}: mean diff {d.mean():+.3f}, 95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]")

# ---------- 2. Feature importance ----------
rf_d = joblib.load(f"{RES}/models/descriptors_RandomForest.joblib")
pi = permutation_importance(rf_d, XD[te], y_te, scoring="roc_auc", n_repeats=30, random_state=42, n_jobs=-1)
imp = pd.DataFrame({"descriptor": DESC, "mean_AUC_drop": pi.importances_mean, "sd": pi.importances_std}).sort_values("mean_AUC_drop", ascending=False).round(4)
imp.to_csv(f"{RES}/descriptor_importance.csv", index=False)
print("\n=== DESCRIPTOR IMPORTANCE (permutation, descriptor-only random forest, test set) ===")
print(imp.to_string(index=False))
plt.figure(figsize=(6, 4))
plt.barh(imp.descriptor[::-1], imp.mean_AUC_drop[::-1], xerr=imp.sd[::-1], color="#4C72B0")
plt.xlabel("Drop in test ROC-AUC when shuffled"); plt.tight_layout()
plt.savefig(f"{FIG}/fig_descriptor_importance.png", dpi=300); plt.close()

rf_f = joblib.load(f"{RES}/models/fingerprints_RandomForest.joblib")
top = np.argsort(rf_f.feature_importances_)[::-1][:10]
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
smi_tr = df.can_smiles.values[tr]
def example_fragment(bit):
    try:
        for i in np.where(FP[tr][:, bit] == 1)[0][:50]:
            m = Chem.MolFromSmiles(smi_tr[i])
            ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
            gen.GetFingerprint(m, additionalOutput=ao)
            info = ao.GetBitInfoMap()
            if bit in info:
                a, r = info[bit][0]
                if r == 0:
                    return m.GetAtomWithIdx(a).GetSymbol() + " (single atom environment)"
                env = Chem.FindAtomEnvironmentOfRadiusN(m, r, a)
                sub = Chem.PathToSubmol(m, env, atomMap={})
                return Chem.MolToSmiles(sub) + f" (radius {r})"
    except Exception:
        pass
    return "n/a"
brow = []
for b in top:
    brow.append(dict(bit=int(b), importance=round(float(rf_f.feature_importances_[b]), 4),
                     frac_approved_with_bit=round(float(FP[tr][y_tr == 1][:, b].mean()), 3),
                     frac_other_with_bit=round(float(FP[tr][y_tr == 0][:, b].mean()), 3),
                     example_fragment=example_fragment(int(b))))
bits = pd.DataFrame(brow)
bits.to_csv(f"{RES}/fingerprint_bit_importance.csv", index=False)
print("\n=== TOP 10 FINGERPRINT BITS (fingerprint-only random forest) ===")
print(bits.to_string(index=False))

# ---------- 3. Applicability domain ----------
def tanimoto(A, Bm):
    inter = A @ Bm.T
    return inter / np.maximum(A.sum(1)[:, None] + Bm.sum(1)[None, :] - inter, 1)
Ftr, Fte = FP[tr], FP[te]
nn_te = tanimoto(Fte, Ftr).max(1)
S = tanimoto(Ftr, Ftr); np.fill_diagonal(S, -1); nn_tr = S.max(1)
thr = np.percentile(nn_tr, 5)
s_best = scores[("both", "RandomForest")]
p_best = (s_best >= 0.5).astype(int)
inside = nn_te >= thr
def summ(mask, label):
    r = dict(group=label, n=int(mask.sum()), accuracy=float((p_best[mask] == y_te[mask]).mean()) if mask.sum() else np.nan)
    r["roc_auc"] = roc_auc_score(y_te[mask], s_best[mask]) if mask.sum() and len(set(y_te[mask])) == 2 else np.nan
    return r
ad = [summ(inside, f"inside domain (nearest-neighbour Tanimoto >= {thr:.2f})"), summ(~inside, "outside domain")]
q = pd.qcut(nn_te, 4, labels=["Q1 (least similar)", "Q2", "Q3", "Q4 (most similar)"])
for lab in q.categories:
    ad.append(summ(np.asarray(q == lab), f"similarity quartile {lab}"))
ad = pd.DataFrame(ad).round(3)
ad.to_csv(f"{RES}/applicability_domain.csv", index=False)
print(f"\n=== APPLICABILITY DOMAIN (random forest, both features) ===")
print(f"Threshold = 5th percentile of training-set nearest-neighbour similarity = {thr:.3f}")
print(f"Median nearest-neighbour similarity: training {np.median(nn_tr):.3f}, test {np.median(nn_te):.3f}")
print(ad.to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(nn_tr, bins=30, alpha=0.6, label="Training (leave-one-out)")
ax[0].hist(nn_te, bins=30, alpha=0.6, label="Test")
ax[0].axvline(thr, color="k", ls="--", label="Domain threshold")
ax[0].set_xlabel("Max Tanimoto similarity to training set"); ax[0].set_ylabel("Compounds"); ax[0].legend(fontsize=8)
qa = ad[ad.group.str.startswith("similarity quartile")]
ax[1].bar(range(4), qa.accuracy.values, color="#55A868")
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["Q1\nleast similar", "Q2", "Q3", "Q4\nmost similar"])
ax[1].set_ylim(0.4, 1); ax[1].set_ylabel("Test accuracy")
plt.tight_layout(); plt.savefig(f"{FIG}/fig_applicability_domain.png", dpi=300); plt.close()

# ---------- 4. Error analysis ----------
T = df.iloc[te].reset_index(drop=True).copy()
T["score"], T["pred"] = s_best, p_best
T["correct"] = T.pred == T.label
fp_n = int(((T.pred == 1) & (T.label == 0)).sum()); fn_n = int(((T.pred == 0) & (T.label == 1)).sum())
print(f"\n=== ERROR ANALYSIS (random forest, both features) ===")
print(f"Test compounds: {len(T)}; errors: {int((~T.correct).sum())} (false positives {fp_n}, false negatives {fn_n})")
print("Fraction of errors with predicted probability between 0.40 and 0.60:",
      round(float(T[~T.correct].score.between(0.4, 0.6).mean()), 3))
print("\nMedian descriptors, correctly vs incorrectly classified:")
print(T.groupby("correct")[DESC].median().round(2).T)
T["MW_bin"] = pd.cut(T.MW, [0, 250, 350, 450, 600, 2000])
mw = T.groupby("MW_bin", observed=True).agg(n=("correct", "size"), error_rate=("correct", lambda s: 1 - s.mean())).round(3)
print("\nError rate by molecular-weight bin:"); print(mw)
plt.figure(figsize=(6, 4)); plt.bar(range(len(mw)), mw.error_rate.values, color="#C44E52")
plt.xticks(range(len(mw)), [str(i) for i in mw.index], rotation=20); plt.ylabel("Error rate"); plt.xlabel("Molecular weight bin")
plt.tight_layout(); plt.savefig(f"{FIG}/fig_error_by_mw.png", dpi=300); plt.close()
wrong = T[~T.correct].copy()
wrong["confidence"] = np.where(wrong.pred == 1, wrong.score, 1 - wrong.score)
wrong.sort_values("confidence", ascending=False)[["chembl_id", "can_smiles", "label", "score", "confidence"] + DESC].to_csv(f"{RES}/misclassified_compounds.csv", index=False)
print("\nMost confident false positives (other compounds predicted approved):")
print(wrong[wrong.label == 0].sort_values("confidence", ascending=False).head(5)[["chembl_id", "score", "can_smiles"]].round(3).to_string(index=False))
print("\nMost confident false negatives (approved drugs predicted other):")
print(wrong[wrong.label == 1].sort_values("confidence", ascending=False).head(5)[["chembl_id", "score", "can_smiles"]].round(3).to_string(index=False))
print("\nAll analysis files saved in", RES, "and", FIG)
