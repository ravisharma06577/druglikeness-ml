"""Step 10: wider feature panels, repeated grouped cross-validation, and partial-dependence (direction of effects)."""
import os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.inspection import partial_dependence
from sklearn.metrics import roc_auc_score
from dl_common import *

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
N_SEEDS = int(os.environ.get("DL_N_SEEDS", "5"))
meta, feats = load_pool()
pos, neg = build_dataset(meta, "random", 42)
idx, y = dataset_frame(meta, pos, neg)
df = meta.loc[idx].reset_index(drop=True)
groups = make_groups(df)
print(f"Reference dataset: {len(df)} compounds ({int(y.sum())} approved)\n", flush=True)

# ---------- (a) feature panels over several scaffold splits ----------
PANELS = ["desc10", "rdkit_all", "morgan_bits", "morgan_counts", "maccs", "rdkit_all+counts"]
MODELS = ["RandomForest", "GradBoost"]
rows, t0 = [], time.time()
for fs in PANELS:
    X = feature_matrix(meta, feats, idx, fs)
    for seed in range(1, N_SEEDS + 1):
        tr, te = split_dataset(groups, y, "random", seed)
        for name in MODELS:
            est = make_models()[name].fit(X[tr], y[tr])
            rows.append(dict(features=fs, n_features=X.shape[1], seed=seed, model=name, test_auc=roc_auc_score(y[te], get_score(est, X[te]))))
    print(f"panel {fs} done ({time.time()-t0:.0f}s)", flush=True)
pan = pd.DataFrame(rows)
ps = pan.groupby(["features", "n_features", "model"]).test_auc.agg(["mean", "std", "min", "max"]).round(3).reset_index().sort_values("mean", ascending=False)
ps.to_csv(f"{RES}/feature_panels_summary.csv", index=False); pan.to_csv(f"{RES}/feature_panels_raw.csv", index=False)
print("\n=== (a) TEST ROC-AUC BY FEATURE PANEL (mean, SD, range over scaffold splits; fixed hyperparameters) ===")
print(ps.to_string(index=False))

# ---------- (b) repeated grouped cross-validation ----------
REP = int(os.environ.get("DL_N_REPEATS", "5"))
rows = []
for fs in ["desc10", "morgan_bits", "both"]:
    X = feature_matrix(meta, feats, idx, fs)
    for name in make_models():
        for r in range(REP):
            for k, (a, b) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r).split(X, y, groups)):
                est = make_models()[name].fit(X[a], y[a])
                rows.append(dict(features=fs, model=name, repeat=r, fold=k, auc=roc_auc_score(y[b], get_score(est, X[b]))))
    print(f"repeated CV {fs} done ({time.time()-t0:.0f}s)", flush=True)
cvr = pd.DataFrame(rows)
cs = cvr.groupby(["features", "model"]).auc.agg(mean="mean", sd="std", low=lambda s: np.percentile(s, 2.5), high=lambda s: np.percentile(s, 97.5), n_folds="size").round(3).reset_index().sort_values("mean", ascending=False)
cs.to_csv(f"{RES}/repeated_cv_summary.csv", index=False)
print(f"\n=== (b) REPEATED GROUPED CROSS-VALIDATION ({REP} repeats x 5 folds over the whole reference dataset; folds are not independent) ===")
print(cs.to_string(index=False))

# ---------- (c) partial dependence (direction of descriptor effects) ----------
tr, te = split_dataset(groups, y, "random", 42)
Xd = feature_matrix(meta, feats, idx, "desc10")
rf = make_models()["RandomForest"].fit(Xd[tr], y[tr])
fig, axes = plt.subplots(2, 5, figsize=(14, 5.5), sharey=True)
rows = []
for j, (ax, name) in enumerate(zip(axes.ravel(), DESC10)):
    pdp = partial_dependence(rf, Xd[tr], [j], grid_resolution=25, kind="average")
    xs, ys = pdp["grid_values"][0], pdp["average"][0]
    ax.plot(xs, ys); ax.set_xlabel(name)
    lo, hi = np.percentile(Xd[tr][:, j], [10, 90])
    f = lambda v: float(np.interp(v, xs, ys))
    rows.append(dict(descriptor=name, p10=round(lo, 2), p90=round(hi, 2), predicted_prob_at_p10=round(f(lo), 3), predicted_prob_at_p90=round(f(hi), 3),
                     change_p10_to_p90=round(f(hi) - f(lo), 3), direction="higher value -> more approved-like" if f(hi) > f(lo) else "higher value -> less approved-like"))
axes[0, 0].set_ylabel("Predicted probability approved"); axes[1, 0].set_ylabel("Predicted probability approved")
plt.tight_layout(); plt.savefig(f"{FIG}/fig_partial_dependence.png", dpi=300); plt.close()
pdf = pd.DataFrame(rows).sort_values("change_p10_to_p90", key=abs, ascending=False)
pdf.to_csv(f"{RES}/partial_dependence_summary.csv", index=False)
print("\n=== (c) DIRECTION OF DESCRIPTOR EFFECTS (descriptor-only random forest; partial dependence, 10th to 90th percentile) ===")
print(pdf.to_string(index=False))
print("\nSaved feature_panels_summary.csv, repeated_cv_summary.csv, partial_dependence_summary.csv and fig_partial_dependence.png")
