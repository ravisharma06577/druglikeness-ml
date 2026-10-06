"""Step 8: how much do the conclusions depend on how the 'non-drug' class is defined?"""
import os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from dl_common import *

N_SEEDS = int(os.environ.get("DL_N_SEEDS", "5"))
KINDS = ["random", "property_matched", "era_matched", "phase23"]
LABELS = {"random": "Random other compounds", "property_matched": "Property-matched", "era_matched": "Era-matched (first-report year)", "phase23": "Phase 2-3, never approved"}
FSETS = ["desc10", "morgan_bits", "both"]

meta, feats = load_pool()
print("Pool:", meta.group.value_counts().to_dict(), flush=True)
rows, t0 = [], time.time()
for kind in KINDS:
    for seed in range(1, N_SEEDS + 1):
        pos, neg = build_dataset(meta, kind, seed)
        idx, y = dataset_frame(meta, pos, neg)
        df = meta.loc[idx].reset_index(drop=True)
        groups = make_groups(df)
        tr, te = split_dataset(groups, y, kind, seed)
        for fs in FSETS:
            X = feature_matrix(meta, feats, idx, fs)
            for name, est in make_models().items():
                est.fit(X[tr], y[tr])
                s = get_score(est, X[te])
                rows.append(dict(definition=kind, seed=seed, features=fs, model=name, n_pos=len(pos), n_neg=len(neg),
                                 n_train=len(tr), n_test=len(te), test_auc=roc_auc_score(y[te], s), test_pr_auc=average_precision_score(y[te], s)))
        print(f"{kind:17s} seed {seed}: {len(pos)} approved vs {len(neg)} others; train {len(tr)}, test {len(te)}; done ({time.time()-t0:.0f}s elapsed)", flush=True)
        pd.DataFrame(rows).to_csv(f"{RES}/negative_sensitivity_raw.csv", index=False)

res = pd.DataFrame(rows)
summ = res.groupby(["definition", "features", "model"]).agg(
    n_pos=("n_pos", "mean"), mean_auc=("test_auc", "mean"), sd_auc=("test_auc", "std"),
    min_auc=("test_auc", "min"), max_auc=("test_auc", "max"), mean_pr_auc=("test_pr_auc", "mean")).round(3).reset_index()
summ.to_csv(f"{RES}/negative_sensitivity_summary.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
print("\n=== TEST ROC-AUC BY NEGATIVE-CLASS DEFINITION (mean and SD over seeds) ===")
print(summ.to_string(index=False))

print("\n=== RANKING OF MODELS WITHIN EACH DEFINITION AND FEATURE SET (best first) ===")
for (d, f), g in summ.groupby(["definition", "features"]):
    print(f"{d:17s} {f:12s}", " > ".join(g.sort_values("mean_auc", ascending=False).model))

print("\n=== FINGERPRINTS MINUS DESCRIPTORS (random forest, mean AUC difference per definition) ===")
rf = summ[summ.model == "RandomForest"].pivot(index="definition", columns="features", values="mean_auc")
print((rf["morgan_bits"] - rf["desc10"]).round(3).to_string())
print("\n=== 'BOTH' MINUS FINGERPRINTS (random forest) ===")
print((rf["both"] - rf["morgan_bits"]).round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 4.5))
w = 0.25
for j, fs in enumerate(FSETS):
    g = summ[(summ.model == "RandomForest") & (summ.features == fs)].set_index("definition").loc[KINDS]
    ax.bar(np.arange(len(KINDS)) + (j - 1) * w, g.mean_auc, w, yerr=g.sd_auc, capsize=3, label=fs)
ax.set_xticks(range(len(KINDS))); ax.set_xticklabels([LABELS[k].replace(" (", "\n(") for k in KINDS], fontsize=8)
ax.axhline(0.5, color="k", lw=0.8, ls="--"); ax.set_ylabel("Test ROC-AUC (random forest)"); ax.set_ylim(0.4, 1.0); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_negative_definitions.png", dpi=300); plt.close()
print("\nSaved negative_sensitivity_raw.csv, negative_sensitivity_summary.csv and fig_negative_definitions.png")
