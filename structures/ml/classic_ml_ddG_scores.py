import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Check which measure of substitution deleteriousness predicts mutation best.
# Same setup as ml/classic_ml_full_dataset.py - full dataset, 10-fold stratified CV, the same eight models. 
# Varies the single column: the Grantham distance versus the three protein-stability predictions -
# structure-based ddG and the two DDGun baselines.

set_config(transform_output="pandas")

HERE = Path(__file__).resolve().parent
IN_TSV = HERE / "ml_table_all_with_ddG.tsv"
STATUS_TSV = HERE.parent / "ddG_results.tsv"
RESULTS_DIR = HERE / "results_classic_ml_ddG"
TARGET = "any_mutation"
N_SPLITS = 10
SEED = 1
os.makedirs(RESULTS_DIR, exist_ok=True)


def out_path(filename):
    return os.path.join(RESULTS_DIR, filename)


df = pd.read_csv(IN_TSV, sep='\t')
status = pd.read_csv(STATUS_TSV, sep='\t')[["position", "status"]]
df = df.merge(status, on="position", how="left", validate="one_to_one")

LOOP_BOUNDARY_COLS = [c for c in df.columns
                      if (c.startswith("loop_boundary_") or c == "in_loop_boundary")
                      and "sequence" not in c]
df[LOOP_BOUNDARY_COLS] = df[LOOP_BOUNDARY_COLS].fillna(0.0)

BASES = ["A", "C", "G", "T"]
THIRD_NT_COLS = [f"third_nt_{b}" for b in BASES]
MINUS1_NT_COLS = [f"minus1_nt_{b}" for b in BASES]
df = pd.get_dummies(df, columns=["third_nt", "minus1_nt"], dtype="uint8")

STABILITY_COLS = ["ddG", "ddgun_seq", "ddgun_3d"]
SCORE_COLS = ["grantham"] + STABILITY_COLS

is_nonsense = (df["status"] == "nonsense").values
print(f"n_rows={len(df)}, n_positives={int(df[TARGET].sum())} ({df[TARGET].mean():.1%})")
for col in STABILITY_COLS:
    col_max = df[col].max()
    missing = df[col].isna()
    df.loc[missing & is_nonsense, col] = col_max
    n_max = int((missing & is_nonsense).sum())
    n_zero = int((missing & ~is_nonsense).sum())
    df[col] = df[col].fillna(0.0)
    print(f"  {col:10s} filled: {n_max} nonsense -> {col_max:.3f} (column max), "
          f"{n_zero} unchanged-protein -> 0")

FEATURES_CONTEXT = THIRD_NT_COLS + MINUS1_NT_COLS + LOOP_BOUNDARY_COLS
FEATURE_SETS = {score: [score] + FEATURES_CONTEXT for score in SCORE_COLS}

y = df[TARGET].values


def make_models(y_true):
    """Same eight models as ml/classic_ml_full_dataset.py."""
    n_pos = int(y_true.sum())
    scale_pos_weight = (len(y_true) - n_pos) / n_pos
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "NaiveBayes": GaussianNB(),
        "SVM": CalibratedClassifierCV(SVC(kernel="rbf", random_state=SEED,
                                          class_weight="balanced"), ensemble=False),
        "LDA": LinearDiscriminantAnalysis(),
        "RandomForest": RandomForestClassifier(random_state=SEED, class_weight="balanced"),
        "LightGBM": LGBMClassifier(random_state=SEED, class_weight="balanced",
                                   n_estimators=100, max_depth=3, verbose=-1, n_jobs=1),
        "XGBoost": XGBClassifier(random_state=SEED, scale_pos_weight=scale_pos_weight,
                                 n_estimators=100, max_depth=3, eval_metric="logloss",
                                 verbosity=0, n_jobs=1),
        "CatBoost": CatBoostClassifier(random_state=SEED, auto_class_weights="Balanced",
                                       iterations=100, depth=3, verbose=False,
                                       allow_writing_files=False, thread_count=1),
    }


SCORE_COLORS = {"grantham": "#7b2cbf", "ddG": "#2a78d6",
                "ddgun_seq": "#e34948", "ddgun_3d": "#2a9d5c"}
SCORE_LABELS = {"grantham": "Grantham distance", "ddG": "FoldX (ΔΔG)",
                "ddgun_seq": "DDGun (sequence)", "ddgun_3d": "DDGun (3D)"}

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
summary_rows = []

print(f"\nrunning {len(make_models(y))} models x {len(FEATURE_SETS)} severity scores"
      f" (hairpin features present throughout)")
for model_name, estimator in make_models(y).items():
    for score, feature_cols in FEATURE_SETS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        prob = cross_val_predict(pipe, df[feature_cols], y, cv=cv,
                                 method="predict_proba", n_jobs=-1)[:, 1]
        roc_auc = roc_auc_score(y, prob)
        pd.DataFrame({"position": df["position"], TARGET: y, "pred_prob": prob}) \
            .to_csv(out_path(f"predictions_{model_name}_{score}.csv"), index=False)
        summary_rows.append({"model": model_name, "score": score,
                             "n_features": len(feature_cols), "roc_auc": roc_auc,
                             "pr_auc": average_precision_score(y, prob)})
        print(f"  {model_name:20s} {score:10s} roc_auc={roc_auc:.4f}")

summary = pd.DataFrame(summary_rows)
summary.sort_values("roc_auc", ascending=False).to_csv(
    out_path("classic_ml_summary_ddG.csv"), index=False)

Y_FLOOR = 0.5
pivot = summary.pivot(index="model", columns="score", values="roc_auc") \
               .loc[list(make_models(y).keys()), SCORE_COLS]
x = np.arange(len(pivot))
width = 0.2

fig, ax = plt.subplots(figsize=(9, 4.6))
for i, score in enumerate(SCORE_COLS):
    heights = pivot[score].values
    offset = (i - (len(SCORE_COLS) - 1) / 2) * width
    ax.bar(x + offset, heights - Y_FLOOR, width, bottom=Y_FLOOR,
           color=SCORE_COLORS[score], label=SCORE_LABELS[score])

ax.axhline(Y_FLOOR, color="grey", linestyle="--", linewidth=0.8, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(pivot.index, fontsize=7, rotation=30, ha="right")
ax.set_ylabel("ROC-AUC", fontsize=9)
ax.set_ylim(Y_FLOOR, pivot.values.max() + 0.015)
ax.legend(fontsize=7, frameon=False, ncol=4)
ax.grid(axis="y", color="lightgrey", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(out_path("bar_full_dataset_severity_scores.png"), dpi=600, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
for ax_p, score in zip(axes.ravel(), SCORE_COLS):
    curves = []
    for model_name in make_models(y).keys():
        pred = pd.read_csv(out_path(f"predictions_{model_name}_{score}.csv"))
        roc_auc = roc_auc_score(pred[TARGET], pred["pred_prob"])
        fpr, tpr, _ = roc_curve(pred[TARGET], pred["pred_prob"])
        (line,) = ax_p.plot(fpr, tpr, linewidth=0.9,
                            label=f"{model_name} (AUC={roc_auc:.3f})")
        curves.append((line, roc_auc))
    ax_p.plot([0, 1], [0, 1], "--", color="lightgrey", linewidth=0.7, zorder=0)
    ax_p.set_title(SCORE_LABELS[score], fontsize=9)
    curves.sort(key=lambda t: t[1], reverse=True)
    ax_p.legend([l for l, _ in curves], [l.get_label() for l, _ in curves],
                loc="lower right", fontsize=6.5)
for ax_p in axes[-1]:
    ax_p.set_xlabel("False positive rate")
for ax_p in axes[:, 0]:
    ax_p.set_ylabel("True positive rate")
fig.tight_layout()
fig.savefig(out_path("roc_full_dataset_severity_scores.png"), dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"\n{pivot.round(4).to_string()}")
print("\nchange in ROC-AUC relative to Grantham distance:")
print((pivot[STABILITY_COLS].sub(pivot["grantham"], axis=0)).round(4).to_string())


N_BOOT = 2000
EQUIV_MARGIN = 0.01     # ~1/3 of the spread between the best and worst model here
BOOT_SEED = 1

preds = {(m, s): pd.read_csv(out_path(f"predictions_{m}_{s}.csv"))["pred_prob"].values
         for m in make_models(y).keys() for s in SCORE_COLS}
rng = np.random.default_rng(BOOT_SEED)
boot_idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))

test_rows = []
for model_name in make_models(y).keys():
    base = preds[(model_name, "grantham")]
    for score in STABILITY_COLS:
        other = preds[(model_name, score)]
        deltas = np.array([roc_auc_score(y[i], other[i]) - roc_auc_score(y[i], base[i])
                           for i in boot_idx])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
        test_rows.append({
            "model": model_name, "score": score,
            "delta_auc": pivot.loc[model_name, score] - pivot.loc[model_name, "grantham"],
            "ci_low": lo, "ci_high": hi, "p_boot": max(p, 1 / N_BOOT),
            "equivalent": bool(lo > -EQUIV_MARGIN and hi < EQUIV_MARGIN),
        })

tests = pd.DataFrame(test_rows)
# Holm correction over the 24 comparisons
order = np.argsort(tests["p_boot"].values)
adj = np.empty(len(tests))
running = 0.0
for rank, i in enumerate(order):
    running = max(running, (len(tests) - rank) * tests["p_boot"].values[i])
    adj[i] = min(running, 1.0)
tests["p_holm"] = adj
tests.to_csv(out_path("delta_auc_vs_grantham.csv"), index=False)

print(f"\npaired bootstrap vs Grantham ({N_BOOT} resamples, equivalence margin "
      f"+/-{EQUIV_MARGIN}):")
print(tests[["model", "score", "delta_auc", "ci_low", "ci_high",
             "p_boot", "p_holm", "equivalent"]].round(4).to_string(index=False))
n_sig = int((tests.p_holm < 0.05).sum())
print(f"\n  significant after Holm correction: {n_sig}/{len(tests)}")
print(f"  equivalent at +/-{EQUIV_MARGIN}:        {int(tests.equivalent.sum())}/{len(tests)}")

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.axvspan(-EQUIV_MARGIN, EQUIV_MARGIN, color="lightgrey", alpha=0.45, zorder=0,
           label=f"equivalence margin ±{EQUIV_MARGIN}")
ax.axvline(0, color="#4a4a4a", linestyle="--", linewidth=0.8, zorder=1)

ypos, labels = [], []
cursor = 0.0
for model_name in list(make_models(y).keys())[::-1]:
    for score in STABILITY_COLS[::-1]:
        row = tests[(tests.model == model_name) & (tests.score == score)].iloc[0]
        ax.plot([row.ci_low, row.ci_high], [cursor, cursor],
                color=SCORE_COLORS[score], linewidth=1.4, solid_capstyle="butt", zorder=2)
        ax.plot(row.delta_auc, cursor, "o", color=SCORE_COLORS[score], markersize=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ypos.append(cursor)
        labels.append(f"{model_name} · {SCORE_LABELS[score]}")
        cursor += 1.0
    cursor += 0.6

ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=6.5)
ax.set_xlabel("ΔROC-AUC vs Grantham distance (paired bootstrap, 95% CI)")
ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.grid(axis="x", color="lightgrey", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(out_path("delta_auc_vs_grantham.png"), dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"\nsaved to {RESULTS_DIR}")
