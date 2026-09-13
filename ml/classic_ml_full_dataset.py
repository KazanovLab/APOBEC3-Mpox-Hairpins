import os
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

# Full dataset (ml_table_all.tsv, 24695 rows), with vs without loop-boundary
# features, across the same models as classic_ml_matched_pairs.py. LOO-CV is
# infeasible at this size, so 10-fold stratified CV is used.
# Only the numeric loop_boundary_* features are used.

set_config(transform_output="pandas")   # keeps feature names through the scaler

IN_TSV = "ml_table_all.tsv"
RESULTS_DIR = "results_classic_ml_full_dataset"
TARGET = "any_mutation"
N_SPLITS = 10
SEED = 1
os.makedirs(RESULTS_DIR, exist_ok=True)


def out_path(filename):
    return os.path.join(RESULTS_DIR, filename)


df = pd.read_csv(IN_TSV, sep='\t')
df = pd.get_dummies(df, columns=["third_nt", "minus1_nt"], dtype="uint8")

LOOP_BOUNDARY_COLS = [c for c in df.columns
                      if (c.startswith("loop_boundary_") or c == "in_loop_boundary")
                      and "sequence" not in c]
df[LOOP_BOUNDARY_COLS] = df[LOOP_BOUNDARY_COLS].fillna(0.0)

BASES = ["A", "C", "G", "T"]
THIRD_NT_COLS = [f"third_nt_{b}" for b in BASES]
MINUS1_NT_COLS = [f"minus1_nt_{b}" for b in BASES]

FEATURES_NO_LOOP = ["grantham"] + THIRD_NT_COLS + MINUS1_NT_COLS
FEATURES_LOOP_BOUNDARY = FEATURES_NO_LOOP + LOOP_BOUNDARY_COLS

FEATURE_SETS = {
    "noloop": FEATURES_NO_LOOP,
    "loopboundary": FEATURES_LOOP_BOUNDARY,
}

y = df[TARGET].values
print(f"n_rows={len(df)}, n_positives={int(y.sum())} ({y.mean():.1%})")
print(f"loop_boundary features ({len(LOOP_BOUNDARY_COLS)}): {LOOP_BOUNDARY_COLS}")


def make_models(y_true):
    """Same set as classic_ml_matched_pairs.py (DecisionTree and KNeighbors are
    left out there too). Class imbalance is handled per library: class_weight
    for sklearn, scale_pos_weight for XGBoost, auto_class_weights for CatBoost."""
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


COLORS = {
    "LogisticRegression": "#2a78d6",
    "NaiveBayes": "#2a9d5c",
    "SVM": "#e34948",
    "LDA": "#7b2cbf",
    "RandomForest": "#c98a1a",
    "LightGBM": "#17a2b8",
    "XGBoost": "#d1495b",
    "CatBoost": "#4a4a4a",
}
STYLES = {"noloop": "--", "loopboundary": "-"}
LABELS = {"noloop": "without hairpin features", "loopboundary": "with hairpin features"}

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
summary_rows = []
curves = []  # (line, roc_auc) -- legend sorted by AUC descending below

fig, ax = plt.subplots(figsize=(7.5, 6.5))
for model_name, estimator in make_models(y).items():
    for tag, feature_cols in FEATURE_SETS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        prob = cross_val_predict(pipe, df[feature_cols], y, cv=cv,
                                 method="predict_proba", n_jobs=-1)[:, 1]
        roc_auc = roc_auc_score(y, prob)
        pr_auc = average_precision_score(y, prob)
        fpr, tpr, _ = roc_curve(y, prob)

        (line,) = ax.plot(fpr, tpr, color=COLORS[model_name], linestyle=STYLES[tag], linewidth=0.9,
                          label=f"{model_name}, {LABELS[tag]} (AUC={roc_auc:.3f})")
        curves.append((line, roc_auc))

        pd.DataFrame({"position": df["position"], TARGET: y, "pred_prob": prob}) \
            .to_csv(out_path(f"predictions_{model_name}_{tag}.csv"), index=False)
        summary_rows.append({"model": model_name, "feature_set": tag,
                             "n_features": len(feature_cols),
                             "roc_auc": roc_auc, "pr_auc": pr_auc})
        print(f"  {model_name:20s} {tag:13s} roc_auc={roc_auc:.4f}  pr_auc={pr_auc:.4f}")

ax.plot([0, 1], [0, 1], "--", color="lightgrey", linewidth=0.7, zorder=0)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
#ax.set_title(f"ROC — full dataset ({len(df)} rows), {TARGET}\n"
#             f"with vs without loop boundary features ({N_SPLITS}-fold OOF)")
curves.sort(key=lambda t: t[1], reverse=True)
ax.legend([l for l, _ in curves], [l.get_label() for l, _ in curves], loc="lower right", fontsize=7)
fig.tight_layout()
fig.savefig(out_path("roc_full_dataset_loopboundary_vs_noloop.png"), dpi=600, bbox_inches="tight")
plt.close(fig)

summary = pd.DataFrame(summary_rows).sort_values("roc_auc", ascending=False)
summary.to_csv(out_path("classic_ml_summary_full_dataset.csv"), index=False)
print(f"\n{summary.to_string(index=False)}")

# --- grouped bar chart: the two feature sets side by side, one pair per model ---
# The y axis starts at 0.5 rather than 0: for ROC-AUC the no-information point is
# chance, not zero, so a bar from 0 would spend most of its length on a range no
# classifier can fall into. The axis break is marked by the dashed chance line.
# same blue/red pair as the matched-pairs figures, where the variant carrying
# the hairpin information is the red one
BAR_COLORS = {"noloop": "#2a78d6", "loopboundary": "#e34948"}
Y_FLOOR = 0.5

pivot = (summary.pivot(index="model", columns="feature_set", values="roc_auc")
                .loc[list(make_models(y).keys())])
x = range(len(pivot))
width = 0.38

fig, ax = plt.subplots(figsize=(5.2, 4.6))
# the two bars of a pair are often within 0.001 of each other, so their labels
# would sit at the same height and collide; the second series is nudged 1 mm to
# the right. bar_label offsets are in points, hence 72/25.4 per mm.
MM = 72 / 25.4
LABEL_DX = {"noloop": 0.0, "loopboundary": MM}
for offset, tag in ((-width / 2, "noloop"), (width / 2, "loopboundary")):
    heights = pivot[tag].values
    bars = ax.bar([i + offset for i in x], heights - Y_FLOOR, width, bottom=Y_FLOOR,
                  color=BAR_COLORS[tag], label=LABELS[tag].capitalize())
    texts = ax.bar_label(bars, labels=[f"{h:.3f}" for h in heights], padding=2, fontsize=5)
    for t in texts:
        t.set_position((LABEL_DX[tag], 2))

ax.axhline(Y_FLOOR, color="grey", linestyle="--", linewidth=0.8, zorder=0)
ax.set_xticks(list(x))
ax.set_xticklabels(pivot.index, fontsize=7, rotation=30, ha="right")
ax.set_ylabel("ROC-AUC", fontsize=9)
ax.set_ylim(Y_FLOOR, max(pivot.max()) + 0.02)
ax.legend(fontsize=7, frameon=False)
ax.grid(axis="y", color="lightgrey", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(out_path("bar_full_dataset_loopboundary_vs_noloop.png"), dpi=600, bbox_inches="tight")
plt.close(fig)

delta = (pivot["loopboundary"] - pivot["noloop"]).sort_values()
print("\nchange in ROC-AUC from adding the hairpin features:")
print(delta.round(4).to_string())
print(f"\nsaved to {RESULTS_DIR}/")
