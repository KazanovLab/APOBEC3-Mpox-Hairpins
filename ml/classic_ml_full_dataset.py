import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Full dataset (ml_table_all.tsv, 24695 rows),
# the three best-performing classic models, with vs without loop-boundary
# features. LOO-CV is infeasible, so 10-fold stratified CV is used.
# Only the numeric loop_boundary_* features are used.

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

MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "LDA": lambda: LinearDiscriminantAnalysis(),
    "NaiveBayes": lambda: GaussianNB(),
}

COLORS = {"LogisticRegression": "#2a78d6", "LDA": "#e34948", "NaiveBayes": "#2a9d5c"}
STYLES = {"noloop": "--", "loopboundary": "-"}
LABELS = {"noloop": "without hairpin features", "loopboundary": "with hairpin features"}

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
summary_rows = []
curves = []  # (line, roc_auc) -- legend sorted by AUC descending below

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for model_name, make_estimator in MODELS.items():
    for tag, feature_cols in FEATURE_SETS.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_estimator())])
        prob = cross_val_predict(pipe, df[feature_cols], y, cv=cv,
                                  method="predict_proba", n_jobs=-1)[:, 1]
        roc_auc = roc_auc_score(y, prob)
        fpr, tpr, _ = roc_curve(y, prob)

        (line,) = ax.plot(fpr, tpr, color=COLORS[model_name], linestyle=STYLES[tag], linewidth=0.9,
                           label=f"{model_name}, {LABELS[tag]} (AUC={roc_auc:.3f})")
        curves.append((line, roc_auc))

        pd.DataFrame({"position": df["position"], TARGET: y, "pred_prob": prob}) \
            .to_csv(out_path(f"predictions_{model_name}_{tag}.csv"), index=False)
        summary_rows.append({"model": model_name, "feature_set": tag,
                              "n_features": len(feature_cols), "roc_auc": roc_auc})
        print(f"  {model_name:20s} {tag:9s} roc_auc={roc_auc:.4f}")

ax.plot([0, 1], [0, 1], "--", color="lightgrey", linewidth=0.7, zorder=0)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
#ax.set_title(f"ROC — full dataset ({len(df)} rows), {TARGET}\n"
#             f"with vs without loop boundary features ({N_SPLITS}-fold OOF)")
curves.sort(key=lambda t: t[1], reverse=True)
ax.legend([l for l, _ in curves], [l.get_label() for l, _ in curves], loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(out_path("roc_full_dataset_loopboundary_vs_noloop.png"), dpi=600, bbox_inches="tight")
plt.close(fig)

summary = pd.DataFrame(summary_rows).sort_values("roc_auc", ascending=False)
summary.to_csv(out_path("classic_ml_summary_full_dataset.csv"), index=False)
print(f"\n{summary.to_string(index=False)}")
print(f"\nsaved to {RESULTS_DIR}/")
