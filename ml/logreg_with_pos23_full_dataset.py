import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# LogisticRegression with one-hot encodings of the bases at +2 and +3
# in addition to the 19 features (27 total).

IN_TSV = "ml_table_all.tsv"
RESULTS_DIR = "results_classic_ml_full_dataset"
TARGET = "any_mutation"
N_SPLITS = 10
SEED = 1
BASES = ["A", "C", "G", "T"]
CENTER_IDX = 50

os.makedirs(RESULTS_DIR, exist_ok=True)


def out_path(filename):
    return os.path.join(RESULTS_DIR, filename)


DEEP_DIR = "results_deep_ml_full_dataset"


def find_predictions(filename):
    for d in (RESULTS_DIR, DEEP_DIR):
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{filename} not found in {RESULTS_DIR} or {DEEP_DIR}")


df = pd.read_csv(IN_TSV, sep='\t')
context = df["context_pm50"].str.upper()
df = pd.get_dummies(df, columns=["third_nt", "minus1_nt"], dtype="uint8")

LOOP_BOUNDARY_COLS = [c for c in df.columns
                      if (c.startswith("loop_boundary_") or c == "in_loop_boundary")
                      and "sequence" not in c]
df[LOOP_BOUNDARY_COLS] = df[LOOP_BOUNDARY_COLS].fillna(0.0)

THIRD_NT_COLS = [f"third_nt_{b}" for b in BASES]
MINUS1_NT_COLS = [f"minus1_nt_{b}" for b in BASES]
FEATURES_BASE = ["grantham"] + THIRD_NT_COLS + MINUS1_NT_COLS + LOOP_BOUNDARY_COLS

POS23_COLS = []
for offset in (2, 3):
    base_at_offset = context.str[CENTER_IDX + offset]
    for b in BASES:
        col = f"plus{offset}_nt_{b}"
        df[col] = (base_at_offset == b).astype("uint8")
        POS23_COLS.append(col)

FEATURE_SETS = {
    "loopboundary": FEATURES_BASE,
    "loopboundary_pos23": FEATURES_BASE + POS23_COLS,
}

y = df[TARGET].values
print(f"n_rows={len(df)}, n_positives={int(y.sum())} ({y.mean():.1%})")
print(f"added positional features ({len(POS23_COLS)}): {POS23_COLS}")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for tag, feature_cols in FEATURE_SETS.items():
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    prob = cross_val_predict(pipe, df[feature_cols], y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    pd.DataFrame({"position": df["position"], TARGET: y, "pred_prob": prob}) \
        .to_csv(out_path(f"predictions_LogisticRegression_{tag}.csv"), index=False)
    print(f"  LogisticRegression {tag:20s} ({len(feature_cols):2d} feats) roc_auc={roc_auc_score(y, prob):.4f}")

# ROC comparison
# SOURCES = [
#     ("LogisticRegression", "predictions_LogisticRegression_loopboundary.csv", "#2a78d6", "--"),
#     ("LogisticRegression + nt at +2/+3", "predictions_LogisticRegression_loopboundary_pos23.csv", "#2a78d6", "-"),
#     ("MultiInputCNN (avgpool2)", "predictions_MultiInputCNN_avgpool2.csv", "#2a9d5c", "--"),
#     ("MultiInputCNN + attention pooling", "predictions_MultiInputCNNAttention_pool.csv", "#c98a1a", "-."),
#     ("MultiInputTransformer", "predictions_MultiInputTransformer.csv", "#7b2cbf", "-"),
# ]

# curves = []
# summary_rows = []
# fig, ax = plt.subplots(figsize=(6.5, 5.5))
# for label, filename, color, style in SOURCES:
#     pred = pd.read_csv(find_predictions(filename))
#     roc_auc = roc_auc_score(pred[TARGET], pred["pred_prob"])
#     fpr, tpr, _ = roc_curve(pred[TARGET], pred["pred_prob"])
#     (line,) = ax.plot(fpr, tpr, color=color, linestyle=style, linewidth=0.9,
#                        label=f"{label} (AUC={roc_auc:.3f})")
#     curves.append((line, roc_auc))
#     summary_rows.append({"model": label, "roc_auc": roc_auc})

# ax.plot([0, 1], [0, 1], "--", color="lightgrey", linewidth=0.7, zorder=0)
# ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
# ax.set_title(f"ROC — full dataset, {TARGET}\n"
#              "does handing the classic model positions +2/+3 close the gap? (10-fold OOF)")
# curves.sort(key=lambda t: t[1], reverse=True)
# ax.legend([l for l, _ in curves], [l.get_label() for l, _ in curves], loc="lower right", fontsize=8)
# fig.tight_layout()
# fig.savefig(os.path.join(DEEP_DIR, "roc_logreg_pos23_vs_deep_models.png"), dpi=600, bbox_inches="tight")
# plt.close(fig)

# summary = pd.DataFrame(summary_rows).sort_values("roc_auc", ascending=False)
# summary.to_csv(os.path.join(DEEP_DIR, "summary_logreg_pos23_vs_deep_models.csv"), index=False)
# print(f"\n{summary.to_string(index=False)}")
# print(f"\nsaved {DEEP_DIR}/roc_logreg_pos23_vs_deep_models.png")
