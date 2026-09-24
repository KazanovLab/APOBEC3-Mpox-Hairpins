import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The same Grantham-versus-stability comparison as classic_ml_ddG_scores.py
# on the deep models of ml/roc_selected_models.py instead of the classic ones.

HERE = Path(__file__).resolve().parent
ML_DIR = HERE.parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

import deep_models_common as dmc
from deep_models_common import run_cv, BASES, BASE_IDX, N_TOKENS
from cnn_full_dataset import make_cnn
from cnn_attention_full_dataset import make_attention_cnn
from cnn_self_attention_full_dataset import make_self_attention_cnn

IN_TSV = HERE / "ml_table_all_with_ddG.tsv"
STATUS_TSV = HERE.parent / "ddG_results.tsv"
RESULTS_DIR = HERE / "results_deep_ml_ddG"
TARGET = "any_mutation"
WINDOW_HALF = 15
N_SPLITS = 10
SEED = 1
CENTER_IDX = 50
os.makedirs(RESULTS_DIR, exist_ok=True)

dmc.RESULTS_DIR = str(RESULTS_DIR)


def out_path(filename):
    return os.path.join(RESULTS_DIR, filename)


STABILITY_COLS = ["ddG", "ddgun_seq", "ddgun_3d"]
SCORE_COLS = ["grantham"] + STABILITY_COLS
SCORE_COLORS = {"grantham": "#7b2cbf", "ddG": "#2a78d6",
                "ddgun_seq": "#e34948", "ddgun_3d": "#2a9d5c"}
SCORE_LABELS = {"grantham": "Grantham distance", "ddG": "FoldX (ΔΔG)",
                "ddgun_seq": "DDGun (sequence)", "ddgun_3d": "DDGun (3D)"}


df = pd.read_csv(IN_TSV, sep='\t')
status = pd.read_csv(STATUS_TSV, sep='\t')[["position", "status"]]
df = df.merge(status, on="position", how="left", validate="one_to_one")
context = df["context_pm50"].str.upper()

is_nonsense = (df["status"] == "nonsense").values
for col in STABILITY_COLS:
    col_max = df[col].max()
    df.loc[df[col].isna() & is_nonsense, col] = col_max
    df[col] = df[col].fillna(0.0)

df = pd.get_dummies(df, columns=["third_nt", "minus1_nt"], dtype="uint8")
THIRD_NT_COLS = [f"third_nt_{b}" for b in BASES]
MINUS1_NT_COLS = [f"minus1_nt_{b}" for b in BASES]
LOOP_BOUNDARY_COLS = [c for c in df.columns
                      if (c.startswith("loop_boundary_") or c == "in_loop_boundary")
                      and "sequence" not in c]
df[LOOP_BOUNDARY_COLS] = df[LOOP_BOUNDARY_COLS].fillna(0.0)

POS23_COLS = []
for offset in (2, 3):
    base_at_offset = context.str[CENTER_IDX + offset]
    for b in BASES:
        col = f"plus{offset}_nt_{b}"
        df[col] = (base_at_offset == b).astype("uint8")
        POS23_COLS.append(col)

y = df[TARGET].values.astype(np.float32)
positions = df["position"].values
print(f"n_rows={len(df)}, n_positives={int(y.sum())} ({y.mean():.1%})")

seqs = context.str[CENTER_IDX - WINDOW_HALF: CENTER_IDX + WINDOW_HALF + 1]
seq_idx = np.stack([np.array([BASE_IDX[ch] for ch in s], dtype=np.int64) for s in seqs])
onehot = np.zeros((len(seqs), N_TOKENS, 2 * WINDOW_HALF + 1), dtype=np.float32)
rows = np.arange(len(seqs))[:, None]
cols = np.arange(seq_idx.shape[1])[None, :]
onehot[rows, seq_idx, cols] = 1.0
seq_onehot = onehot[:, :len(BASES), :]
SEQ_LEN = seq_onehot.shape[2]


def engineered(score):
    """The 19 engineered features with `score` in place of grantham."""
    return [score] + THIRD_NT_COLS + MINUS1_NT_COLS + LOOP_BOUNDARY_COLS


DEEP_MODELS = [
    ("CNN, average pooling", "onehot", lambda: make_cnn("avgpool2")),
    ("CNN, flatten", "onehot", lambda: make_cnn("flatten")),
    ("CNN, self-attention", "onehot", lambda: make_self_attention_cnn("avgpool2")),
    ("CNN, global max pooling", "onehot", lambda: make_cnn("max")),
    ("CNN, attention pooling", "onehot", lambda: make_attention_cnn("pool")),
]
LOGREG_LABEL = "Logistic regression + bases at +2/+3"
MODEL_ORDER = [LOGREG_LABEL] + [label for label, _, _ in DEEP_MODELS]


def tag(label, score):
    slug = (label.replace(" + ", "_").replace(", ", "_").replace(" ", "")
                 .replace("(", "").replace(")", "").replace("+", "p").replace("/", ""))
    return f"{slug}_{score}"


cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
summary_rows = []

for score in SCORE_COLS:
    feature_cols = engineered(score)
    X_feat = df[feature_cols].values.astype(np.float32)

    name = tag(LOGREG_LABEL, score)
    path = out_path(f"predictions_{name}.csv")
    if not os.path.exists(path):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        prob = cross_val_predict(pipe, df[feature_cols + POS23_COLS], y, cv=cv,
                                 method="predict_proba", n_jobs=-1)[:, 1]
        pd.DataFrame({"position": positions, TARGET: y.astype(int), "pred_prob": prob}) \
            .to_csv(path, index=False)
        print(f"  {LOGREG_LABEL:36s} {score:10s} roc_auc={roc_auc_score(y, prob):.4f}")

    for label, encoding, factory in DEEP_MODELS:
        name = tag(label, score)
        if os.path.exists(out_path(f"predictions_{name}.csv")):
            continue
        seq_input = seq_idx if encoding == "idx" else seq_onehot
        run_cv(name, factory(), seq_input, X_feat, y, positions, SEQ_LEN)

for score in SCORE_COLS:
    for label in MODEL_ORDER:
        pred = pd.read_csv(out_path(f"predictions_{tag(label, score)}.csv"))
        summary_rows.append({"model": label, "score": score,
                             "roc_auc": roc_auc_score(pred[TARGET], pred["pred_prob"])})

summary = pd.DataFrame(summary_rows)
summary.sort_values("roc_auc", ascending=False).to_csv(
    out_path("deep_ml_summary_ddG.csv"), index=False)

pivot = summary.pivot(index="model", columns="score", values="roc_auc").loc[MODEL_ORDER, SCORE_COLS]
print(f"\n{pivot.round(4).to_string()}")
print("\nchange in ROC-AUC relative to Grantham distance:")
print(pivot[STABILITY_COLS].sub(pivot["grantham"], axis=0).round(4).to_string())


Y_FLOOR = 0.5
x = np.arange(len(pivot))
width = 0.2

fig, ax = plt.subplots(figsize=(9, 4.6))
for i, score in enumerate(SCORE_COLS):
    offset = (i - (len(SCORE_COLS) - 1) / 2) * width
    ax.bar(x + offset, pivot[score].values - Y_FLOOR, width, bottom=Y_FLOOR,
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
fig.savefig(out_path("bar_deep_severity_scores.png"), dpi=600, bbox_inches="tight")
plt.close(fig)


N_BOOT = 2000
EQUIV_MARGIN = 0.01
preds = {(m, s): pd.read_csv(out_path(f"predictions_{tag(m, s)}.csv"))["pred_prob"].values
         for m in MODEL_ORDER for s in SCORE_COLS}
y_int = y.astype(int)
boot_idx = np.random.default_rng(SEED).integers(0, len(y), size=(N_BOOT, len(y)))

test_rows = []
for model_name in MODEL_ORDER:
    base = preds[(model_name, "grantham")]
    for score in STABILITY_COLS:
        other = preds[(model_name, score)]
        deltas = np.array([roc_auc_score(y_int[i], other[i]) - roc_auc_score(y_int[i], base[i])
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
order = np.argsort(tests["p_boot"].values)          # Holm
adj, running = np.empty(len(tests)), 0.0
for rank, i in enumerate(order):
    running = max(running, (len(tests) - rank) * tests["p_boot"].values[i])
    adj[i] = min(running, 1.0)
tests["p_holm"] = adj
tests.to_csv(out_path("delta_auc_vs_grantham_deep.csv"), index=False)

print(f"\npaired bootstrap vs Grantham ({N_BOOT} resamples, margin +/-{EQUIV_MARGIN}):")
print(tests[["model", "score", "delta_auc", "ci_low", "ci_high",
             "p_boot", "p_holm", "equivalent"]].round(4).to_string(index=False))
print(f"\n  significant after Holm: {int((tests.p_holm < 0.05).sum())}/{len(tests)}")
print(f"  equivalent at +/-{EQUIV_MARGIN}:  {int(tests.equivalent.sum())}/{len(tests)}")

fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.axvspan(-EQUIV_MARGIN, EQUIV_MARGIN, color="lightgrey", alpha=0.45, zorder=0,
           label=f"equivalence margin ±{EQUIV_MARGIN}")
ax.axvline(0, color="#4a4a4a", linestyle="--", linewidth=0.8, zorder=1)
ypos, labels, cursor = [], [], 0.0
for model_name in MODEL_ORDER[::-1]:
    for score in STABILITY_COLS[::-1]:
        row = tests[(tests.model == model_name) & (tests.score == score)].iloc[0]
        ax.plot([row.ci_low, row.ci_high], [cursor, cursor], color=SCORE_COLORS[score],
                linewidth=1.4, solid_capstyle="butt", zorder=2)
        ax.plot(row.delta_auc, cursor, "o", color=SCORE_COLORS[score], markersize=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ypos.append(cursor)
        labels.append(f"{model_name} · {SCORE_LABELS[score]}")
        cursor += 1.0
    cursor += 0.6
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=6)
ax.set_xlabel("ΔROC-AUC vs Grantham distance (paired bootstrap, 95% CI)")
ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.grid(axis="x", color="lightgrey", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(out_path("delta_auc_vs_grantham_deep.png"), dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"\nsaved to {RESULTS_DIR}")
