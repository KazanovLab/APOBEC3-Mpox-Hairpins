import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# This plot shows that the deep models' advantage over logistic
# regression comes from two sequence positions (+2,+3), not from architecture.
# The deep models are grouped by mechanism: variants that keep the
# position axis and a variant that collapses it.
# The intervals are percentile bootstrap over rows.

CLASSIC_DIR = "results_classic_ml_full_dataset"
DEEP_DIR = "results_deep_ml_full_dataset"
RESULTS_DIR = DEEP_DIR
TARGET = "any_mutation"
N_BOOT = 2000
SEED = 1

COL_CLASSIC = "#4a4a4a"
COL_KEEPS = "#2a9d5c"
COL_COLLAPSES = "#c0392b"

# label, filename, group
MODELS = [
    ("Logistic regression",
     "predictions_LogisticRegression_loopboundary.csv", "classic"),
    ("Logistic regression\nincluding +2,+3 positions",
     "predictions_LogisticRegression_loopboundary_pos23.csv", "classic"),

    ("CNN, average pooling",
     "predictions_CNN_avgpool2.csv", "keeps"),
    ("CNN, flatten",
     "predictions_CNN_flatten.csv", "keeps"),
    ("CNN, self-attention",
     "predictions_CNNSelfAttention_avgpool2.csv", "keeps"),

    ("CNN, global max pooling",
     "predictions_CNN_max.csv", "collapses"),
    ("CNN, attention pooling",
     "predictions_CNNAttention_pool.csv", "collapses"),
]

GROUPS = {
    "classic": ("Logistic regression", COL_CLASSIC),
    "keeps": ("CNN, position preserved", COL_KEEPS),
    "collapses": ("CNN, position discarded", COL_COLLAPSES),
}


def find_predictions(filename):
    for d in (CLASSIC_DIR, DEEP_DIR):
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{filename} not found in {CLASSIC_DIR} or {DEEP_DIR}")


def bootstrap_auc(y, prob, boot_idx):
    aucs = np.empty(len(boot_idx))
    for b, i in enumerate(boot_idx):
        aucs[b] = roc_auc_score(y[i], prob[i])
    return aucs


def bootstrap_delta(y, prob, reference, boot_idx):
    """Paired: both models are scored on the same resampled rows, so the shared
    row-to-row variance cancels instead of being counted twice."""
    deltas = np.empty(len(boot_idx))
    for b, i in enumerate(boot_idx):
        deltas[b] = roc_auc_score(y[i], prob[i]) - roc_auc_score(y[i], reference[i])
    return deltas


def main():
    preds = [(label, pd.read_csv(find_predictions(f)), group) for label, f, group in MODELS]
    reference = preds[0][1]["position"].values
    for label, df, _ in preds:
        assert (df["position"].values == reference).all(), f"row mismatch in {label}"

    y = preds[0][1][TARGET].values
    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))

    # panel B compares everything against logistic regression + bases at +2/+3
    reference = preds[1][1]["pred_prob"].values
    reference_auc = roc_auc_score(y, reference)

    rows = []
    for label, df, group in preds:
        prob = df["pred_prob"].values
        lo, hi = np.percentile(bootstrap_auc(y, prob, boot_idx), [2.5, 97.5])
        deltas = bootstrap_delta(y, prob, reference, boot_idx)
        d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
        rows.append({"model": label.replace("\n", " "), "group": group,
                     "roc_auc": roc_auc_score(y, prob), "ci_low": lo, "ci_high": hi,
                     "delta_vs_logreg_pos23": roc_auc_score(y, prob) - reference_auc,
                     "delta_ci_low": d_lo, "delta_ci_high": d_hi})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "paper_figure_position_effect.csv"), index=False)

    # --- plot: one row per model, groups separated by a blank slot ---
    ypos, labels, colours = [], [], []
    y_cursor = 0.0
    previous_group = None
    for (label, df, group) in preds:
        if previous_group is not None and group != previous_group:
            y_cursor += 0.8
        ypos.append(y_cursor)
        labels.append(label)
        colours.append(GROUPS[group][1])
        y_cursor += 1.0
        previous_group = group

    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    #ax.axvline(rows[0]["roc_auc"], color=COL_CLASSIC, linestyle=":", linewidth=0.8, zorder=0)
    #ax.axvline(reference_auc, color=COL_CLASSIC, linestyle="--", linewidth=0.8, zorder=0)
    for yp, row, colour in zip(ypos, rows, colours):
        # the 95% interval as a thick band rather than a hairline, with the
        # point estimate marked on top
        ax.plot([row["ci_low"], row["ci_high"]], [yp, yp], color=colour, linewidth=7,
                alpha=0.75, solid_capstyle="butt", zorder=2)
        ax.plot(row["roc_auc"], yp, "o", color=colour, markersize=7,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    #    ax.text(row["ci_high"] + 0.0015, yp, f"{row['roc_auc']:.3f}",
    #            va="center", ha="left", fontsize=8, color=colour)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("ROC-AUC, 95% bootstrap CI")
    #ax.set_title("The deep models' advantage is positional, and two bases account for it\n"
    #             f"full dataset, n=24695, {TARGET}", fontsize=9.5)
    ax.invert_yaxis()
    ax.grid(axis="x", color="lightgrey", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], color=colour, marker="o", linestyle="-", markersize=7,
                          linewidth=7, alpha=0.6, label=name)
               for name, colour in GROUPS.values()]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, "paper_figure_position_effect.png")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
