import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from paper_figure_position_effect import RESULTS_DIR, TARGET, find_predictions

# ROC curves for the models kept for the paper, window +/-15nt. The older
# roc_classic_vs_deep_models.py keeps all ten models and is left alone.
#
# One colour per model, solid lines throughout. Logistic regression with the
# bases at +2/+3 is not on this figure -- it belongs with the positional
# analysis, not with the architecture comparison.

OUT_PNG = "roc_selected_models.png"
OUT_CSV = "roc_selected_models.csv"
DPI = 600

# label, filename, colour
MODELS = [
    ("Logistic regression (19 engineered features)",
     "predictions_LogisticRegression_loopboundary.csv", "#4a4a4a"),
    ("CNN, average pooling",
     "predictions_CNN_avgpool2.csv", "#2a9d5c"),
    # kept on this figure because the window-side sweep runs on flatten:
    # avgpool2 silently drops the rightmost position at narrow windows
    ("CNN, flatten",
     "predictions_CNN_flatten.csv", "#17a2b8"),
    ("CNN, self-attention",
     "predictions_CNNSelfAttention_avgpool2.csv", "#7b2cbf"),
    ("CNN, global max pooling",
     "predictions_CNN_max.csv", "#c0392b"),
    ("CNN, attention pooling",
     "predictions_CNNAttention_pool.csv", "#e08a1e"),
]


def main():
    curves, rows = [], []
    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    for label, filename, colour in MODELS:
        pred = pd.read_csv(find_predictions(filename))
        y, prob = pred[TARGET].values, pred["pred_prob"].values
        roc_auc = roc_auc_score(y, prob)
        fpr, tpr, _ = roc_curve(y, prob)
        (line,) = ax.plot(fpr, tpr, color=colour, linewidth=0.9,
                          label=f"{label} (AUC={roc_auc:.3f})")
        curves.append((line, roc_auc))
        rows.append({"model": label, "roc_auc": roc_auc})

    ax.plot([0, 1], [0, 1], "--", color="lightgrey", linewidth=0.7, zorder=0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    #ax.set_title(f"ROC — full dataset, n=24695, {TARGET}\n"
    #             "10-fold out-of-fold predictions, window +/-15nt",
    #             fontsize=9.5)
    curves.sort(key=lambda t: t[1], reverse=True)
    ax.legend([l for l, _ in curves], [l.get_label() for l, _ in curves],
              loc="lower right", fontsize=7.5)
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, OUT_PNG)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    summary.to_csv(os.path.join(RESULTS_DIR, OUT_CSV), index=False)
    print(summary.to_string(index=False))
    print(f"\nsaved {out} ({DPI} dpi)")


if __name__ == "__main__":
    main()
