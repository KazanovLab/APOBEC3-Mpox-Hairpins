import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from deep_models_common import load_data, run_cv, RESULTS_DIR, TARGET
from cnn_full_dataset import make_cnn

# How far from the edited C does the sequence signal improve prediction
# Run on the flatten variant, not on avgpool2. 

K_VALUES = list(range(16))      # 0..15
POOLING = "flatten"
OUT_PNG = "roc_auc_vs_window_side.png"
OUT_CSV = "cnn_window_side_sweep_summary.csv"

SERIES = [
    ("Symmetric: -L..+L", "#4a4a4a", "-",  "o", lambda k: (k, k)),
    ("3' side only: 0..+L", "#c0392b", "-", "s", lambda k: (0, k)),
    ("5' side only: -L..0", "#2a78d6", "-", "^", lambda k: (k, 0)),
]


def predictions_path(left, right):
    return os.path.join(RESULTS_DIR, f"predictions_CNNflat_L{left}R{right}.csv")


def run_all():
    wanted = {bounds(k) for _, _, _, _, bounds in SERIES for k in K_VALUES}
    for left, right in sorted(wanted):
        if os.path.exists(predictions_path(left, right)):
            continue
        seq_onehot, _, X_feat, y, positions, _ = load_data(left=left, right=right)
        run_cv(f"CNNflat_L{left}R{right}", make_cnn(POOLING),
               seq_onehot, X_feat, y, positions, seq_onehot.shape[2])


def collect():
    rows = []
    for name, _, _, _, bounds in SERIES:
        for k in K_VALUES:
            left, right = bounds(k)
            path = predictions_path(left, right)
            if not os.path.exists(path):
                continue
            pred = pd.read_csv(path)
            rows.append({"series": name, "k": k, "left": left, "right": right,
                         "seq_len": left + right + 1,
                         "roc_auc": roc_auc_score(pred[TARGET], pred["pred_prob"])})
    return pd.DataFrame(rows)


def main():
    run_all()
    summary = collect()
    summary.to_csv(os.path.join(RESULTS_DIR, OUT_CSV), index=False)

    fig, ax = plt.subplots(figsize=(14.0, 4.8))
    for name, colour, style, marker, bounds in SERIES:
        sub = summary[summary.series == name].sort_values("k")
        ax.plot(sub.k, sub.roc_auc, color=colour, linestyle=style, marker=marker,
                markersize=4, linewidth=0.9, label=name)

    ax.set_xticks(K_VALUES)
    ax.set_xlabel("Window half-size, L")
    ax.set_ylabel("ROC-AUC")
    #ax.set_title(f"Which side carries the sequence signal? {TARGET}\n"
    #             f"CNN ({POOLING}), 19 engineered features held fixed", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(color="lightgrey", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, OUT_PNG)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("\n=== ROC-AUC by window extent and side ===")
    print(summary.pivot(index="k", columns="series", values="roc_auc").round(4).to_string())
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
