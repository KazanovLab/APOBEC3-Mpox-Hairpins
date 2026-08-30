import os
import warnings
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# SHAP analysis on the matched-pairs dataset.
# LogisticRegression is used as logistic regression
# is the standard model for matched case-control designs.

IN_TSV = "matched_pairs_dataset.tsv"
RESULTS_DIR = "results_shap_matched_pairs"
os.makedirs(RESULTS_DIR, exist_ok=True)


def out_path(filename):
    return os.path.join(RESULTS_DIR, filename)


matched = pd.read_csv(IN_TSV, sep='\t')

TARGET = "any_mutation"
BASES = ["A", "C", "G", "T"]
THIRD_NT_COLS = [f"third_nt_{b}" for b in BASES]
MINUS1_NT_COLS = [f"minus1_nt_{b}" for b in BASES]
LOOP_BOUNDARY_COLS = ["loop_boundary_length", "loop_boundary_stem_length", "loop_boundary_pin_energy"]
FEATURES_WITH_LOOP = ["grantham"] + THIRD_NT_COLS + MINUS1_NT_COLS + LOOP_BOUNDARY_COLS

X = matched[FEATURES_WITH_LOOP].astype(float)
y = matched[TARGET].values
print(f"n_rows={len(X)}, n_features={len(FEATURES_WITH_LOOP)}, n_positives={int(y.sum())}")

SHAP_FEATURES = ["grantham", "third_nt", "minus1_nt"] + LOOP_BOUNDARY_COLS
CATEGORICAL_FEATURES = ["third_nt", "minus1_nt"]

FEATURE_LABELS = {
    "grantham": "Grantham distance",
    "third_nt": "3' nucleotide (TCN)",
    "minus1_nt": "5' nucleotide (NTC)",
    "loop_boundary_length": "Loop length",
    "loop_boundary_stem_length": "Stem length",
    "loop_boundary_pin_energy": "Hairpin energy (ΔG)",
}


def label(feature):
    return FEATURE_LABELS.get(feature, feature)


DISPLAY_NAMES = [label(f) for f in SHAP_FEATURES]

BAR_XLABEL = "mean(|SHAP value|)"
BAR_TITLE = f"SHAP feature importance — LogisticRegression, matched pairs, withloop, {TARGET}"


def restyle_colorbar(fig, main_ax):
    """Colorbar."""
    for ax in fig.axes:
        if ax is main_ax or not ax.get_ylabel():
            continue
        ax.set_ylabel("Feature value")
        if len(ax.get_yticks()) == 2:
            ax.set_yticklabels(["Low", "High"])

X_cat = pd.DataFrame({
    "grantham": X["grantham"].values,
    "third_nt": matched[THIRD_NT_COLS].values.argmax(axis=1).astype(float),
    "minus1_nt": matched[MINUS1_NT_COLS].values.argmax(axis=1).astype(float),
    **{c: X[c].values for c in LOOP_BOUNDARY_COLS},
})[SHAP_FEATURES]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
pipe.fit(X, y)


def predict_shap_input(data):
    """Expand the integer-coded categorical columns back to the one-hot layout
    the fitted pipeline expects, then predict."""
    cat_df = pd.DataFrame(data, columns=SHAP_FEATURES)
    full = pd.DataFrame(0.0, index=cat_df.index, columns=FEATURES_WITH_LOOP)
    for col in ["grantham"] + LOOP_BOUNDARY_COLS:
        full[col] = cat_df[col].values
    for cat_col, onehot_cols in [("third_nt", THIRD_NT_COLS), ("minus1_nt", MINUS1_NT_COLS)]:
        # masker output is continuous, so snap to the nearest valid category
        codes = np.clip(np.rint(cat_df[cat_col].values).astype(int), 0, len(onehot_cols) - 1)
        for k, onehot_col in enumerate(onehot_cols):
            full[onehot_col] = (codes == k).astype(float)
    return pipe.predict_proba(full)[:, 1]


masker = shap.maskers.Independent(X_cat, max_samples=len(X_cat))
explainer = shap.Explainer(predict_shap_input, masker, feature_names=SHAP_FEATURES)
shap_values = explainer(X_cat, silent=True)

# summary (beeswarm) plot
fig = plt.figure()
shap.summary_plot(shap_values.values, X_cat, feature_names=DISPLAY_NAMES, show=False)
ax = plt.gca()
ax.set_xlabel("SHAP value")
restyle_colorbar(fig, ax)
plt.tight_layout()
plt.savefig(out_path("shap_summary.png"), dpi=600, bbox_inches="tight")
plt.close()

# bar plot of mean |SHAP|
plt.figure()
shap.summary_plot(shap_values.values, X_cat, feature_names=DISPLAY_NAMES,
                  plot_type="bar", show=False)
plt.gca().set_xlabel(BAR_XLABEL)
if BAR_TITLE:
    plt.gca().set_title(BAR_TITLE)
plt.tight_layout()
plt.savefig(out_path("shap_importance_bar.png"), dpi=600, bbox_inches="tight")
plt.close()

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance = pd.DataFrame({"feature": SHAP_FEATURES,
                           "label": DISPLAY_NAMES,
                           "mean_abs_shap": mean_abs_shap}) \
    .sort_values("mean_abs_shap", ascending=False)
importance.to_csv(out_path("shap_importance.csv"), index=False)
print(importance.to_string(index=False))

# dependence plots for the top features
for feat in importance["feature"].head(4):
    plt.figure()
    shap.dependence_plot(feat, shap_values.values, X_cat, show=False, interaction_index=None)
    if feat in CATEGORICAL_FEATURES:
        plt.xticks(range(len(BASES)), BASES)
    plt.xlabel(label(feat))
    plt.ylabel(f"SHAP value for\n{label(feat)}")
    plt.title(f"SHAP dependence — {label(feat)}")
    plt.tight_layout()
    plt.savefig(out_path(f"shap_dependence_{feat}.png"), dpi=600, bbox_inches="tight")
    plt.close()

print(f"\nsaved SHAP plots/summary to {RESULTS_DIR}/")
