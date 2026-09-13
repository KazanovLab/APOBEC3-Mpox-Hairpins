import os
import warnings
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# SHAP with RANDOM controls instead of confound-matched ones.
#
# shap_analysis_matched_pairs.py pairs each hairpin row with a control that has
# the same grantham/third_nt/minus1_nt by construction. That makes those three
# features useless for separating the two groups, so their SHAP values are
# suppressed by the design rather than by the data. Here the controls are drawn
# at random, which lets the context features contribute whatever they actually
# contribute.
#
# One random draw is noisy, so the draw is repeated N_ITER times. Each iteration
# ranks the features by mean(|SHAP|); the ranks are averaged across iterations to
# give the final feature order. The beeswarm then shows the points of a single
# representative iteration -- the one whose own ranking is closest to the
# averaged one -- so the spread and the colouring stay those of a real fit
# rather than an average over different rows.
#
# Everything else (features, pipeline, masker, the categorical collapse/expand
# trick) is kept identical to the matched-pairs script so the two are comparable.

IN_TSV = "ml_table_all.tsv"
RESULTS_DIR = "results_shap_random_controls"
TARGET = "any_mutation"
N_ITER = 100
N_CONTROLS = 71
SEED = 1

# loop_boundary_pin_energy is NaN wherever there is no hairpin. It is a folding
# free energy, so negative means a stable hairpin and the cases run from -5.3 to
# +0.1. Filling "no hairpin" with 0.0 would place controls inside the upper end
# of the case range; a large positive sentinel places them where an absent
# hairpin belongs -- past the least stable case.
NO_HAIRPIN_ENERGY = 10.0
os.makedirs(RESULTS_DIR, exist_ok=True)


def out_path(filename):
    return os.path.join(RESULTS_DIR, filename)


BASES = ["A", "C", "G", "T"]
THIRD_NT_COLS = [f"third_nt_{b}" for b in BASES]
MINUS1_NT_COLS = [f"minus1_nt_{b}" for b in BASES]
LOOP_BOUNDARY_COLS = ["loop_boundary_length", "loop_boundary_stem_length", "loop_boundary_pin_energy"]

FEATURES_WITH_LOOP = ["grantham"] + THIRD_NT_COLS + MINUS1_NT_COLS + LOOP_BOUNDARY_COLS

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


def restyle_colorbar(fig, main_ax):
    """Colorbar."""
    for ax in fig.axes:
        if ax is main_ax or not ax.get_ylabel():
            continue
        ax.set_ylabel("Feature value")
        if len(ax.get_yticks()) == 2:
            ax.set_yticklabels(["Low", "High"])


# --- data ---------------------------------------------------------------
df = pd.read_csv(IN_TSV, sep='\t')

# same core definition as build_matched_pairs_dataset.py
core_mask = (((df.loop_boundary_length == 3) & (df.loop_boundary_stem_length == 7)) |
             ((df.loop_boundary_length == 3) & (df.loop_boundary_stem_length == 6)) |
             ((df.loop_boundary_length == 3) & (df.loop_boundary_stem_length == 5)) |
             ((df.loop_boundary_length == 4) & (df.loop_boundary_stem_length == 6)) |
             ((df.loop_boundary_length == 4) & (df.loop_boundary_stem_length == 5)))
core = df.loc[core_mask].sort_values("position").reset_index(drop=True)
pool = df.loc[(~core_mask) & (df.in_loop_boundary == 0)].reset_index(drop=True)
print(f"core rows: {len(core)}")
print(f"control pool (no loop boundary at all): {len(pool)}")


def build_iteration_frame(controls):
    """Core + sampled controls, one-hot encoded exactly as the matched-pairs
    dataset is. third_nt/minus1_nt are cast to a fixed-category dtype first:
    with random controls a base can be missing from the 142 rows, and plain
    get_dummies would then emit fewer columns and silently shift the integer
    codes that X_cat depends on."""
    d = pd.concat([core, controls], ignore_index=True)
    for col in ("third_nt", "minus1_nt"):
        d[col] = pd.Categorical(d[col], categories=BASES)
    d = pd.get_dummies(d, columns=["third_nt", "minus1_nt"], dtype="uint8")
    # pin_energy is NaN for every row outside a loop boundary, so this has to
    # run after the concat. Length and stem length are already literal 0 there.
    d["loop_boundary_pin_energy"] = d["loop_boundary_pin_energy"].fillna(NO_HAIRPIN_ENERGY)
    d[LOOP_BOUNDARY_COLS] = d[LOOP_BOUNDARY_COLS].fillna(0.0)
    missing = [c for c in FEATURES_WITH_LOOP if c not in d.columns]
    assert not missing, f"missing columns after get_dummies: {missing}"
    return d


def run_iteration(controls):
    """Fit the pipeline and return (shap values, X_cat, y) for the 142 rows."""
    d = build_iteration_frame(controls)
    X = d[FEATURES_WITH_LOOP].astype(float)
    y = d[TARGET].values
    if len(np.unique(y)) < 2:
        return None

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X, y)

    X_cat = pd.DataFrame({
        "grantham": X["grantham"].values,
        "third_nt": d[THIRD_NT_COLS].values.argmax(axis=1).astype(float),
        "minus1_nt": d[MINUS1_NT_COLS].values.argmax(axis=1).astype(float),
        **{c: X[c].values for c in LOOP_BOUNDARY_COLS},
    })[SHAP_FEATURES]

    def predict(data, pipe=pipe):
        """Expand the integer-coded categorical columns back to the one-hot
        layout the fitted pipeline expects, then predict."""
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
    explainer = shap.Explainer(predict, masker, feature_names=SHAP_FEATURES)
    return explainer(X_cat, silent=True).values, X_cat, y


# --- the loop -----------------------------------------------------------
seeds = np.random.SeedSequence(SEED).spawn(N_ITER)
all_values, all_X_cat, importances, n_positives = [], [], [], []

for i, child in enumerate(seeds):
    controls = pool.sample(n=N_CONTROLS, replace=False,
                           random_state=np.random.default_rng(child).integers(2**31))
    result = run_iteration(controls)
    if result is None:
        print(f"  iteration {i}: single-class target, skipped")
        continue
    values, X_cat, y = result
    all_values.append(values)
    all_X_cat.append(X_cat)
    importances.append(np.abs(values).mean(axis=0))
    n_positives.append(int(y.sum()))
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{N_ITER} iterations done")

R = np.array(importances)                       # (n_iter, n_features)
n_done = len(R)
print(f"\n{n_done} iterations, {np.mean(n_positives):.1f} positives per 142 rows on average")


# --- aggregation --------------------------------------------------------
# rank 1 = most important within an iteration
ranks = np.argsort(np.argsort(-R, axis=1), axis=1) + 1
mean_rank = ranks.mean(axis=0)

shares = R / R.sum(axis=1, keepdims=True)
RULES = {
    "mean_rank": np.argsort(mean_rank),
    "mean_importance": np.argsort(-R.mean(axis=0)),
    "mean_share": np.argsort(-shares.mean(axis=0)),
    "median_importance": np.argsort(-np.median(R, axis=0)),
}
print("\nfeature order under each aggregation rule:")
for name, order in RULES.items():
    print(f"  {name:18s} {' > '.join(SHAP_FEATURES[i] for i in order)}")
if len({tuple(o) for o in RULES.values()}) > 1:
    print("  WARNING: the rules disagree -- the ordering is not stable")
else:
    print("  all four rules agree")

# mean rank is the headline rule: it is invariant to the per-iteration scale of
# the SHAP values, which drifts because the model is refit on every draw
idx = RULES["mean_rank"]
avg_positions = np.empty(len(SHAP_FEATURES), dtype=int)
avg_positions[idx] = np.arange(len(SHAP_FEATURES))


# --- representative iteration -------------------------------------------
footrule, taus = [], []
for r in R:
    order = np.argsort(-r)
    pos = np.empty(len(SHAP_FEATURES), dtype=int)
    pos[order] = np.arange(len(SHAP_FEATURES))
    footrule.append(int(np.abs(pos - avg_positions).sum()))
    taus.append(kendalltau(pos, avg_positions).statistic)
footrule = np.array(footrule)

best = np.flatnonzero(footrule == footrule.min())
# tie-break on the shape of the importance profile, not its magnitude
l1 = np.abs(shares[best] - shares.mean(axis=0)).sum(axis=1)
rep = int(best[np.argmin(l1)])
n_exact = int((footrule == 0).sum())

print(f"\nrepresentative iteration: {rep}"
      f"  (footrule={footrule[rep]}, kendall_tau={taus[rep]:.3f},"
      f" exact={'yes' if footrule[rep] == 0 else 'no'})")
print(f"  iterations reproducing the averaged order exactly: {n_exact}/{n_done}")
if footrule[rep] > 4:
    print("  WARNING: even the closest iteration differs a lot from the averaged order;"
          " the representative-iteration figure is not a faithful summary")


# --- tables -------------------------------------------------------------
pd.DataFrame(R, columns=SHAP_FEATURES).to_csv(
    out_path("shap_importance_iterations.csv"), index=False)

summary = pd.DataFrame({
    "feature": SHAP_FEATURES,
    "label": DISPLAY_NAMES,
    "mean_abs_shap": R.mean(axis=0),
    "sd": R.std(axis=0, ddof=1),
    "cv": R.std(axis=0, ddof=1) / R.mean(axis=0),
    "median": np.median(R, axis=0),
    "pct2_5": np.percentile(R, 2.5, axis=0),
    "pct97_5": np.percentile(R, 97.5, axis=0),
    "mean_rank": mean_rank,
    "rank_sd": ranks.std(axis=0, ddof=1),
    "top1_frequency": (ranks == 1).mean(axis=0),
}).sort_values("mean_rank")
summary.to_csv(out_path("shap_importance_summary.csv"), index=False)
print(f"\n{summary.to_string(index=False)}")


# --- plots --------------------------------------------------------------
rep_values, rep_X_cat = all_values[rep], all_X_cat[rep]
ordered_names = [DISPLAY_NAMES[i] for i in idx]

# beeswarm: points from the representative iteration, rows in the averaged
# order. summary_plot sorts by importance unless sort=False, and with sort=False
# it draws column 0 at the top -- so values, features and names must all be
# permuted by the same idx.
fig = plt.figure()
shap.summary_plot(rep_values[:, idx], rep_X_cat.iloc[:, idx],
                  feature_names=ordered_names, sort=False, show=False)
ax = plt.gca()
ax.set_xlabel("SHAP value")
restyle_colorbar(fig, ax)

# third_nt and minus1_nt are categories carried as codes 0-3, so on the shared
# colour ramp each base lands on one fixed colour. Reproducing summary_plot's
# own normalisation (5th/95th percentile of the row, see shap/plots/_beeswarm.py)
# gives those exact colours, which can then be spelled out in a legend.
def base_colours(values):
    vmin, vmax = np.nanpercentile(values, 5), np.nanpercentile(values, 95)
    if vmin == vmax:
        vmin, vmax = np.nanpercentile(values, 1), np.nanpercentile(values, 99)
        if vmin == vmax:
            vmin, vmax = np.min(values), np.max(values)
    if vmin > vmax:
        vmin = vmax
    span = vmax - vmin if vmax > vmin else 1.0
    return [shap.plots.colors.red_blue(np.clip((k - vmin) / span, 0, 1))
            for k in range(len(BASES))]


palettes = {f: base_colours(rep_X_cat[f].values) for f in CATEGORICAL_FEATURES}
first = palettes[CATEGORICAL_FEATURES[0]]
if all(np.allclose(p, first) for p in palettes.values()):
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                          color=c, label=b) for b, c in zip(BASES, first)]
    ax.legend(handles=handles, title="Nucleotide rows", loc="lower right",
              fontsize=8, title_fontsize=8, frameon=False, ncol=4,
              handletextpad=0.2, columnspacing=0.9)
else:
    print("WARNING: the two categorical rows normalise differently, "
          "so one shared nucleotide legend would be wrong")
plt.tight_layout()
plt.savefig(out_path("shap_summary.png"), dpi=600, bbox_inches="tight")
plt.close()

# third_nt and minus1_nt are categories collapsed to codes 0-3, so on the
# beeswarm they get the same continuous blue-to-red ramp as everything else and
# the individual bases are unreadable. summary_plot takes a single cmap for all
# rows, so the discrete palette cannot go on that figure -- these rows are drawn
# separately here, one strip per base.
BASE_COLORS = {"A": "#2a9d5c", "C": "#2a78d6", "G": "#e08a1e", "T": "#e34948"}
rng_jitter = np.random.default_rng(0)

fig, axes = plt.subplots(len(CATEGORICAL_FEATURES), 1,
                         figsize=(6.5, 2.2 * len(CATEGORICAL_FEATURES)), sharex=True)
for ax_n, feat in zip(np.atleast_1d(axes), CATEGORICAL_FEATURES):
    col = SHAP_FEATURES.index(feat)
    codes = np.rint(rep_X_cat[feat].values).astype(int)
    sv = rep_values[:, col]
    for k, base in enumerate(BASES):
        m = codes == k
        if not m.any():
            continue
        y = len(BASES) - 1 - k + rng_jitter.uniform(-0.18, 0.18, m.sum())
        ax_n.scatter(sv[m], y, s=12, color=BASE_COLORS[base], alpha=0.75,
                     linewidths=0, zorder=3)
        ax_n.text(ax_n.get_xlim()[0], len(BASES) - 1 - k, "", va="center")
    ax_n.axvline(0, color="grey", linewidth=0.8, zorder=1)
    ax_n.set_yticks(range(len(BASES))[::-1])
    ax_n.set_yticklabels(BASES, fontsize=9)
    ax_n.set_ylabel(label(feat), fontsize=9)
    ax_n.set_ylim(-0.6, len(BASES) - 0.4)
    ax_n.grid(axis="x", color="lightgrey", linewidth=0.4, zorder=0)
    ax_n.set_axisbelow(True)
np.atleast_1d(axes)[-1].set_xlabel("SHAP value")
fig.tight_layout()
fig.savefig(out_path("shap_by_nucleotide.png"), dpi=600, bbox_inches="tight")
plt.close(fig)

# mean SHAP per base, so the direction of each nucleotide is in the tables too
per_base = []
for feat in CATEGORICAL_FEATURES:
    col = SHAP_FEATURES.index(feat)
    codes = np.rint(rep_X_cat[feat].values).astype(int)
    for k, base in enumerate(BASES):
        m = codes == k
        if m.any():
            per_base.append({"feature": feat, "label": label(feat), "base": base,
                             "n": int(m.sum()), "mean_shap": rep_values[m, col].mean()})
per_base = pd.DataFrame(per_base)
per_base.to_csv(out_path("shap_by_nucleotide.csv"), index=False)
print("\nmean SHAP per base (representative iteration):")
print(per_base.to_string(index=False))

# bar: heights averaged over all iterations, whiskers = 2.5-97.5 percentile
fig, ax = plt.subplots(figsize=(6.5, 4))
ypos = np.arange(len(idx))[::-1]
ax.barh(ypos, R.mean(axis=0)[idx], color="#2a78d6",
        xerr=[R.mean(axis=0)[idx] - np.percentile(R, 2.5, axis=0)[idx],
              np.percentile(R, 97.5, axis=0)[idx] - R.mean(axis=0)[idx]],
        error_kw={"ecolor": "#4a4a4a", "elinewidth": 0.9, "capsize": 2.5})
ax.set_yticks(ypos)
ax.set_yticklabels(ordered_names, fontsize=8.5)
ax.set_xlabel(f"{BAR_XLABEL}, mean over {n_done} random-control resamples")
ax.grid(axis="x", color="lightgrey", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(out_path("shap_importance_bar.png"), dpi=600, bbox_inches="tight")
plt.close(fig)

# rank stability: how much the ordering moves between draws
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.boxplot([ranks[:, i] for i in idx], vert=False, widths=0.6,
           medianprops={"color": "#e34948"})
ax.set_yticklabels(ordered_names, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel(f"rank within iteration (1 = most important), {n_done} resamples")
ax.grid(axis="x", color="lightgrey", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(out_path("shap_rank_stability.png"), dpi=600, bbox_inches="tight")
plt.close(fig)

# dependence plots for the top features, from the representative iteration
for feat in summary["feature"].head(4):
    plt.figure()
    shap.dependence_plot(feat, rep_values, rep_X_cat, show=False, interaction_index=None)
    if feat in CATEGORICAL_FEATURES:
        plt.xticks(range(len(BASES)), BASES)
    plt.xlabel(label(feat))
    plt.ylabel(f"SHAP value for\n{label(feat)}")
    plt.title(f"SHAP dependence — {label(feat)}")
    plt.tight_layout()
    plt.savefig(out_path(f"shap_dependence_{feat}.png"), dpi=600, bbox_inches="tight")
    plt.close()

print(f"\nsaved SHAP plots/summary to {RESULTS_DIR}/")
