import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# Shared setup for the multi-input deep models on the full dataset
# (cnn_full_dataset.py, cnn_attention_full_dataset.py,
# transformer_full_dataset.py). Each model gets two inputs:
#   1. the raw +/- L nt nucleotide window around the position
#   2. the same 19 engineered features the best classic model uses
#      (grantham + third_nt/minus1_nt one-hot + the 10 loop_boundary_* columns)

IN_TSV = "ml_table_all.tsv"
RESULTS_DIR = "results_deep_ml_full_dataset"
TARGET = "any_mutation"
WINDOW_HALF = 15
N_SPLITS = 10
SEED = 1
N_EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

BASES = "ACGT"
# The assembly has a few N's at both ends of the genome, so they only enter the
# window at half-widths above 20 (8 of 24695 rows at +/-50, none of them
# mutated). N gets its own token: an all-zero one-hot column, meaning "no
# evidence for any base", and a fifth embedding row for the transformer.
N_TOKENS = len(BASES) + 1
BASE_IDX = {b: i for i, b in enumerate(BASES)}
BASE_IDX["N"] = len(BASES)


def load_data(window_half=WINDOW_HALF, left=None, right=None):
    """Returns (X_seq_onehot, X_seq_idx, X_feat, y, positions, feature_names).

    The window is symmetric by default. Passing left/right (both counts of
    nucleotides, so left=0, right=3 means offsets 0..+3) makes it one-sided,
    which is what separates "the signal sits 2-3nt away" from "the signal sits
    2-3nt away on the 3' side" -- a symmetric sweep adds both sides at once and
    can only report a distance.
    """
    if left is None:
        left = window_half
    if right is None:
        right = window_half
    df = pd.read_csv(IN_TSV, sep='\t')
    df = pd.get_dummies(df, columns=["third_nt", "minus1_nt"], dtype="uint8")

    loop_boundary_cols = [c for c in df.columns
                          if (c.startswith("loop_boundary_") or c == "in_loop_boundary")
                          and "sequence" not in c]
    df[loop_boundary_cols] = df[loop_boundary_cols].fillna(0.0)

    third_nt_cols = [f"third_nt_{b}" for b in BASES]
    minus1_nt_cols = [f"minus1_nt_{b}" for b in BASES]
    feature_cols = ["grantham"] + third_nt_cols + minus1_nt_cols + loop_boundary_cols

    seq_len = left + right + 1
    seqs = df["context_pm50"].str[50 - left: 50 + right + 1].str.upper()
    assert (seqs.str.len() == seq_len).all()

    seq_idx = np.stack([np.array([BASE_IDX[ch] for ch in s], dtype=np.int64) for s in seqs])
    onehot = np.zeros((len(seqs), N_TOKENS, seq_len), dtype=np.float32)
    rows = np.arange(len(seqs))[:, None]
    cols = np.arange(seq_idx.shape[1])[None, :]
    onehot[rows, seq_idx, cols] = 1.0
    seq_onehot = onehot[:, :len(BASES), :]   # drops the N row, leaving it all-zero

    X_feat = df[feature_cols].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    window = f"+/-{left}" if left == right else f"-{left}..+{right}"
    print(f"n_rows={len(df)}, n_positives={int(y.sum())} ({y.mean():.1%}), "
          f"n_engineered_features={len(feature_cols)}, window={window}nt (L={seq_len})")
    return seq_onehot, seq_idx, X_feat, y, df["position"].values, feature_cols


def run_cv(model_name, make_model, seq_input, X_feat, y, positions, seq_len):
    """10-fold stratified CV, returns out-of-fold probabilities."""
    n = len(y)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    prob = np.zeros(n, dtype=np.float64)
    n_params = sum(p.numel() for p in make_model(X_feat.shape[1], seq_len).parameters())
    print(f"{model_name}: {n_params} params")

    t0 = time.time()
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_feat, y)):
        torch.manual_seed(SEED + fold_i)

        scaler = StandardScaler().fit(X_feat[train_idx])
        Xf_train = torch.from_numpy(scaler.transform(X_feat[train_idx]).astype(np.float32))
        Xf_test = torch.from_numpy(scaler.transform(X_feat[test_idx]).astype(np.float32))
        Xs_train = torch.from_numpy(seq_input[train_idx])
        Xs_test = torch.from_numpy(seq_input[test_idx])
        y_train = torch.from_numpy(y[train_idx])

        pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()])
        model = make_model(X_feat.shape[1], seq_len)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        loader = DataLoader(TensorDataset(Xs_train, Xf_train, y_train),
                            batch_size=BATCH_SIZE, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED + fold_i))

        model.train()
        for _ in range(N_EPOCHS):
            for xb_seq, xb_feat, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb_seq, xb_feat), yb)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            prob[test_idx] = torch.sigmoid(model(Xs_test, Xf_test)).numpy()

        print(f"  fold {fold_i + 1}/{N_SPLITS} done ({time.time() - t0:.0f}s elapsed)")

    y_int = y.astype(int)
    roc_auc = roc_auc_score(y_int, prob)
    pr_auc = average_precision_score(y_int, prob)
    print(f"\n=== {model_name} ===\nroc_auc={roc_auc:.4f}  pr_auc={pr_auc:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = pd.DataFrame({"position": positions, TARGET: y_int, "pred_prob": prob})
    out_path = os.path.join(RESULTS_DIR, f"predictions_{model_name}.csv")
    out.to_csv(out_path, index=False)
    print(f"saved {out_path}")
    return prob, roc_auc
