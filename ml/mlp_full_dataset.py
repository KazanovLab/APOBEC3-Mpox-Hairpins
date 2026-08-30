import torch
import torch.nn as nn

from deep_models_common import load_data, run_cv, DROPOUT

# Multi-input MLP: the (4 x L) one-hot matrix of each site is flattened into a
# single 4*L vector and concatenated with the engineered features, giving
# 4*L + n_feat inputs. Those go through one hidden layer of N_HIDDEN_UNITS units
# (ReLU + dropout) into a single output logit.

N_HIDDEN_UNITS = 32


class MLP(nn.Module):
    def __init__(self, n_feat, seq_len, n_hidden_units=N_HIDDEN_UNITS, dropout=DROPOUT):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(4 * seq_len + n_feat, n_hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden_units, 1),
        )

    def forward(self, x_seq, x_feat):
        flat = x_seq.flatten(start_dim=1)          # (B, 4*L), position preserved
        return self.head(torch.cat([flat, x_feat], dim=1)).squeeze(-1)


if __name__ == "__main__":
    seq_onehot, seq_idx, X_feat, y, positions, feature_cols = load_data()
    seq_len = seq_onehot.shape[2]
    run_cv("MLP", MLP, seq_onehot, X_feat, y, positions, seq_len)
