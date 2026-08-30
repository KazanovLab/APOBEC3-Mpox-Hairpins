import sys

import torch
import torch.nn as nn

from deep_models_common import load_data, run_cv, DROPOUT

# Multi-input CNN: a 1D-CNN over the raw +/- L nt window, concatenated with the
# 19 engineered features, into a small MLP head.
# How the convolution map is reduced before the head is switchable via
# POOLING. See the class docstring for the three options.

N_FILTERS = 8
KERNEL_SIZE = 5
POOLING = "avgpool2"


class CNN(nn.Module):
    """Three ways to turn the (n_filters x L) convolution map into a vector:
    "flatten": whole map - n_filters * L head inputs.
    "max": collapses the map to one value per filter
    "avgpoolK": averages neighbouring positions in groups of K and then
    flattens, keeping position at reduced resolution for n_filters * (L // K).
    """

    def __init__(self, n_feat, seq_len, n_filters=N_FILTERS, kernel_size=KERNEL_SIZE,
                 dropout=DROPOUT, pooling=POOLING):
        super().__init__()
        self.pooling_mode = pooling
        self.conv = nn.Conv1d(4, n_filters, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)

        if pooling == "max":
            self.pool_layer = None
            n_seq_out = n_filters
        elif pooling == "flatten":
            self.pool_layer = None
            n_seq_out = n_filters * seq_len
        elif pooling.startswith("avgpool"):
            k = int(pooling.removeprefix("avgpool"))
            self.pool_layer = nn.AvgPool1d(k)
            n_seq_out = n_filters * (seq_len // k)
        else:
            raise ValueError(f"unknown pooling {pooling!r}")

        self.head = nn.Sequential(
            nn.Linear(n_seq_out + n_feat, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq, x_feat):
        h = torch.relu(self.conv(x_seq))            # (B, n_filters, L)
        if self.pooling_mode == "max":
            h = torch.amax(h, dim=2)                # (B, n_filters) -- position lost
        else:
            if self.pool_layer is not None:         # avgpool yes, flatten no
                h = self.pool_layer(h)              # (B, n_filters, L // k)
            h = h.flatten(start_dim=1)
        h = self.dropout(h)
        return self.head(torch.cat([h, x_feat], dim=1)).squeeze(-1)


def make_cnn(pooling):
    def make_model(n_feat, seq_len):
        return CNN(n_feat, seq_len, pooling=pooling)
    return make_model


if __name__ == "__main__":
    # optional CLI arg picks the pooling variant: `python cnn_full_dataset.py max`
    pooling = sys.argv[1] if len(sys.argv) > 1 else POOLING
    seq_onehot, seq_idx, X_feat, y, positions, feature_cols = load_data()
    seq_len = seq_onehot.shape[2]
    run_cv(f"CNN_{pooling}", make_cnn(pooling),
           seq_onehot, X_feat, y, positions, seq_len)
