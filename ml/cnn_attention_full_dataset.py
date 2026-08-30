import sys

import torch
import torch.nn as nn

from deep_models_common import load_data, run_cv, DROPOUT

# Same as cnn_full_dataset.py, but the sequence branch's pooling is replaced by
# an attention mechanism. Which kind of attention is switchable via ATTENTION.
# See the class docstring for the two options.

N_FILTERS = 8
KERNEL_SIZE = 5
ATTENTION = "gate"
GATE_POOL_K = 2


class CNNAttention(nn.Module):
    """Two ways to use a Linear(n_filters -> 1) scorer over the conv map:
    "pool": attention pooling - softmax over positions, weighted sum.
    "gate": the same scorer with sigmoid instead of softmax.
    """

    def __init__(self, n_feat, seq_len, n_filters=N_FILTERS, kernel_size=KERNEL_SIZE,
                 dropout=DROPOUT, attention=ATTENTION):
        super().__init__()
        self.attention = attention
        self.conv = nn.Conv1d(4, n_filters, kernel_size, padding=kernel_size // 2)
        self.attn_score = nn.Linear(n_filters, 1)
        self.dropout = nn.Dropout(dropout)

        if attention == "pool":
            self.pool_layer = None
            n_seq_out = n_filters
        elif attention == "gate":
            self.pool_layer = nn.AvgPool1d(GATE_POOL_K)
            n_seq_out = n_filters * (seq_len // GATE_POOL_K)
        else:
            raise ValueError(f"unknown attention {attention!r}")

        self.head = nn.Sequential(
            nn.Linear(n_seq_out + n_feat, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq, x_feat, return_attn=False):
        h = torch.relu(self.conv(x_seq))                    # (B, n_filters, L)
        scores = self.attn_score(h.transpose(1, 2)).squeeze(-1)   # (B, L)

        if self.attention == "pool":
            weights = torch.softmax(scores, dim=1)                # (B, L), sums to 1
            h = torch.bmm(weights.unsqueeze(1),
                          h.transpose(1, 2)).squeeze(1)           # (B, n_filters)
        else:
            weights = torch.sigmoid(scores)                       # (B, L), independent
            h = h * weights.unsqueeze(1)                          # (B, n_filters, L)
            h = self.pool_layer(h).flatten(start_dim=1)           # (B, n_filters * L//k)

        h = self.dropout(h)
        logit = self.head(torch.cat([h, x_feat], dim=1)).squeeze(-1)
        if return_attn:
            return logit, weights
        return logit


def make_attention_cnn(attention):
    def make_model(n_feat, seq_len):
        return CNNAttention(n_feat, seq_len, attention=attention)
    return make_model


if __name__ == "__main__":
    # optional CLI arg picks the variant: `python cnn_attention_full_dataset.py pool`
    attention = sys.argv[1] if len(sys.argv) > 1 else ATTENTION
    seq_onehot, seq_idx, X_feat, y, positions, feature_cols = load_data()
    seq_len = seq_onehot.shape[2]
    run_cv(f"CNNAttention_{attention}", make_attention_cnn(attention),
           seq_onehot, X_feat, y, positions, seq_len)
