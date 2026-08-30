import sys

import torch
import torch.nn as nn

from deep_models_common import load_data, run_cv, DROPOUT

# CNN + self-attention: the same 1D-CNN as cnn_full_dataset.py, followed by one
# self-attention block over the convolution map, then the same pooling choices.


N_FILTERS = 8
KERNEL_SIZE = 5
N_HEADS = 2
POOLING = "avgpool2"


class SelfAttentionBlock(nn.Module):
    """Multi-head scaled dot-product self-attention with a residual connection.
    Input and output are both (B, L, C)."""

    def __init__(self, n_channels, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=n_channels, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(n_channels)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)      # query = key = value = x
        return self.norm(x + attn_out)        # residual, so the block refines


class CNNSelfAttention(nn.Module):
    def __init__(self, n_feat, seq_len, n_filters=N_FILTERS, kernel_size=KERNEL_SIZE,
                 n_heads=N_HEADS, dropout=DROPOUT, pooling=POOLING):
        super().__init__()
        self.pooling_mode = pooling
        self.conv = nn.Conv1d(4, n_filters, kernel_size, padding=kernel_size // 2)
        self.self_attn = SelfAttentionBlock(n_filters, n_heads, dropout)
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
        h = self.self_attn(h.transpose(1, 2))       # (B, L, n_filters) -- shape kept
        h = h.transpose(1, 2)                       # back to (B, n_filters, L)

        if self.pooling_mode == "max":
            h = torch.amax(h, dim=2)                # (B, n_filters) -- position lost
        else:
            if self.pool_layer is not None:         # avgpool yes, flatten no
                h = self.pool_layer(h)              # (B, n_filters, L // k)
            h = h.flatten(start_dim=1)

        h = self.dropout(h)
        return self.head(torch.cat([h, x_feat], dim=1)).squeeze(-1)


def make_self_attention_cnn(pooling):
    def make_model(n_feat, seq_len):
        return CNNSelfAttention(n_feat, seq_len, pooling=pooling)
    return make_model


if __name__ == "__main__":
    # optional CLI arg picks the pooling variant, as in cnn_full_dataset.py
    pooling = sys.argv[1] if len(sys.argv) > 1 else POOLING
    seq_onehot, seq_idx, X_feat, y, positions, feature_cols = load_data()
    seq_len = seq_onehot.shape[2]
    run_cv(f"CNNSelfAttention_{pooling}", make_self_attention_cnn(pooling),
           seq_onehot, X_feat, y, positions, seq_len)
