import torch
import torch.nn as nn

from deep_models_common import load_data, run_cv, DROPOUT, N_TOKENS

# Multi-input transformer: nucleotide embedding + learned positional embedding
# -> 1-layer TransformerEncoderLayer (2 heads) -> mean-pool over positions,
# concatenated with the 19 engineered features into a small MLP head.


EMBED_DIM = 16
N_HEADS = 2
FF_DIM = 32
N_LAYERS = 1


class Transformer(nn.Module):
    def __init__(self, n_feat, seq_len, embed_dim=EMBED_DIM, n_heads=N_HEADS,
                 ff_dim=FF_DIM, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.token_emb = nn.Embedding(N_TOKENS, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(embed_dim + n_feat, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq_idx, x_feat):
        B, L = x_seq_idx.shape
        positions = torch.arange(L, device=x_seq_idx.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x_seq_idx) + self.pos_emb(positions)
        h = self.encoder(h)
        pooled = self.dropout(h.mean(dim=1))
        return self.head(torch.cat([pooled, x_feat], dim=1)).squeeze(-1)


if __name__ == "__main__":
    seq_onehot, seq_idx, X_feat, y, positions, feature_cols = load_data()
    seq_len = seq_idx.shape[1]
    run_cv("Transformer", Transformer, seq_idx, X_feat, y, positions, seq_len)
