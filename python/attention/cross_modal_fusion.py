import torch
import torch.nn as nn
import os
import sys

# Add python dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import EMBEDDING_DIM

class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim: int = EMBEDDING_DIM, num_heads: int = 4, use_fusion: bool = True,
                 num_modalities: int = 2):
        """
        num_modalities: number of input feature columns per node (2 = methylation + ATAC,
        1 = methylation only). When 1, cross-modal attention has nothing to attend across,
        so this falls back to a direct linear projection regardless of use_fusion --
        the ablation flag only has meaning when there are 2+ modalities to fuse.
        """
        super().__init__()
        self.use_fusion = use_fusion
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.last_attn_weights = None

        if num_modalities == 1:
            # Single-modality path (e.g. methylation-only real data bootstrap, USE_ATAC=False).
            # No cross-modal attention possible with one modality -- straight projection.
            self.single_proj = nn.Linear(1, embed_dim)
        elif self.use_fusion:
            # Independent projections for each modality
            self.meth_proj = nn.Linear(1, embed_dim)
            self.atac_proj = nn.Linear(1, embed_dim)

            # MHA: batch_first=True expects [Batch, SeqLen, EmbedDim]
            self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        else:
            # Fallback for ablation: concatenate (size 2) and project
            self.fallback_proj = nn.Linear(2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Input tensor of shape [N, num_modalities]. When num_modalities=2, column 0 is
        methylation and column 1 is ATAC. When num_modalities=1, it's methylation only.
        Returns tensor of shape [N, EMBEDDING_DIM]
        """
        if self.num_modalities == 1:
            # [N, 1] -> [N, EMBEDDING_DIM], no attention -- nothing to cross-attend with
            self.last_attn_weights = None
            return self.single_proj(x)

        if not self.use_fusion:
            # Ablation path: simple projection
            # [N, 2] -> [N, EMBEDDING_DIM]
            self.last_attn_weights = None
            return self.fallback_proj(x)

        # Split modalities
        # meth: [N, 1]
        meth = x[:, 0:1]
        # atac: [N, 1]
        atac = x[:, 1:2]

        # Project independently
        # [N, 1] -> [N, EMBEDDING_DIM]
        meth_emb = self.meth_proj(meth)
        # [N, 1] -> [N, EMBEDDING_DIM]
        atac_emb = self.atac_proj(atac)

        # Stack into sequence of length 2 per node
        # [N, 2, EMBEDDING_DIM] -> stacked modality tokens before attention
        seq = torch.stack([meth_emb, atac_emb], dim=1)

        # Self-attention over the two modalities
        # seq: [N, 2, EMBEDDING_DIM]
        attn_out, attn_weights = self.attention(query=seq, key=seq, value=seq)
        # attn_out: [N, 2, EMBEDDING_DIM]
        # Stash attention weights for interpretability extraction (Section 5.2.6 of proposal)
        self.last_attn_weights = attn_weights.detach()

        # Pool across the sequence dimension (mean pooling)
        # [N, 2, EMBEDDING_DIM] -> [N, EMBEDDING_DIM]
        out = attn_out.mean(dim=1)

        return out
