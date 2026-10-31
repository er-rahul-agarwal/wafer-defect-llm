"""
detector/model.py -- a small Vision Transformer (ViT) for wafer-map defect
detection, built from scratch for learning.

Input : a batch of 52x52 single-channel wafer maps  -> (B, 1, 52, 52)
Output: 8 raw logits per map (one per base defect type) -> (B, 8)
        No sigmoid here; it is applied at loss time (multi-label setup).

A ViT works in four moves:
  1. Patch embedding -- cut the image into small squares, flatten each into a
     vector (a "token"). The image becomes a short sequence of tokens.
  2. Add a learnable [CLS] token and positional embeddings, so the model knows
     token order (attention is otherwise order-blind) and has one summary slot.
  3. Transformer encoder -- stacked self-attention + MLP blocks let every patch
     attend to every other patch, building a global view of the wafer.
  4. Classification head -- read the [CLS] token, project it to 8 logits.

Design values are tagged [DESIGN]; they are starting points, tunable later.

Run this file directly to self-test it:
    python detector/model.py
"""

import os
import sys

import torch
import torch.nn as nn

# config.py lives in the project root (one level up from detector/).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import config


class PatchEmbedding(nn.Module):
    """Cut the image into patches and embed each into a vector.

    A clean trick: a Conv2d with kernel = stride = patch_size does exactly
    this -- each non-overlapping patch becomes one embedding vector. Output
    is (B, num_patches, embed_dim).
    """

    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        # Number of patches along one side, then total. 52 / 4 = 13 -> 169.
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        # One conv stamp per patch produces that patch's embedding.
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                 # (B, embed_dim, grid, grid)
        x = x.flatten(2)                 # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)            # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """One encoder block: self-attention, then an MLP, each with a residual
    connection and layer-norm (the standard pre-norm transformer recipe).
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # batch_first=True so tensors stay (B, tokens, dim) throughout.
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm: normalize, attend, then add back (residual).
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class WaferViT(nn.Module):
    """The full small ViT for wafer-defect multi-label classification."""

    def __init__(self,
                 image_size=config.IMAGE_SIZE,   # 52, from data inspection
                 patch_size=4,                   # [DESIGN] 52/4=13 -> 169 patches
                 in_channels=1,                  # [DESIGN] wafer maps are 1-channel
                 num_labels=config.NUM_LABELS,   # 8 base defect types
                 embed_dim=128,                  # [DESIGN] token width (small model)
                 depth=6,                        # [DESIGN] number of transformer blocks
                 num_heads=4,                    # [DESIGN] attention heads (128/4=32 each)
                 mlp_ratio=4.0,                  # [DESIGN] MLP hidden = 4x embed_dim
                 dropout=0.1):                   # [DESIGN] light regularization
        super().__init__()

        self.patch_embed = PatchEmbedding(image_size, patch_size,
                                          in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token: one extra token whose final state summarizes
        # the whole image for classification.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embeddings: one per token (patches + the CLS token), so
        # the model can tell where each patch sits on the wafer.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Head: CLS summary -> 8 logits (no activation; sigmoid is at loss time).
        self.head = nn.Linear(embed_dim, num_labels)

        # Small-std init for the learnable tokens (standard ViT practice).
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                      # (B, num_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, embed_dim)
        x = torch.cat((cls, x), dim=1)               # prepend CLS token
        x = x + self.pos_embed                       # add positions
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]                             # take the CLS token's state
        return self.head(cls_out)                    # (B, num_labels) logits


# ----------------------------------------------------------------------
# Self-test (Section 8): run this file directly to verify the model before
# any training. Checks output shape, gradient flow, and parameter count.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    model = WaferViT()

    # A fake batch of 2 wafer maps, shaped like our real data.
    dummy = torch.randn(2, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    out = model(dummy)

    # 1) Output shape must be (batch, num_labels).
    print("Output shape:", tuple(out.shape), "(expected (2, %d))" % config.NUM_LABELS)
    assert out.shape == (2, config.NUM_LABELS), "Wrong output shape!"

    # 2) Gradient flow: a backward pass must reach the parameters.
    loss = out.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print("Gradients reach all parameters:", has_grad)
    assert has_grad, "Some parameters got no gradient!"

    # 3) Parameter count (a small ViT should be a few million params).
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    print("\nSelf-test passed.")