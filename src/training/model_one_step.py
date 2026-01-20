import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import numpy as np
import py360convert
from pathlib import Path

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def load_dinov2(model_name: str = "dinov2_vitb14", pretrained: bool = True) -> nn.Module:
    """Load DINOv2 ViT backbone (base/14) and strip classification heads."""
    vit = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)
    for attr in ("head", "heads"):
        if hasattr(vit, attr):
            setattr(vit, attr, nn.Identity())
    vit.eval()
    return vit

# -----------------------------------------------------------------------------
# Learned 2‑D positional encoding (row⊕col embedding)
# -----------------------------------------------------------------------------

class Learned2DPositionalEncoding(nn.Module):
    def __init__(self, h: int, w: int, dim: int):
        super().__init__()
        self.row_embed = nn.Parameter(torch.randn(h, dim // 2))
        self.col_embed = nn.Parameter(torch.randn(w, dim // 2))

    def forward(self):
        pos = torch.cat([
            self.row_embed[:, None, :].expand(-1, self.col_embed.size(0), -1),
            self.col_embed[None, :, :].expand(self.row_embed.size(0), -1, -1),
        ], dim=-1)  # (H, W, D)
        return pos.view(-1, pos.size(-1))               # (H*W, D)

# -----------------------------------------------------------------------------
# Cross‑decoder – shared‑weight TransformerDecoder
# -----------------------------------------------------------------------------

class CrossDecoder(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerDecoderLayer(dim, num_heads, dim_feedforward=4*dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, tgt_a: torch.Tensor, tgt_b: torch.Tensor):
        out_a = self.decoder(tgt=tgt_a, memory=tgt_b)
        out_b = self.decoder(tgt=tgt_b, memory=tgt_a)
        return out_a, out_b

# -----------------------------------------------------------------------------
# PoseRegressor – predicts (pitch, yaw, roll) – with *attention pooling*
# -----------------------------------------------------------------------------

class OrientationRegressor(nn.Module):
    """Estimate (pitch, yaw, roll) of a query image relative to a panorama.

    Uses attention‑pooling over the token sequence (query & pano) after a masked
    cross‑encoder. No regression token is required.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        hidden_dim: int = 512,
        pretrained: bool = True,
        num_rows: int = 4,
        num_cols: int = 8,
        fov: tuple = (45, 45),
        tile_size: tuple = (518, 518),
        dec_heads: int = 8,
        enc_heads: int = 8,
        enc_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Frozen backbone (DINOv2 Base/14)
        # ------------------------------------------------------------------
        self.backbone = load_dinov2(model_name, pretrained)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.embed_dim = self.backbone.embed_dim  # 768 for vit‑b/14

        # ------------------------------------------------------------------
        # Transformer blocks
        # ------------------------------------------------------------------
        self.cross_decoder = CrossDecoder(self.embed_dim, num_heads=dec_heads, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=enc_heads, dim_feedforward=4*self.embed_dim, dropout=dropout, batch_first=True)
        self.cross_encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # ------------------------------------------------------------------
        # Attention pooling + MLP head
        # ------------------------------------------------------------------
        self.attn_pool = nn.Linear(self.embed_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # pitch, yaw, roll
        )

        # ------------------------------------------------------------------
        # Panorama tiling parameters
        # ------------------------------------------------------------------
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.fov = fov
        self.tile_size = tile_size

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Pre‑computed tile centres
        j = torch.arange(num_rows, dtype=torch.float32)
        self.pitch_vals = 90.0 - (j + 0.5) * (180.0 / num_rows)
        i = torch.arange(num_cols, dtype=torch.float32)
        self.yaw_vals = -180.0 + (i + 0.5) * (360.0 / num_cols)

        # Learned positional encodings for 14×14 patch grid
        self.h_feat = self.w_feat = 37
        self.pos_enc = Learned2DPositionalEncoding(self.h_feat, self.w_feat, self.embed_dim)

    # ----------------------------------------------------------------------
    # Helper – cross‑attention mask (blocks self‑attention within halves)
    # ----------------------------------------------------------------------
    def _cross_mask(self, Nq: int, Np: int, device):
        """Return a bool tensor of shape (Nq+Np, Nq+Np).

        * True  → position is MASKED (ignored by attention)
        * False → position is allowed
        """
        N_tot = Nq + Np
        mask = torch.zeros((N_tot, N_tot), dtype=torch.bool, device=device)  # all False (allowed)
        mask[:Nq, :Nq] = True   # block Q ↔ Q
        mask[Nq:, Nq:] = True   # block P ↔ P
        return mask

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, query_img: torch.Tensor, pano: list[Image.Image]):
        B = query_img.size(0)
        device = query_img.device

        # --------------------------------------------------------------
        # 1. Panorama → perspective tiles
        # --------------------------------------------------------------
        n_tiles = self.num_rows * self.num_cols
        tiles = []
        for img in pano:
            if isinstance(img, (str, Path)):
                img = Image.open(img)
            img = img.convert("RGB")
            for pv in self.pitch_vals.tolist():
                for yv in self.yaw_vals.tolist():
                    tile_np = py360convert.e2p(np.array(img), self.fov, float(yv), float(pv), (self.tile_size[1], self.tile_size[0]))
                    tiles.append(self.transform(Image.fromarray(tile_np)))
        pano_tiles = torch.stack(tiles, 0).to(device)              # (B*n_tiles,3,H,W)

        # --------------------------------------------------------------
        # 2. Extract ViT patch‑tokens
        # --------------------------------------------------------------
        feat_tiles = self.backbone.forward_features(pano_tiles)
        tile_tokens = feat_tiles["x_norm_patchtokens"]            # (B*n_tiles, Nt, D)
        Nt = tile_tokens.size(1)
        tile_tokens = tile_tokens.view(B, n_tiles, Nt, self.embed_dim)  # (B, n_tiles, Nt, D)
        pos_tile = self.pos_enc().unsqueeze(0).unsqueeze(0)        # (1,1,Nt,D)
        tile_tokens = tile_tokens + pos_tile
        pano_seq = tile_tokens.view(B, n_tiles*Nt, self.embed_dim) # (B, Np, D)

        # Query image tokens
        feat_q = self.backbone.forward_features(query_img)
        query_tokens = feat_q["x_norm_patchtokens"]               # (B, Nq, D)

        # --------------------------------------------------------------
        # 3. Cross‑decoder refinement
        # --------------------------------------------------------------
        query_ref, pano_ref = self.cross_decoder(query_tokens, pano_seq)

        # --------------------------------------------------------------
        # 4. Masked cross‑encoder (no reg token)
        # --------------------------------------------------------------
        concat_seq = torch.cat([query_ref, pano_ref], dim=1)       # (B, N_tot, D)
        mask = self._cross_mask(query_ref.size(1), pano_ref.size(1), device)
        enc_out = self.cross_encoder(concat_seq, mask=mask)

        # --------------------------------------------------------------
        # 5. Attention‑pool → MLP head
        # --------------------------------------------------------------
        weights = F.softmax(self.attn_pool(enc_out).squeeze(-1), dim=1).unsqueeze(-1)  # (B,N_tot,1)
        pooled = (weights * enc_out).sum(dim=1)                     # (B, D)
        angles = self.head(pooled)                                  # (B, 3)
        return angles
