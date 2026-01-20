import torch
import torch.nn as nn
import torch.nn.functional as F

def load_dinov2(model_name: str = "dinov2_vitl14",
                pretrained: bool = True) -> nn.Module:
    vit = torch.hub.load("facebookresearch/dinov2",
                         model_name,
                         pretrained=pretrained)
    # strip off heads
    for attr in ("head", "heads"):
        if hasattr(vit, attr):
            setattr(vit, attr, nn.Identity())
    return vit

class PoseRegressor(nn.Module):
    def __init__(self,
                 model_name: str = "dinov2_vitl14",
                 hidden_dim: int = 512,
                 pretrained: bool = True):
        super().__init__()

        self.backbone = load_dinov2(model_name, pretrained)
        self.embed_dim = self.backbone.embed_dim  # e.g. 768

        for p in self.backbone.parameters(): # freeze
            p.requires_grad = False

        # num_blocks = len(self.backbone.blocks)
        # for blk in self.backbone.blocks[num_blocks-2 : num_blocks]:
        #     for p in blk.parameters():
        #         p.requires_grad = False # True if fine-tuning

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            batch_first=True
        )

        self.attn_pool = nn.Linear(self.embed_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(2 * self.embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)  # yaw, pitch, roll
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)
        return tokens['x_norm_patchtokens']             # [B, N, D]

    def forward(self,
                query: torch.Tensor,    # [B, 3, H, W]
                cutout: torch.Tensor):  # [B, 3, H, W]
        Pt = self._encode(query)    # [B, N, D]
        Rt = self._encode(cutout)   # [B, N, D]


        P_ca, _ = self.cross_attn(Pt, Rt, Rt)  # [B, N, D]
        R_ca, _ = self.cross_attn(Rt, Pt, Pt)  # [B, N, D]

        Pt = Pt + P_ca
        Rt = Rt + R_ca

        w_p = F.softmax(self.attn_pool(Pt).squeeze(-1), dim=1).unsqueeze(-1)
        w_r = F.softmax(self.attn_pool(Rt).squeeze(-1), dim=1).unsqueeze(-1)
        pooled_p = (w_p * Pt).sum(dim=1)     # [B, D]
        pooled_r = (w_r * Rt).sum(dim=1)     # [B, D]

        fused = torch.cat([pooled_p, pooled_r], dim=-1)  # [B, 2D]
        angles = self.head(fused)                        # [B, 3]
        return angles
