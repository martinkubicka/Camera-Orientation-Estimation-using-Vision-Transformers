import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import py360convert
import torchvision.transforms as T
import numpy as np

def load_dinov2(model_name: str = "dinov2_vitb14", pretrained: bool = True) -> nn.Module:
    vit = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)
    for attr in ("head", "heads"):  
        if hasattr(vit, attr): setattr(vit, attr, nn.Identity())
    return vit

class PoseRegressor(nn.Module):
    def __init__(self,
                 model_name: str = "dinov2_vitb14",
                 hidden_dim: int = 512,
                 pretrained: bool = True,
                 num_rows: int = 4,
                 num_cols: int = 8,
                 fov: tuple = (45, 45),
                 tile_size: tuple = (518, 518)):
        super().__init__()
        self.backbone = load_dinov2(model_name, pretrained)
        self.embed_dim = self.backbone.embed_dim 
        for p in self.backbone.parameters(): p.requires_grad = False

        self.cross_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, batch_first=True)
        self.attn_pool = nn.Linear(self.embed_dim, 1)

        # added
        self.norm_pano = nn.LayerNorm(self.embed_dim)  # <-- added
        self.norm_query = nn.LayerNorm(self.embed_dim)  # <-- added
        self.norm_post_pano = nn.LayerNorm(self.embed_dim)  # <-- added
        self.norm_post_query = nn.LayerNorm(self.embed_dim)  # <-- added
        self.norm_fused = nn.LayerNorm(2 * self.embed_dim)  # <-- added

        n_tiles = num_rows * num_cols
        self.pos_embed = nn.Parameter(torch.zeros(n_tiles, self.embed_dim))

        self.pitch_bins = 180  # [-90, 90) -> 180 classes, 1° each
        self.yaw_bins   = 360  # [-180, 180) -> 360 classes
        self.roll_bins  = 360  # [0, 360) -> 360 classes

        self.head_pitch = nn.Sequential(
            nn.Linear(2 * self.embed_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, self.pitch_bins)
        )
        self.head_yaw = nn.Sequential(
            nn.Linear(2 * self.embed_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, self.yaw_bins)
        )
        self.head_roll = nn.Sequential(
            nn.Linear(2 * self.embed_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, self.roll_bins)
        )

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.fov = fov
        self.tile_size = tile_size
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

        j = torch.arange(num_rows, dtype=torch.float32)
        self.pitch_vals = 90.0 - (j + 0.5) * (180.0 / num_rows)
        i = torch.arange(num_cols, dtype=torch.float32)
        self.yaw_vals = -180.0 + (i + 0.5) * (360.0 / num_cols)

    def forward(self, query_img, pano):
        B = len(pano)
        n_tiles = self.num_rows * self.num_cols
        all_tiles = []
        for img in pano:
            for pv in self.pitch_vals.tolist():
                for yv in self.yaw_vals.tolist():
                    tile = py360convert.e2p(np.array(Image.open(img).convert("RGB")), self.fov, float(yv), float(pv),
                                            (self.tile_size[1], self.tile_size[0]))
                    all_tiles.append(self.transform(tile).to(query_img.device))
        pano_tiles = torch.stack(all_tiles, 0)  # [B*32,3,518,518]
        feats_p = self.backbone.forward_features(pano_tiles)
        tile_patches = feats_p['x_norm_patchtokens']
        Np = tile_patches.size(1)
        
        pano_seq = tile_patches.view(B, n_tiles, Np, self.embed_dim) # [B, 32, Np, D]
        pos = self.pos_embed.unsqueeze(0).unsqueeze(2)
        pano_seq = pano_seq + pos
        pano_emb = pano_seq.view(B, n_tiles * Np, self.embed_dim)

        feats_q = self.backbone.forward_features(query_img)
        query_patches = feats_q['x_norm_patchtokens']

        pano_emb = self.norm_pano(pano_emb)        # <-- added
        query_patches = self.norm_query(query_patches)  # <-- added

        P_ca, _ = self.cross_attn(pano_emb, query_patches, query_patches)
        R_ca, _ = self.cross_attn(query_patches, pano_emb, pano_emb)
        pano_emb = self.norm_post_pano(pano_emb + P_ca)  # <-- added (residual + norm)
        query_patches = self.norm_post_query(query_patches + R_ca)  # <-- added (residual + norm)

        w_p = F.softmax(self.attn_pool(pano_emb).squeeze(-1), dim=1).unsqueeze(-1)
        w_q = F.softmax(self.attn_pool(query_patches).squeeze(-1), dim=1).unsqueeze(-1)
        pooled_p = (w_p * pano_emb).sum(1)       # [B,D]
        pooled_q = (w_q * query_patches).sum(1)  # [B,D]

        fused = torch.cat([pooled_p, pooled_q], dim=-1)  # [B,2D]
        fused = self.norm_fused(fused)  # <-- added

        logits_pitch = self.head_pitch(fused)  # [B, 180]
        logits_yaw   = self.head_yaw(fused)    # [B, 360]
        logits_roll  = self.head_roll(fused)   # [B, 360]
        return logits_pitch, logits_yaw, logits_roll
