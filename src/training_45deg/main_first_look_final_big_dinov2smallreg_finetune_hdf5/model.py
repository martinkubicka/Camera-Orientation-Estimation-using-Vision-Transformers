import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import py360convert
import torchvision.transforms as T
import numpy as np

# TODO
def load_dinov2(model_name: str = "dinov2_vits14_reg", pretrained: bool = True) -> nn.Module:
    vit = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)
    for attr in ("head", "heads"):  
        if hasattr(vit, attr): setattr(vit, attr, nn.Identity())
    return vit

class PoseRegressor(nn.Module):
    def __init__(self,
                 model_name: str = "dinov2_vits14_reg", # TODO
                 hidden_dim: int = 512,
                 pretrained: bool = True,
                 num_rows: int = 4,
                 num_cols: int = 8,
                 fov: tuple = (45, 45),
                 tile_size: tuple = (518, 518)):
        super().__init__()
        self.backbone = load_dinov2(model_name, pretrained)
        self.embed_dim = self.backbone.embed_dim 
        
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
        B = query_img.size(0)
        n_tiles = self.num_rows * self.num_cols   # 32

        feats_list = []
        for i in range(n_tiles):
            # pano[:, i] -> [B, 3, H, W]
            tile = pano[:, i, :, :, :]            # already a batch of tiles

            # Forward through DINOv2
            feats_i = self.backbone.forward_features(tile)

            # Extract patch tokens (already flattened)
            # shape: [B, Np, D]
            token_i = feats_i["x_norm_patchtokens"]

            feats_list.append(token_i)

        # Stack tokens in tile dimension
        tile_patches = torch.stack(feats_list, dim=1)   # [B, 32, Np, D]

        B, n_tiles, Np, D = tile_patches.shape

        # Add positional encoding
        pos = self.pos_embed.unsqueeze(0).unsqueeze(2)   # [1, 32, 1, D]
        pano_seq = tile_patches + pos 
        
        ## SIN POS - pano
        pos_div = torch.exp(torch.arange(0, self.embed_dim, 2, device=pano_seq.device, dtype=pano_seq.dtype) * (-torch.log(torch.tensor(10000.0, device=pano_seq.device, dtype=pano_seq.dtype)) / self.embed_dim))
        patch_idx = torch.arange(Np, device=pano_seq.device, dtype=pano_seq.dtype).unsqueeze(1)  # [Np,1]
        pe = torch.zeros(Np, self.embed_dim, device=pano_seq.device, dtype=pano_seq.dtype)       # [Np,D]
        pe[:, 0::2] = torch.sin(patch_idx * pos_div)
        pe[:, 1::2] = torch.cos(patch_idx * pos_div)
        pano_seq = pano_seq + pe.unsqueeze(0).unsqueeze(1)  # [1,1,Np,D] broadcast
        # END OF SIN POS
        
        pano_emb = pano_seq.view(B, n_tiles * Np, self.embed_dim)

        feats_q = self.backbone.forward_features(query_img)
        query_patches = feats_q['x_norm_patchtokens']

        # SIN POS - query
        Nq = query_patches.size(1)
        Dq = query_patches.size(2)
        q_div = torch.exp(torch.arange(0, Dq, 2, device=query_patches.device, dtype=query_patches.dtype) *
                          (-torch.log(torch.tensor(10000.0, device=query_patches.device, dtype=query_patches.dtype)) / Dq))
        q_idx = torch.arange(Nq, device=query_patches.device, dtype=query_patches.dtype).unsqueeze(1)  # [Nq,1]
        q_pe = torch.zeros(Nq, Dq, device=query_patches.device, dtype=query_patches.dtype)             # [Nq,D]
        q_pe[:, 0::2] = torch.sin(q_idx * q_div)
        q_pe[:, 1::2] = torch.cos(q_idx * q_div)
        query_patches = query_patches + q_pe.unsqueeze(0)
        # END OF SIN POS
        
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
