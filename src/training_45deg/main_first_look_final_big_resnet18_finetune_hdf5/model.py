import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models  # CHANGED
from PIL import Image
import py360convert
import torchvision.transforms as T
import numpy as np
# from transformers import AutoModel  # CHANGED (no longer needed)

def load_backbone(device):
    # model_name = "timm/vit_pe_spatial_tiny_patch16_512.fb"  # CHANGED
    encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # CHANGED
    encoder = nn.Sequential(*list(encoder.children())[:-2])  # CHANGED: keep spatial map (drop avgpool & fc)
    encoder = encoder.to(device)

    return encoder

class OrientationEstimator(nn.Module):
    def __init__(self,
                 device,
                 num_rows: int = 4,
                 num_cols: int = 8,
                 fov: tuple = (45, 45),
                 tile_size: tuple = (512, 512)):
        super().__init__()
        self.backbone = load_backbone(device)
        self.embed_dim = 512  # CHANGED (ResNet18 feature channels at conv head)

        self.cross_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, batch_first=True)
        self.attn_pool = nn.Linear(self.embed_dim, 1)

        # added
        self.norm_pano = nn.LayerNorm(self.embed_dim)
        self.norm_query = nn.LayerNorm(self.embed_dim)
        self.norm_post_pano = nn.LayerNorm(self.embed_dim)
        self.norm_post_query = nn.LayerNorm(self.embed_dim)
        self.norm_fused = nn.LayerNorm(2 * self.embed_dim)

        n_tiles = num_rows * num_cols
        self.pos_embed = nn.Parameter(torch.zeros(n_tiles, self.embed_dim))  # tile-level bias stays

        self.pitch_bins = 180
        self.yaw_bins   = 360
        self.roll_bins  = 360

        self.head_pitch = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim), 
            nn.ReLU(), 
            nn.Linear(self.embed_dim, self.pitch_bins)
        )
        self.head_yaw = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim), 
            nn.ReLU(), 
            nn.Linear(self.embed_dim, self.yaw_bins)
        )
        self.head_roll = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim), 
            nn.ReLU(), 
            nn.Linear(self.embed_dim, self.roll_bins)
        )

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.fov = fov
        self.tile_size = tile_size
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),  # CHANGED (ResNet normalization)
                        std=(0.229, 0.224, 0.225)),
        ])

        j = torch.arange(num_rows, dtype=torch.float32)
        self.pitch_vals = 90.0 - (j + 0.5) * (180.0 / num_rows)
        i = torch.arange(num_cols, dtype=torch.float32)
        self.yaw_vals = -180.0 + (i + 0.5) * (360.0 / num_cols)

    def forward(self, query_img, pano):
        B = len(pano)
        n_tiles = self.num_rows * self.num_cols     # e.g., 32

        # pano: [B, n_tiles, 3, H, W]
        # Extract tile batches
        feats_list = []

        for i in range(n_tiles):
            # pano[:, i] = [B, 3, H, W]
            feats_i = self.backbone(pano[:, i, :, :, :])   # → [B, 512, 16, 16]

            # Convert ResNet spatial map → tokens
            feats_i = feats_i.flatten(2).permute(0, 2, 1)   # → [B, 256, 512]

            feats_list.append(feats_i)

        # Stack tile tokens
        tile_patches = torch.stack(feats_list, dim=1)       # [B, n_tiles, 256, 512]

        # Extract dimensions
        B, n_tiles, Np, D = tile_patches.shape   # D=512, Np=256

        # Positional embeddings
        pos = self.pos_embed.unsqueeze(0).unsqueeze(2)      # [1, n_tiles, 1, D]
        pano_seq = tile_patches + pos                       # CHANGED: tile-level bias kept

        # ADDED: simple 1D sinusoidal pos enc for patch index (helps attention)
        pos_div = torch.exp(torch.arange(0, self.embed_dim, 2, device=pano_seq.device, dtype=pano_seq.dtype) *
                            (-torch.log(torch.tensor(10000.0, device=pano_seq.device, dtype=pano_seq.dtype)) / self.embed_dim))
        patch_idx = torch.arange(Np, device=pano_seq.device, dtype=pano_seq.dtype).unsqueeze(1)
        pe = torch.zeros(Np, self.embed_dim, device=pano_seq.device, dtype=pano_seq.dtype)
        pe[:, 0::2] = torch.sin(patch_idx * pos_div)
        pe[:, 1::2] = torch.cos(patch_idx * pos_div)
        pano_seq = pano_seq + pe.unsqueeze(0).unsqueeze(1)  # ADDED

        pano_emb = pano_seq.reshape(B, n_tiles * Np, self.embed_dim)  # CHANGED: many pano tokens

        # CHANGED ↓ query path keeps spatial tokens too
        feats_q = self.backbone(query_img)                      # [B, 512, H', W']
        query_patches = feats_q.flatten(2).permute(0, 2, 1)     # CHANGED: [B, Nq, 512]

        # ADDED: sinusoidal pos for query tokens
        Nq, Dq = query_patches.size(1), query_patches.size(2)
        q_div = torch.exp(torch.arange(0, Dq, 2, device=query_patches.device, dtype=query_patches.dtype) *
                          (-torch.log(torch.tensor(10000.0, device=query_patches.device, dtype=query_patches.dtype)) / Dq))
        q_idx = torch.arange(Nq, device=query_patches.device, dtype=query_patches.dtype).unsqueeze(1)
        q_pe = torch.zeros(Nq, Dq, device=query_patches.device, dtype=query_patches.dtype)
        q_pe[:, 0::2] = torch.sin(q_idx * q_div)
        q_pe[:, 1::2] = torch.cos(q_idx * q_div)
        query_patches = query_patches + q_pe.unsqueeze(0)       # ADDED
        
        pano_emb = self.norm_pano(pano_emb)
        query_patches = self.norm_query(query_patches)
        
        P_ca, _ = self.cross_attn(pano_emb, query_patches, query_patches)
        R_ca, _ = self.cross_attn(query_patches, pano_emb, pano_emb)
        pano_emb = self.norm_post_pano(pano_emb + P_ca)
        query_patches = self.norm_post_query(query_patches + R_ca)

        w_p = F.softmax(self.attn_pool(pano_emb).squeeze(-1), dim=1).unsqueeze(-1)
        w_q = F.softmax(self.attn_pool(query_patches).squeeze(-1), dim=1).unsqueeze(-1)
        pooled_p = (w_p * pano_emb).sum(1)
        pooled_q = (w_q * query_patches).sum(1)

        fused = torch.cat([pooled_p, pooled_q], dim=-1)
        fused = self.norm_fused(fused)

        logits_pitch = self.head_pitch(fused)
        logits_yaw   = self.head_yaw(fused)
        logits_roll  = self.head_roll(fused)
        
        return logits_pitch, logits_yaw, logits_roll
