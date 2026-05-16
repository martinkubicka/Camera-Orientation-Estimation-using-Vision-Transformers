import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import py360convert
import torchvision.transforms as T
import numpy as np
from transformers import AutoModel

def load_backbone(device):
    model_name = "timm/vit_pe_spatial_tiny_patch16_512.fb"
    encoder = AutoModel.from_pretrained(model_name)
    encoder = encoder.to(device)
    return encoder

class OrientationEstimator(nn.Module):
    def __init__(self, tf, device, num_rows=4, num_cols=8, fov=(45, 45), tile_size=(512, 512)):
        super().__init__()

        self.transform = tf
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.fov = fov
        self.tile_size = tile_size

        self.backbone = load_backbone(device)
        self.embed_dim = self.backbone.config.num_features

        self.cross_attn = nn.MultiheadAttention(self.embed_dim, num_heads=8, batch_first=True)
        self.attn_pool = nn.Linear(self.embed_dim, 1)

        self.norm_pano = nn.LayerNorm(self.embed_dim)
        self.norm_query = nn.LayerNorm(self.embed_dim)
        self.norm_post_pano = nn.LayerNorm(self.embed_dim)
        self.norm_post_query = nn.LayerNorm(self.embed_dim)
        self.norm_fused = nn.LayerNorm(2 * self.embed_dim)

        n_tiles = num_rows * num_cols
        self.pos_embed = nn.Parameter(torch.zeros(n_tiles, self.embed_dim))

        self.head_pitch = nn.Sequential(nn.Linear(2*self.embed_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 180))
        self.head_yaw   = nn.Sequential(nn.Linear(2*self.embed_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 360))
        self.head_roll  = nn.Sequential(nn.Linear(2*self.embed_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 360))

        j = torch.arange(num_rows, dtype=torch.float32)
        self.pitch_vals = 90.0 - (j + 0.5) * (180.0 / num_rows)

        i = torch.arange(num_cols, dtype=torch.float32)
        self.yaw_vals = -180.0 + (i + 0.5) * (360.0 / num_cols)


    def forward(self, query_img, pano, return_attention=True):

        B = len(pano)
        n_tiles = self.num_rows * self.num_cols

        all_tiles = []
        orig_tiles = []

        for img in pano:
            pano_np = np.array(Image.open(img).convert("RGB"))

            for pv in self.pitch_vals.tolist():
                for yv in self.yaw_vals.tolist():
                    tile = py360convert.e2p(
                        pano_np, self.fov, float(yv), float(pv),
                        (self.tile_size[1], self.tile_size[0])
                    )

                    orig_tiles.append(tile)
                    all_tiles.append(self.transform(tile).to(query_img.device))

        pano_tiles = torch.stack(all_tiles, 0)
        orig_tiles = np.stack(orig_tiles, 0)

        feats_p = self.backbone(pano_tiles)
        tile_patches = feats_p.last_hidden_state[:, 1:, :]

        Np = tile_patches.size(1)

        pano_seq = tile_patches.view(B, n_tiles, Np, self.embed_dim)

        pos = self.pos_embed.unsqueeze(0).unsqueeze(2)
        pano_seq = pano_seq + pos

        pano_emb = pano_seq.view(B, n_tiles * Np, self.embed_dim)

        feats_q = self.backbone(query_img)
        query_patches = feats_q.last_hidden_state[:, 1:, :]

        Nq = query_patches.size(1)

        pano_emb = self.norm_pano(pano_emb)
        query_patches = self.norm_query(query_patches)

        P_ca, attn_pq = self.cross_attn(pano_emb, query_patches, query_patches)
        R_ca, attn_qp = self.cross_attn(query_patches, pano_emb, pano_emb)

        pano_emb = self.norm_post_pano(pano_emb + P_ca)
        query_patches = self.norm_post_query(query_patches + R_ca)

        w_p = F.softmax(self.attn_pool(pano_emb).squeeze(-1), dim=1).unsqueeze(-1)
        w_q = F.softmax(self.attn_pool(query_patches).squeeze(-1), dim=1).unsqueeze(-1)

        pooled_p = (w_p * pano_emb).sum(1)
        pooled_q = (w_q * query_patches).sum(1)

        fused = torch.cat([pooled_p, pooled_q], dim=-1)
        fused = self.norm_fused(fused)

        return attn_qp, attn_pq, pano_tiles, orig_tiles, Np, Nq, self.head_pitch(fused), self.head_yaw(fused), self.head_roll(fused)
    