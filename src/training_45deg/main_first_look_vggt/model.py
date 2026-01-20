import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import py360convert
import torchvision.transforms as T
import numpy as np

from vggt.models.vggt import VGGT

def load_vggt(model_name: str = "facebook/VGGT-1B", pretrained: bool = True, device=None) -> nn.Module:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT.from_pretrained(model_name).to(device)
    model.eval()
    return model

def resize_with_padding(image: Image.Image, target_size: tuple) -> Image.Image:
    image.thumbnail(target_size, Image.LANCZOS)
    padded = Image.new("RGB", target_size, (0, 0, 0))
    offset_x = (target_size[0] - image.width) // 2
    offset_y = (target_size[1] - image.height) // 2
    padded.paste(image, (offset_x, offset_y))
    return padded

class PoseRegressor(nn.Module):
    def __init__(self,
                 model_name: str = "facebook/VGGT-1B",
                 hidden_dim: int = 512,
                 pretrained: bool = True,
                 num_rows: int = 4,
                 num_cols: int = 8,
                 fov: tuple = (45, 45),
                 tile_size: tuple = (518, 518)):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backbone = load_vggt(model_name, pretrained=pretrained, device=self.device)
        self.tile_size = tile_size
        
        with torch.no_grad():
            # create a dummy input: batch=1, sequence=1, 3x518x518 normalized tensor
            dummy = torch.zeros(1, 1, 3, self.tile_size[0], self.tile_size[1], device=self.device)
            agg_tokens_list, _ = self.backbone.aggregator(dummy)
            last = agg_tokens_list[-1]
            if last.dim() == 4:  # [B, S, T, D]
                _, S, TT, D = last.shape
                inferred_dim = D
            elif last.dim() == 3:  # [B, S*T, D]
                _, ST, D = last.shape
                inferred_dim = D
            else:
                raise RuntimeError(f"Unexpected VGGT output shape during init: {last.shape}")
        self.embed_dim = inferred_dim    


        # Freeze VGGT parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.cross_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, batch_first=True)
        self.attn_pool = nn.Linear(self.embed_dim, 1)

        n_tiles = num_rows * num_cols
        self.pos_embed = nn.Parameter(torch.zeros(n_tiles, self.embed_dim))

        self.head = nn.Sequential(
            nn.Linear(2 * self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.fov = fov

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

        j = torch.arange(num_rows, dtype=torch.float32)
        self.pitch_vals = 90.0 - (j + 0.5) * (180.0 / num_rows)
        i = torch.arange(num_cols, dtype=torch.float32)
        self.yaw_vals = -180.0 + (i + 0.5) * (360.0 / num_cols)

    def _encode_with_vggt(self, images_tensor: torch.Tensor):
        """
        Given preprocessed images (B, C, H, W), run VGGT aggregator to get aggregated tokens.
        Returns: sequence tensor to be used for cross-attention, shape [B, T, D]
        """
        # VGGT expects input of shape (B, C, H, W) already normalized
        with torch.no_grad():
            # Depending on VGGT version, model.aggregator returns (aggregated_tokens_list, ps_idx)
            aggregated_tokens_list, ps_idx = self.backbone.aggregator(images_tensor)
            # Use the last aggregated tokens as a sequence representation
            # aggregated_tokens_list is typically a list per stage, each of shape [B, T, D]
            seq = aggregated_tokens_list[-1]  # [B, T, D]
        return seq

    def forward(self, query_img, pano):
        B = len(pano)
        n_tiles = self.num_rows * self.num_cols
        all_tiles = []
        for img in pano:
            pil = Image.open(img).convert("RGB")
            for pv in self.pitch_vals.tolist():
                for yv in self.yaw_vals.tolist():
                    tile = py360convert.e2p(np.array(pil), self.fov, float(yv), float(pv),
                                            (self.tile_size[1], self.tile_size[0]))
                    all_tiles.append(self.transform(Image.fromarray(tile)).to(query_img.device))
        pano_tiles = torch.stack(all_tiles, 0)  # [B*32,3,518,518]
        pano_tiles = pano_tiles.view(B, n_tiles, 3, self.tile_size[0], self.tile_size[1])  # (B, S=n_tiles, C, H, W)

        # Prepare query image: assume already PIL or tensor; adapt if necessary
        if isinstance(query_img, Image.Image):
            query_tensor = self.transform(query_img).unsqueeze(0).to(self.device)  # [1,3,H,W]
        else:
            query_tensor = query_img  # assume preprocessed

        if query_tensor.dim() == 4:
            query_tensor = query_tensor.unsqueeze(1)

        # VGGT encoding
        pano_seq_raw = self._encode_with_vggt(pano_tiles)  # could be [B, n_tiles * T, D] or [B, n_tiles, T, D]
        if pano_seq_raw.dim() == 3:  # [B, n_tiles * T, D]
            _, total_tokens, D = pano_seq_raw.shape
            T = total_tokens // n_tiles
            pano_seq = pano_seq_raw.view(B, n_tiles, T, D)  # [B, n_tiles, T, D]
        elif pano_seq_raw.dim() == 4:  # [B, n_tiles, T, D]
            pano_seq = pano_seq_raw
            _, _, T, D = pano_seq.shape
        else:
            raise RuntimeError(f"Unexpected pano_seq_raw shape: {pano_seq_raw.shape}")

        # Add positional embedding per tile
        pos = self.pos_embed.unsqueeze(0).unsqueeze(2)  # [1, n_tiles, 1, D]
        pano_seq = pano_seq + pos  # [B, n_tiles, T, D]

        pano_emb = pano_seq.view(B, n_tiles * T, D)  # flatten for cross-attn

        # Query encoding
        query_patches = self._encode_with_vggt(query_tensor)  # could be [B, T_q, D] or [B, 1, T_q, D]
        if query_patches.dim() == 4:  # [B, S_q, T_q, D]
            B_q, S_q, T_q, D = query_patches.shape
            query_patches = query_patches.view(B_q, S_q * T_q, D)  # flatten sequence
        elif query_patches.dim() != 3:
            raise RuntimeError(f"Unexpected query_patches shape: {query_patches.shape}")

        # Cross-attention
        P_ca, _ = self.cross_attn(pano_emb, query_patches, query_patches)
        R_ca, _ = self.cross_attn(query_patches, pano_emb, pano_emb)
        pano_emb = pano_emb + P_ca
        query_patches = query_patches + R_ca

        # Attention pooling
        w_p = F.softmax(self.attn_pool(pano_emb).squeeze(-1), dim=1).unsqueeze(-1)
        w_q = F.softmax(self.attn_pool(query_patches).squeeze(-1), dim=1).unsqueeze(-1)
        pooled_p = (w_p * pano_emb).sum(1)       # [B, D]
        pooled_q = (w_q * query_patches).sum(1)  # [B, D]

        fused = torch.cat([pooled_p, pooled_q], dim=-1)  # [B,2D]
        angles = self.head(fused)                        # [B,3]
        return angles
