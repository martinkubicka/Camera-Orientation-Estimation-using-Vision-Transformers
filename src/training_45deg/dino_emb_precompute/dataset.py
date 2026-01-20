import os
from typing import Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_embedding_txt(path: str) -> np.ndarray:
    """
    Reads the custom .txt format written by preprocess_embeddings.py
    Returns np.ndarray with original shape.
    """
    with open(path, "r") as f:
        header = f.readline().strip()
        if not header.startswith("# SHAPE "):
            raise ValueError(f"{path} missing '# SHAPE ...' header")
        shape = tuple(int(x) for x in header.replace("# SHAPE ", "").split())
        data = np.loadtxt(f, dtype=np.float32)
    # Handle 1D/2D/3D restoration
    if len(shape) == 1:
        arr = data.reshape(shape[0])
    elif len(shape) == 2:
        arr = data.reshape(shape[0], shape[1])
    elif len(shape) == 3:
        arr = data.reshape(shape[0], shape[1], shape[2])
    else:
        # General fallback
        prod = 1
        for s in shape:
            prod *= s
        arr = data.reshape(prod).reshape(*shape)
    return arr


class OrientationEmbedDataset(Dataset):
    """
    Loads precomputed embeddings:
      - query: [Nq, D]
      - panorama tiles: [num_rows*num_cols, Np, D]
      - GT: pitch, yaw, roll (floats)

    Expected files in `root`:
      <id>_query.txt
      <id>_panorama_tiles.txt
      <id>_gt.csv
    """

    def __init__(
        self,
        root: str,
        num_rows: int = 4,
        num_cols: int = 8,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.root = root
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.device = torch.device(device)
        self.dtype = dtype

        self.items: List[Dict] = []
        for fname in os.listdir(root):
            if not fname.endswith("_query.txt"):
                continue
            base = fname.replace("_query.txt", "")
            q = os.path.join(root, f"{base}_query.txt")
            p = os.path.join(root, f"{base}_panorama_tiles.txt")
            g = os.path.join(root, f"{base}_gt.csv")
            if os.path.isfile(q) and os.path.isfile(p) and os.path.isfile(g):
                self.items.append({"base": base, "q": q, "p": p, "g": g})

        if not self.items:
            raise RuntimeError(f"No embedded samples found in {root}")

        # Optional: sanity check one sample to infer shapes
        sample_q = _read_embedding_txt(self.items[0]["q"])
        sample_p = _read_embedding_txt(self.items[0]["p"])
        assert sample_q.ndim == 2, "Query embedding must be [Nq, D]"
        assert sample_p.ndim == 3, "Panorama tiles embedding must be [n_tiles, Np, D]"
        assert sample_p.shape[0] == self.num_rows * self.num_cols, (
            f"n_tiles mismatch: got {sample_p.shape[0]}, "
            f"expected {self.num_rows*self.num_cols}"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]

        q_np = _read_embedding_txt(rec["q"])         # [Nq, D]
        p_np = _read_embedding_txt(rec["p"])         # [n_tiles, Np, D]

        # GT
        with open(rec["g"], "r") as f:
            row = f.readline().strip().split(",")
        pitch_gt, yaw_gt, roll_gt, _ = map(float, row)

        # tensors
        q = torch.from_numpy(q_np).to(self.dtype)
        p = torch.from_numpy(p_np).to(self.dtype)
        gt = torch.tensor([pitch_gt, yaw_gt, roll_gt], dtype=self.dtype)

        # NOTE: move to device in training loop to avoid pinning/unpinning during DataLoader workers
        return q, p, gt, rec["base"]

    # Convenience: pitch/yaw grids (so your model can rebuild positional context)
    def pitch_yaw_grids(self) -> Tuple[torch.Tensor, torch.Tensor]:
        j = torch.arange(self.num_rows, dtype=torch.float32)
        pitch_vals = 90.0 - (j + 0.5) * (180.0 / self.num_rows)
        i = torch.arange(self.num_cols, dtype=torch.float32)
        yaw_vals = -180.0 + (i + 0.5) * (360.0 / self.num_cols)
        return pitch_vals, yaw_vals

# returne pano v # [B, 32, Np, D] shape

# 2) Use dataset
# from dataset_embed import OrientationEmbedDataset
# ds = OrientationEmbedDataset("../../training/dataset/out_train_first_look_45_augmented_embed",
#                              num_rows=4, num_cols=8)
# q, p, gt, base = ds[0]
