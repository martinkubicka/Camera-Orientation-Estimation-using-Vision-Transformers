import argparse
import os
import shutil
from typing import Tuple, List
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
import py360convert

def write_ndarray_txt(path: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    shape = " ".join(str(x) for x in arr.shape)
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else arr.reshape(-1, 1)
    with open(path, "w") as f:
        f.write(f"# SHAPE {shape}\n")
        np.savetxt(f, flat, fmt="%.10f")

def load_dinov2(model_name: str, device: torch.device):
    model = torch.hub.load(
        "facebookresearch/dinov2", model_name, pretrained=True
    )
    model.eval().to(device)
    return model

@torch.no_grad()
def extract_patch_tokens(model, x: torch.Tensor) -> torch.Tensor:
    out = model.forward_features(x)
    return out["x_norm_patchtokens"] # [B, Np, D]

def make_pitch_yaw(num_rows: int, num_cols: int) -> Tuple[torch.Tensor, torch.Tensor]:
    j = torch.arange(num_rows, dtype=torch.float32)
    pitch_vals = 90.0 - (j + 0.5) * (180.0 / num_rows)
    i = torch.arange(num_cols, dtype=torch.float32)
    yaw_vals = -180.0 + (i + 0.5) * (360.0 / num_cols)
    return pitch_vals, yaw_vals


def tile_panorama(
    pano_path: str,
    num_rows: int,
    num_cols: int,
    fov: Tuple[float, float],
    tile_size: Tuple[int, int],
) -> List[Image.Image]:
    tiles: List[Image.Image] = []
    pitch_vals, yaw_vals = make_pitch_yaw(num_rows, num_cols)
    for pv in pitch_vals.tolist():
        for yv in yaw_vals.tolist():
            tile_np = py360convert.e2p(
                np.array(Image.open(pano_path).convert("RGB")),
                fov,
                float(yv),
                float(pv),
                (tile_size[1], tile_size[0]),  # (w, h)
            )
            tiles.append(Image.fromarray(tile_np))
    return tiles


# ---------- Main pipeline ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True,
                        help="Path to original dataset folder (with *_query.jpg, *_panorama.jpg, *_gt.csv).")
    parser.add_argument("--dst", type=str, required=True,
                        help="Output folder for *_query.txt, *_panorama_tiles.txt, *_gt.csv.")
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_rows", type=int, default=4)
    parser.add_argument("--num_cols", type=int, default=8)
    parser.add_argument("--fov_h", type=float, default=45.0)
    parser.add_argument("--fov_v", type=float, default=45.0)
    parser.add_argument("--tile_w", type=int, default=518)
    parser.add_argument("--tile_h", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=16, help="Embedding batch size for tiles.")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    device = torch.device(args.device)
    model = load_dinov2(args.model_name, device)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    bases = []
    for fname in os.listdir(args.src):
        if not fname.endswith("_panorama.jpg"):
            continue
        base = fname.replace("_panorama.jpg", "")
        q = os.path.join(args.src, f"{base}_query.jpg")
        p = os.path.join(args.src, f"{base}_panorama.jpg")
        g = os.path.join(args.src, f"{base}_gt.csv")
        if not (os.path.isfile(q) and os.path.isfile(g)):
            print(f"Skipping {base} (missing query or gt).")
            continue
        bases.append(base)

    print(f"Found {len(bases)} samples.")

    fov = (float(args.fov_h), float(args.fov_v))
    tile_size = (int(args.tile_h), int(args.tile_w))

    for base in tqdm(bases, desc="Precomputing"):
        query_path = os.path.join(args.src, f"{base}_query.jpg")
        pano_path  = os.path.join(args.src, f"{base}_panorama.jpg")
        gt_path    = os.path.join(args.src, f"{base}_gt.csv")

        out_q_txt  = os.path.join(args.dst, f"{base}_query.txt")
        out_p_txt  = os.path.join(args.dst, f"{base}_panorama_tiles.txt")
        out_gt_csv = os.path.join(args.dst, f"{base}_gt.csv")

        q_img = Image.open(query_path).convert("RGB")
        q_tensor = transform(q_img).unsqueeze(0).to(device)  # [1,3,H,W]
        q_tokens = extract_patch_tokens(model, q_tensor).squeeze(0).cpu().numpy()  # [Nq, D]
        write_ndarray_txt(out_q_txt, q_tokens)

        tiles = tile_panorama(
            pano_path,
            num_rows=args.num_rows,
            num_cols=args.num_cols,
            fov=fov,
            tile_size=tile_size,
        )

        tile_feats = []
        bs = max(1, int(args.batch_size))
        for i in range(0, len(tiles), bs):
            batch_imgs = tiles[i:i+bs]
            batch_t = torch.stack([transform(im) for im in batch_imgs], dim=0).to(device)
            tokens = extract_patch_tokens(model, batch_t)     # [m, Np, D]
            tile_feats.append(tokens.cpu())

        tiles_tokens = torch.cat(tile_feats, dim=0).numpy()   # [n_tiles, Np, D]
                
        write_ndarray_txt(out_p_txt, tiles_tokens)        

        shutil.copyfile(gt_path, out_gt_csv)

    print("Done.")

if __name__ == "__main__":
    main()

# python3 get_dinov2_emb_dataset.py \
#   --src ../../training/dataset/out_train_first_look_45_augmented \
#   --dst ../../training/dataset/out_train_first_look_45_augmented_embed \
#   --num_rows 4 --num_cols 8 --fov_h 45 --fov_v 45 --tile_w 518 --tile_h 518 \
#   --model_name dinov2_vitb14 --batch_size 32
