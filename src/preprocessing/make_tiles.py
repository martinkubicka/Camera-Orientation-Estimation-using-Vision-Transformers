import os
import pytorch360convert
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import ToTensor
import h5py
import csv

IN_DATASET_PATH = "../training/dataset/geopose_test_big512/"
OUT_HDF5_PATH = "../training/dataset/geopose_test_big512.h5"

# Create output folder if needed
os.makedirs(os.path.dirname(OUT_HDF5_PATH), exist_ok=True)

# Tile and projection settings
num_rows = 4
num_cols = 8
tile_size = (512, 512)
fov = (45, 45)

# Define pitch/yaw values
j = torch.arange(num_rows, dtype=torch.float32)
pitch_vals = 90.0 - (j + 0.5) * (180.0 / num_rows)
i = torch.arange(num_cols, dtype=torch.float32)
yaw_vals = -180.0 + (i + 0.5) * (360.0 / num_cols)

# Open one shared HDF5 file for writing
with h5py.File(OUT_HDF5_PATH, "w") as h5f:
    panorama_group = h5f.create_group("panoramas")

    for fname in os.listdir(IN_DATASET_PATH):
        if not fname.endswith("_panorama.jpg"):
            continue

        base_name = fname.replace("_panorama.jpg", "")
        img_path = os.path.join(IN_DATASET_PATH, fname)
        csv_path = os.path.join(IN_DATASET_PATH, f"{base_name}_gt.csv")
        query_path = os.path.join(IN_DATASET_PATH, f"{base_name}_query.jpg")

        # Read the ground truth CSV
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            gt_row = next(reader)
            gt_values = np.array([float(v) for v in gt_row], dtype=np.float32)

        # Read query image
        query_img = ToTensor()(Image.open(query_path).convert("RGB"))

        # Create a group for this panorama
        grp = panorama_group.create_group(base_name)
        grp.create_dataset("ground_truth", data=gt_values)
        grp.create_dataset("query_image", data=query_img.numpy(), compression="gzip")

        # Convert panorama to tensor
        pano_tensor = ToTensor()(Image.open(img_path).convert("RGB"))

        # Generate and save tiles
        tiles = []
        count = 0
        for pv in pitch_vals.tolist():
            for yv in yaw_vals.tolist():
                tile = pytorch360convert.e2p(pano_tensor, fov, float(yv), float(pv), (tile_size[1], tile_size[0]))
                tiles.append(tile.numpy())
                count += 1

        # Stack all tiles into one dataset
        tiles_np = np.stack(tiles, axis=0)  # shape: (num_tiles, 3, H, W)
        grp.create_dataset("tiles", data=tiles_np, compression="gzip")