import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

DATASET_PATH = "../../training/dataset/geopose_lar_augmented_tiled/"

class OrientationDataset(Dataset):
    def __init__(self, root=DATASET_PATH, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.data = []

        for fname in os.listdir(root):
            if not fname.endswith("0_panorama.jpg"):
                continue

            base = fname.replace("0_panorama.jpg", "")
            panorama_path = os.path.join(root, fname)
            query_path  = os.path.join(root, f"{base}_query.jpg")
            csv_path    = os.path.join(root, f"{base}_gt.csv")

            if not (os.path.isfile(query_path) and os.path.isfile(csv_path)):
                print(f"Missing query/csv for {base} – skipping")
                continue

            with open(csv_path, "r") as f:
                row = f.readline().strip().split(",")
            pitch_gt, yaw_gt, roll_gt, _ = map(float, row)

            self.data.append({
                "query_path":  query_path,
                "panorama_path": panorama_path,
                "pitch_gt":    pitch_gt,
                "yaw_gt":      yaw_gt,
                "roll_gt":     roll_gt,
            })

        if not self.data:
            raise RuntimeError(f"No samples found in {root}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        query_img  = Image.open(item["query_path"]).convert("RGB")
        panorama_img = item["panorama_path"]

        if self.transform:
            query_img  = self.transform(query_img)
            # panorama_img = self.transform(panorama_img)

        return (
            query_img, panorama_img,
            item["pitch_gt"], item["yaw_gt"], item["roll_gt"]
        )    
    