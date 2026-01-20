import os
from PIL import Image
from torch.utils.data import Dataset

DATASET_PATH = "./dataset/out_train_augmented/"

class OrientationDataset(Dataset):
    def __init__(self, root=DATASET_PATH, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.data = []

        for fname in os.listdir(root):
            if not fname.endswith("_cutout.jpg"):
                continue

            base = fname.replace("_cutout.jpg", "")
            cutout_path = os.path.join(root, fname)
            query_path  = os.path.join(root, f"{base}_query.jpg")
            csv_path    = os.path.join(root, f"{base}_gt.csv")

            if not (os.path.isfile(query_path) and os.path.isfile(csv_path)):
                print(f"Missing query/csv for {base} – skipping")
                continue

            with open(csv_path, "r") as f:
                row = f.readline().strip().split(",")
            pitch_gt, yaw_gt, roll_gt, _, yaw_cut, pitch_cut, cw, ch, qh, qw, fovx, fovy = map(float, row)

            self.data.append({
                "query_path":  query_path,
                "cutout_path": cutout_path,
                "pitch_gt":    pitch_gt,
                "yaw_gt":      yaw_gt,
                "roll_gt":     roll_gt,
                "yaw_cutout":  yaw_cut,
                "pitch_cutout":pitch_cut,
                "cutout_w":    cw,
                "cutout_h":    ch,
                "query_h":     qh,
                "query_w":     qw,
                "cutout_fov_x":fovx,
                "cutout_fov_y":fovy,
            })

        if not self.data:
            raise RuntimeError(f"No samples found in {root}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        query_img  = Image.open(item["query_path"]).convert("RGB")
        cutout_img = Image.open(item["cutout_path"]).convert("RGB")

        if self.transform:
            query_img  = self.transform(query_img)
            cutout_img = self.transform(cutout_img)

        return (
            query_img, cutout_img,
            item["pitch_gt"], item["yaw_gt"], item["roll_gt"],
            item["yaw_cutout"], item["pitch_cutout"],
            item["cutout_w"], item["cutout_h"],
            item["query_h"], item["query_w"],
            item["cutout_fov_x"], item["cutout_fov_y"],
        )    
    