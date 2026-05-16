import os
import cv2
import albumentations as A
import numpy as np
import random
import csv

N_AUGS = 6

DATASET_PATH = "../dataset/"
OUTPUT_PATH  = "../dataset_augmented/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

color_aug = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.4),
    A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.4),
])

def augment_keep_black(img_bgr: np.ndarray) -> np.ndarray:
    thresh = 5
    black_mask = (img_bgr[..., 0] < thresh) & \
                 (img_bgr[..., 1] < thresh) & \
                 (img_bgr[..., 2] < thresh)

    aug = color_aug(image=img_bgr)["image"]
    aug[black_mask] = 0
    return aug

for file in os.listdir(DATASET_PATH):
    if not file.endswith("_query.jpg"):
        continue

    img_path = os.path.join(DATASET_PATH, file)
    image = cv2.imread(img_path)
    base_name = file.replace("_query.jpg", "")

    pano_path = os.path.join(DATASET_PATH, f"{base_name}_panorama.jpg")
    gt_path = os.path.join(DATASET_PATH, f"{base_name}_gt.csv")

    pano_img = cv2.imread(pano_path)

    for i in range(N_AUGS):
        aug_img = augment_keep_black(image)
        flip_applied = random.random() < 0.5

        if flip_applied:
            aug_img = cv2.flip(aug_img, 1)
            pano_flipped = cv2.flip(pano_img, 1)
        else:
            pano_flipped = pano_img.copy()

        out_query = os.path.join(OUTPUT_PATH, f"{base_name}_{i}_query.jpg")
        out_pano  = os.path.join(OUTPUT_PATH, f"{base_name}_{i}_panorama.jpg")
        out_gt    = os.path.join(OUTPUT_PATH, f"{base_name}_{i}_gt.csv")

        cv2.imwrite(out_query, aug_img)
        cv2.imwrite(out_pano, pano_flipped)

        with open(gt_path, "r") as f_in:
            row = next(csv.reader(f_in))

        if flip_applied and len(row) >= 2:
            try:
                row[1] = f"{-float(row[1]):.6f}"
                row[2] = f"{-float(row[2]):.6f}"
            except ValueError:
                pass

        with open(out_gt, "w", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(row)
