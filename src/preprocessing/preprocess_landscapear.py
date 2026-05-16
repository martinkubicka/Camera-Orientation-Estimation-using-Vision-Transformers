
import os
import csv
from PIL import Image, ImageOps
import math

QUERY_SRC_PATH = "../query_photos"
PANO_SRC_BASE = "../new_pano_dataset"
CSV_INFO_PATH = os.path.join(PANO_SRC_BASE, "datasetInfoClean.csv")
OUT_PATH = "../lar_processed"

QUERY_SIZE = (512, 512)
PANO_SIZE = (4096, 2048)

os.makedirs(OUT_PATH, exist_ok=True)

def resize_with_padding(image_path, target_size, out_path):
    img = Image.open(image_path).convert("RGB")
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    img_padded = ImageOps.pad(img, target_size, color=(0, 0, 0))
    img_padded.save(out_path, quality=100)

for filename in os.listdir(QUERY_SRC_PATH):
    if not filename.lower().endswith(".jpg"):
        continue
    
    src_file = os.path.join(QUERY_SRC_PATH, filename)
    base_name = os.path.splitext(filename)[0]
    dst_file = os.path.join(OUT_PATH, f"{base_name}_query.jpg")

    resize_with_padding(src_file, QUERY_SIZE, dst_file)

for folder in os.listdir(PANO_SRC_BASE):
    if not folder.startswith("lsar_"):
        continue

    pano_path = os.path.join(PANO_SRC_BASE, folder, "cyl", "pano.png")
    if not os.path.exists(pano_path):
        continue

    base_name = folder.replace("lsar_", "")
    dst_file = os.path.join(OUT_PATH, f"{base_name}_panorama.jpg")

    resize_with_padding(pano_path, PANO_SIZE, dst_file)

with open(CSV_INFO_PATH, "r") as csv_in:
    reader = csv.reader(csv_in)

    for row in reader:
        filename = row[1].strip()
        yaw = math.degrees(float(row[-4]))
        pitch = math.degrees(float(row[-3]))
        roll = math.degrees(float(row[-2]))
        fov = math.degrees(float(row[-1]))

        roll = -roll
        yaw = 180 - yaw
        yaw = ((yaw + 180) % 360) - 180

        out_csv = os.path.join(OUT_PATH, f"{filename}_gt.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{pitch:.6f}",f"{yaw:.6f}",f"{roll:.6f}",f"{fov:.6f}"])
