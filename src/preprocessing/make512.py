import os
from PIL import Image
import shutil

IN_PATH = "/storage/brno2/home/xkubic45/DP/src/training/dataset/geopose_test_big518/"
OUT_PATH = "/storage/brno2/home/xkubic45/DP/src/training/dataset/geopose_test_big512/"

os.makedirs(OUT_PATH, exist_ok=True)

for fname in os.listdir(IN_PATH):
    if not fname.endswith("_panorama.jpg"):
        continue

    base = fname.replace("_panorama.jpg", "")
    panorama_path = os.path.join(IN_PATH, fname)
    query_path = os.path.join(IN_PATH, f"{base}_query.jpg")
    csv_path = os.path.join(IN_PATH, f"{base}_gt.csv")

    if os.path.exists(csv_path):
        shutil.copy(csv_path, os.path.join(OUT_PATH, f"{base}_gt.csv"))

    if os.path.exists(query_path):
        with Image.open(query_path) as img:
            img = img.resize((512, 512), Image.LANCZOS)
            img.save(os.path.join(OUT_PATH, f"{base}_query.jpg"), quality=95)

    if os.path.exists(panorama_path):
        with Image.open(panorama_path) as img:
            img = img.resize((4096, 2048), Image.LANCZOS)
            img.save(os.path.join(OUT_PATH, f"{base}_panorama.jpg"), quality=95)
