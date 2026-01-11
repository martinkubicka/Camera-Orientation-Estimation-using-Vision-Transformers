import os
import shutil

IN_PATH = "/Users/martinkubicka/Documents/DP/final_datasets/lar_full"
OUT_PATH = "/Users/martinkubicka/Documents/DP/final_datasets/lar_filtered"
TXT_PATH = "/Users/martinkubicka/Documents/DP/final_datasets/readme.txt"

os.makedirs(OUT_PATH, exist_ok=True)

with open(TXT_PATH, "r") as f:
    exclude_names = [line.strip() for line in f if line.strip()]

exclude_set = set(exclude_names)

for filename in os.listdir(IN_PATH):
    base = (
        filename.replace("_panorama.jpg", "")
        .replace("_query.jpg", "")
        .replace("_gt.csv", "")
    )

    if base in exclude_set:
        continue
    else:
        src = os.path.join(IN_PATH, filename)
        dst = os.path.join(OUT_PATH, filename)
        shutil.copy2(src, dst)
