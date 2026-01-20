import os
import shutil

DATASET_PATH = '../dataset/out_second_step/'
TARGET_PATH = '../dataset/few_matches'
os.makedirs(TARGET_PATH, exist_ok=True)

for file_name in os.listdir(DATASET_PATH):
    if not file_name.endswith("_matches.csv"):
        continue

    base_name = file_name.replace("_matches.csv", "")
    full_match_path = os.path.join(DATASET_PATH, file_name)

    with open(full_match_path, 'r') as f:
        first_line = f.readline().strip()

    clean_line = first_line.strip('[]')
    if not clean_line:
        count = 0
    else:
        pairs = clean_line.split('], [')
        count = len(pairs)

    if count < 1:
        files_to_copy = [
            file_name,  # _matches.csv
            f"{base_name}_cutout.jpg",
            f"{base_name}_query.jpg",
            f"{base_name}_gt.csv"
        ]

        for fname in files_to_copy:
            src = os.path.join(DATASET_PATH, fname)
            dst = os.path.join(TARGET_PATH, fname)
            if os.path.exists(src):
                shutil.move(src, dst)
            else:
                print(f"Warning: {src} not found")

