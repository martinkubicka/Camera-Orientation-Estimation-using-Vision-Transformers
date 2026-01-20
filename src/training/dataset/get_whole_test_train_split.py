import os
import shutil
import random

DATASET_PATH = "./out_first_step/"
DATASET_TRAIN_PATH = "./out_train"
DATASET_TEST_PATH = "./out_test"

os.makedirs(DATASET_TRAIN_PATH, exist_ok=True)
os.makedirs(DATASET_TEST_PATH, exist_ok=True)

def copy_related_files(base_names, target_folder):
    for base in base_names:
        for suffix in ["_panorama.jpg", "_query.jpg", ".csv"]:
            filename = base + suffix
            
            src = os.path.join(DATASET_PATH, filename)
            dst = os.path.join(target_folder, filename)
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"Warning: {src} not found.")

names = []

if __name__ == '__main__':
    for file in os.listdir(DATASET_PATH):
        if file.endswith("_panorama.jpg"):
            base_name = file.replace("_panorama.jpg", "")
            names.append(base_name)

    random.shuffle(names)
    test_count = int(len(names) * 0.15)
    test_names = names[:test_count]
    train_names = names[test_count:]

    copy_related_files(train_names, DATASET_TRAIN_PATH)
    copy_related_files(test_names, DATASET_TEST_PATH)

    print(f"Split done: {len(train_names)} train, {len(test_names)} test.")
