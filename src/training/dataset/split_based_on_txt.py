import shutil
from pathlib import Path

test_list_path = Path("geoPose3K_final_test.txt")
all_dir = Path("./out_all_first_look_45")
test_dir = Path("./out_test_first_look_45")
train_dir = Path("./out_train_first_look_45")

test_dir.mkdir(parents=True, exist_ok=True)
train_dir.mkdir(parents=True, exist_ok=True)

with test_list_path.open("r", encoding="utf-8") as f:
    test_prefixes = [line.strip() for line in f if line.strip()]

all_files = [
    p for p in all_dir.iterdir()
    if p.is_file() and p.name.endswith("_panorama.jpg")
]

for file_path in all_files:
    filename = file_path.name
    
    prefix = filename.replace("_panorama.jpg", "")
    
    if prefix in test_prefixes:
        shutil.copy2(file_path, test_dir / filename)
        shutil.copy2(file_path.with_name(filename.replace("_panorama.jpg", "_query.jpg")), test_dir / (prefix + "_query.jpg"))
        shutil.copy2(file_path.with_name(filename.replace("_panorama.jpg", "_gt.csv")), test_dir / (prefix + "_gt.csv"))
    else:
        shutil.copy2(file_path, train_dir / filename)
        shutil.copy2(file_path.with_name(filename.replace("_panorama.jpg", "_query.jpg")), train_dir / (prefix + "_query.jpg"))
        shutil.copy2(file_path.with_name(filename.replace("_panorama.jpg", "_gt.csv")), train_dir / (prefix + "_gt.csv"))
