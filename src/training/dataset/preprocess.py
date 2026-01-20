# 1. getting geopose dataset -> geoposequir (pano + gt) + non equirect second input file

import os
import shutil
from pathlib import Path

first_path = Path("/Users/martinkubicka/Documents/BP/src_old/preprocessing/geoPoseEquir")
second_path = Path("/Users/martinkubicka/Documents/geoPose3K_final_publish/")
output_path = Path("./out_first_step")
output_path.mkdir(exist_ok=True)

panorama_files = list(first_path.glob("*_panorama.jpg"))

for panorama_file in panorama_files:
    base_name = panorama_file.name.replace("_panorama.jpg", "")

    shutil.copy(panorama_file, output_path / panorama_file.name)

    csv_file = first_path / f"{base_name}.csv"
    if csv_file.exists():
        shutil.copy(csv_file, output_path / csv_file.name)
    else:
        print(f"Warning: CSV not found for {base_name}")

    image_dir = second_path / base_name
    if image_dir.exists() and image_dir.is_dir():
        for ext in [".jpg", ".jpeg", ".png"]:
            photo_file = image_dir / f"photo{ext}"
            if photo_file.exists():
                shutil.copy(photo_file, output_path / f"{base_name}_query.jpg")
                break
        else:
            print(f"Warning: No photo.[jpg|jpeg|png] found in {image_dir}")
    else:
        print(f"Warning: Directory {image_dir} does not exist")
