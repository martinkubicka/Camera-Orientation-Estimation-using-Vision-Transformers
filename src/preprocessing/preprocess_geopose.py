from pathlib import Path
from PIL import Image
import os
import numpy as np
import csv
import math

IN_PANOS_PATH = Path("../geoPose3K-panos/")
IN_DATASET_PATH = Path("../geoPose3K_final_publish/")
OUTPUT_PATH = Path("../GeoPose3K_processed")
OUTPUT_PATH.mkdir(exist_ok=True)

def normalize_angle(angle):
    return (angle + 180) % 360 - 180

def resize_with_padding(image, target_size):
    image.thumbnail(target_size, Image.LANCZOS)
    padded = Image.new("RGB", target_size, (0, 0, 0))
    offset_x = (target_size[0] - image.width) // 2
    offset_y = (target_size[1] - image.height) // 2
    padded.paste(image, (offset_x, offset_y))
    
    return padded

for base_name in os.listdir(IN_PANOS_PATH):
    try:
        for ext in [".jpg", ".jpeg", ".png"]:
            pano_file = IN_PANOS_PATH / base_name / "cyl" / f"pano{ext}"

            if pano_file.exists():
                out_path = OUTPUT_PATH / f"{base_name}_panorama.jpg"

                img = Image.open(pano_file).convert("RGB")
                img = resize_with_padding(img, (4096, 2048))

                img.save(out_path, "JPEG", quality=100, subsampling=0)
                break
    except Exception:
        print(f"Warning: PANORAMA not found for {base_name}")
        continue

    gt_file = IN_DATASET_PATH / base_name / "info.txt"

    with open(gt_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    yaw, pitch, roll = map(float, lines[1].split())
    fov = float(lines[5])
    
    fov = np.degrees(fov)
    
    yaw = math.degrees(math.pi - (yaw % (-2 * math.pi)))
    
    pitch = math.degrees(pitch)

    roll = math.degrees(-(roll % (-2 * math.pi)))
    roll = ((roll + 180.0) % 360.0 - 180.0) / 2.0

    yaw = normalize_angle(yaw)
    roll = normalize_angle(roll)
    
    out_csv = OUTPUT_PATH / f"{base_name}_gt.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([pitch, yaw, roll, fov])

    image_dir = IN_DATASET_PATH / base_name

    if image_dir.exists() and image_dir.is_dir():
        try:
            for ext in [".jpg", ".jpeg", ".png"]:
                photo_file = image_dir / f"photo{ext}"

                if photo_file.exists():
                    out_path = OUTPUT_PATH / f"{base_name}_query.jpg"

                    img = Image.open(photo_file).convert("RGB")
                    img = resize_with_padding(img, (512, 512))

                    img.save(out_path, "JPEG", quality=100, subsampling=0)
                    break

        except Exception:
            print(f"Warning: QUERY not found in {image_dir}")