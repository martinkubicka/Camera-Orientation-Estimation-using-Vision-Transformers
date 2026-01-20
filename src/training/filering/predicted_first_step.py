import os
import shutil

DATASET_PATH = '../dataset/out_second_step/'
TARGET_PATH = '../dataset/inncorrect_cutout/'
os.makedirs(TARGET_PATH, exist_ok=True)

def angular_diff(a, b):
    diff = (a - b + 180) % 360 - 180
    return diff

for file_name in os.listdir(DATASET_PATH):
    if not file_name.endswith("_gt.csv"):
        continue

    base_name = file_name.replace("_gt.csv", "")
    full_match_path = os.path.join(DATASET_PATH, file_name)

    with open(full_match_path, 'r') as f:
        pitch_gt, yaw_gt, _, _, yaw_cutout, pitch_cutout, cutout_width, cutout_height, _, _, cutout_fox_x, cutout_fov_y = f.readline().strip().split(",")
        
        pitch_gt = float(pitch_gt) - 90
        yaw_gt = float(yaw_gt) - 180

        pitch_cutout = float(pitch_cutout)
        yaw_cutout = float(yaw_cutout)
        cutout_fov_x = float(cutout_fox_x)
        cutout_fov_y = float(cutout_fov_y)

        delta_yaw = angular_diff(yaw_gt, yaw_cutout)
        delta_pitch = angular_diff(pitch_gt, pitch_cutout)

        inside = ( # we dont want on the edge results -> <=
            abs(delta_yaw) < cutout_fov_x / 2 and
            abs(delta_pitch) < cutout_fov_y / 2
        )
        
        if not inside:
            files_to_copy = [
                file_name,  # _gt.csv
                f"{base_name}_cutout.jpg",
                f"{base_name}_query.jpg",
                f"{base_name}_matches.csv"
            ]

            for fname in files_to_copy:
                src = os.path.join(DATASET_PATH, fname)
                dst = os.path.join(TARGET_PATH, fname)
                if os.path.exists(src):
                    shutil.move(src, dst)
                else:
                    print(f"Warning: {src} not found")

