import os
import shutil
import csv

IN_DATASET_PATH = "/Users/martinkubicka/Documents/DP/src/training/dataset/out_train_first_look_45"
OUT_DATASET_PATH = "/Users/martinkubicka/Documents/DP/final_datasets/geopose_train"

os.makedirs(OUT_DATASET_PATH, exist_ok=True)

for filename in os.listdir(IN_DATASET_PATH):
    in_file = os.path.join(IN_DATASET_PATH, filename)

    if filename.endswith("_panorama.jpg") or filename.endswith("_query.jpg"):
        out_file = os.path.join(OUT_DATASET_PATH, filename)
        shutil.copy(in_file, out_file)

    elif filename.endswith("_gt.csv"):
        out_csv_path = os.path.join(OUT_DATASET_PATH, filename)

        with open(in_file, "r", newline="") as csv_in, open(out_csv_path, "w", newline="") as csv_out:
            reader = csv.reader(csv_in)
            writer = csv.writer(csv_out)

            for row in reader:
                roll = float(row[2])
                roll = (((roll + 180.0) % 360.0) - 180.0) / 2.0
                row[2] = f"{roll:.6f}"
                writer.writerow(row)
