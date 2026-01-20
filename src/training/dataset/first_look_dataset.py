from PIL import Image
import os

DATASET_PATH = "./out_test/"
OUT_PATH = "./out_test_first_look/"
os.makedirs(OUT_PATH, exist_ok=True)

def resize_with_padding(image, target_size):
    image.thumbnail(target_size, Image.LANCZOS)
    padded = Image.new("RGB", target_size, (0, 0, 0))
    offset_x = (target_size[0] - image.width) // 2
    offset_y = (target_size[1] - image.height) // 2
    padded.paste(image, (offset_x, offset_y))
    return padded

for panorama_file in os.listdir(DATASET_PATH):
    if not "_panorama.jpg" in panorama_file:
        continue
    
    base_name = panorama_file.replace("_panorama.jpg", "")
    
    with open(DATASET_PATH + base_name + ".csv", "r") as f_in:
        original_line = f_in.read().strip()
        pitch_gt, yaw_gt, roll_gt, fov_gt = original_line.split(",")
        pitch_gt = float(pitch_gt) - 90
        yaw_gt = float(yaw_gt) - 180
    
    with open(os.path.join(OUT_PATH, base_name + "_gt.csv"), "w") as f_out:
        f_out.write(f"{pitch_gt},{yaw_gt},{roll_gt},{fov_gt}")

    img = Image.open(DATASET_PATH + base_name + "_query.jpg")
    img = resize_with_padding(img, (518, 518))
    img.save(os.path.join(OUT_PATH, base_name + "_query.jpg"))
    

    img = Image.open(DATASET_PATH + base_name + "_panorama.jpg")
    img = img.resize((4144, 2072))
    img.save(os.path.join(OUT_PATH, base_name + "_panorama.jpg"))
