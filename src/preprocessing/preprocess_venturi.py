
import math
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import os
import sys
sys.path.append(os.path.abspath(".."))
import visualization.py360convert_lib.Perspec2Equirec as P2E

# Paths
VENTURI_DATASET_PATH = "../venturi"
VENTURI_PANOS_PATH = "../venturi_panos"
OUTPUT_PATH = "../output_venturi_preprocessed"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def resize_with_padding(img, target_size):
    target_w, target_h = target_size
    w, h = img.size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2

    new_img.paste(img_resized, (paste_x, paste_y))
    
    return new_img

def load_rotation_matrix(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    C0 = root.find('C0')
    C1 = root.find('C1')
    C2 = root.find('C2')

    right = np.array([float(C0.get(f'x{i}')) for i in range(3)])
    up = np.array([float(C1.get(f'x{i}')) for i in range(3)])
    forward = np.array([float(C2.get(f'x{i}')) for i in range(3)])

    R = np.column_stack((right, up, forward))

    return R

def normalize_angle(angle):
    return (angle + 180) % 360 - 180

def rotation_matrix_to_angles(R):
    right   = R[:, 0]
    up      = R[:, 1]
    forward = R[:, 2]

    yaw = np.degrees(np.arctan2(forward[0], forward[1]))
    yaw += 90
    yaw = normalize_angle(yaw)

    pitch = -np.degrees(np.arcsin(forward[2]))

    roll = np.degrees(np.arctan2(right[2], up[2]))
    roll = -normalize_angle(roll)

    return pitch, yaw, roll


def get_camera_angles(xml_path):
    R = load_rotation_matrix(xml_path)
    pitch, yaw, roll = rotation_matrix_to_angles(R)
    
    return pitch, yaw, roll

def load_fov(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    fov_rad = float(root.find("field-of-view").get("value"))
    return math.degrees(fov_rad)

for folder in sorted(os.listdir(VENTURI_DATASET_PATH)):
    folder_path = os.path.join(VENTURI_DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    tmp = folder_path.split("/")[-1]
    tmp2 = tmp.split("_")[1]
    pano_tmp = "venturi_" + tmp2 + "_00000001/"
    
    pano_path = os.path.join(VENTURI_PANOS_PATH, pano_tmp, "pano.png")
    rot_path = os.path.join(folder_path, "rotationC2G.xml")
    img_path = os.path.join(folder_path, "image.jpg")
    data_path = os.path.join(folder_path, "data.xml")

    if not (os.path.exists(pano_path) and
            os.path.exists(rot_path) and
            os.path.exists(img_path) and
            os.path.exists(data_path)):
        continue

    try:
        pano = Image.open(pano_path).convert("RGB")
        pano_r = pano.resize((4096, 2048), Image.LANCZOS)
        w, h = pano.size
        pano_resized = resize_with_padding(pano, (4096, 2048))

        img = Image.open(img_path).convert("RGB")
        img_resized = resize_with_padding(img, (512, 512))
        
        pitch, yaw, roll = get_camera_angles(rot_path)
        fov = load_fov(data_path)

        # === VISUALIZE ===
        # equ = P2E.Perspective(img_path, FOV=fov, THETA=yaw, PHI=pitch, ROLL=roll)
        # proj_img, _ = equ.GetEquirec(height=h, width=w)

        # pano_np = np.array(pano)
        # pano_np = cv2.cvtColor(pano_np, cv2.COLOR_RGB2BGR)

        # proj_img = cv2.resize(proj_img, (w, h))

        # mask = np.any(proj_img > 10, axis=2)
        # mask3 = mask[:, :, None]

        # alpha = 0.5
        # overlay = pano_np.copy()
        # overlay[mask] = (
        #     pano_np[mask] * (1 - alpha) +
        #     proj_img[mask] * alpha
        # )

        # overlay = overlay.astype(np.uint8)

        # cv2.imshow("", overlay)
        # key = cv2.waitKey(0)
        # if key == ord('q'):  # quit
        #     break
        # elif key == ord('n'):  # next
        #     continue

        # =========================

        base = folder

        pano_resized.save(os.path.join(OUTPUT_PATH, f"{base}_panorama.jpg"), quality=100)
        img_resized.save(os.path.join(OUTPUT_PATH, f"{base}_query.jpg"), quality=100)

        with open(os.path.join(OUTPUT_PATH, f"{base}_gt.csv"), "w") as f:
            f.write(f"{pitch},{yaw},{roll},{fov}\n")

        print(f"Processed: {folder}")

    except Exception as e:
        print(f"Error in {folder}: {e}")

cv2.destroyAllWindows()
