# 2. using main_get_areas.py for getting matches, cutout and copying second image + ground truth

# _gt.csv: degrees format - pitch, yaw, roll, fov (gt whole pano), pitch_cutout, yaw_cutout, cutout_width, cutout_height, query_h, query_w (after mast3r resize), cutout_fox_x, cutout_fov_y
# _matches.csv - first line matches cutout, second line matches query ([[x,y], [x,y]] both lines)

import sys
import os
# we can use main out of mast3r folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'mast3r')))
import numpy as np
import cv2
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
import numpy as np
import torch
import torchvision.transforms.functional
from PIL import Image
import py360convert
import shutil
import json
import random
from py360convert.utils import xyzpers, xyz2uv, uv2coor

DATASET_PATH = "./out_train/"
OUTPUT_PATH = "./out_train_correct_cutouts45/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

NUM_CUTOUTS_PER_PANO = 10

INPUT_HEIGHT = 2048
INPUT_WIDTH = 4096

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_name = "../models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
# model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

#### cutout utils

def get_tile(img, params, height, width, fov_x, fov_y):
    return py360convert.e2p(img, (fov_x, fov_y), params['yaw'], params['pitch'], (height, width))

def xy_to_yaw_pitch(x, y):
    yaw = (x / INPUT_WIDTH) * 360.0 - 180.0
    pitch = 90.0 - (y / INPUT_HEIGHT) * 180.0
    return yaw, pitch

def resize_with_padding(image, target_size):
    image.thumbnail(target_size, Image.LANCZOS)
    padded = Image.new("RGB", target_size, (0, 0, 0))
    offset_x = (target_size[0] - image.width) // 2
    offset_y = (target_size[1] - image.height) // 2
    padded.paste(image, (offset_x, offset_y))
    return padded

def get_biased_cutout_center(pitch_gt, yaw_gt, fov_y, fov_x, max_shift_pitch=15, max_shift_yaw=35):
    max_pitch_shift = min(max_shift_pitch, fov_y / 2)
    max_yaw_shift = min(max_shift_yaw, fov_x / 2)

    pitch_offset = random.uniform(-max_pitch_shift, max_pitch_shift)
    yaw_offset = random.uniform(-max_yaw_shift, max_yaw_shift)

    pitch_cutout = pitch_gt + pitch_offset
    yaw_cutout = yaw_gt + yaw_offset

    yaw_cutout = (yaw_cutout + 180) % 360 - 180

    pitch_cutout = max(-90, min(90, pitch_cutout))

    return pitch_cutout, yaw_cutout

def get_eq(px, py, out_w, out_h, fovx_deg, fovy_deg, yaw_deg, pitch_deg):
    W, H = 4096, 2048 

    fovx = np.deg2rad(fovx_deg)
    fovy = np.deg2rad(fovy_deg)
    yaw = -np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    in_rot = 0.0

    xyz_map = xyzpers(fovx, fovy, yaw, pitch, (out_h, out_w), in_rot)
    vec = xyz_map[py, px]

    u, v = xyz2uv(vec)

    u_eq, v_eq = uv2coor(u, v, H, W)
    return [u_eq, v_eq]

def pixel_to_pitch_yaw(x, y, W, H):
    yaw = (x / W) * 360.0 - 180.0
    pitch = 90.0 - (y / H) * 180.0
    return pitch, yaw

def pitch_yaw_to_pixel(pitch, yaw, width, height):
    x = (yaw + 180.0) / 360.0 * width
    y = (90.0 - pitch) / 180.0 * height
    return x, y

#### main
if __name__ == '__main__':
    for panorama_file in os.listdir(DATASET_PATH):
        if not "_panorama.jpg" in panorama_file:
            continue
        
        base_name = panorama_file.replace("_panorama.jpg", "")
        
        with open(DATASET_PATH + base_name + ".csv", "r") as f_in:
            original_line = f_in.read().strip()
            pitch_gt, yaw_gt, roll_gt, fov_gt = original_line.split(",")
            pitch_gt = float(pitch_gt) - 90
            yaw_gt = float(yaw_gt) - 180
        
        img = Image.open(DATASET_PATH + base_name + "_query.jpg")
        query_w, query_h = img.size
        
        target_size = (518, 518)
        cutout_h = 518
        cutout_w = 518
        fov_x = 90
        fov_y = 90
        
        if query_h < query_w:
            target_size = (518, 518)
            cutout_h = 518
            cutout_w = 518
            fov_x = 90
            fov_y = 90
            
        result_img = resize_with_padding(img, target_size)
        
        img = Image.open(DATASET_PATH + panorama_file)
        img = np.asarray(img)
        
        for i in range(NUM_CUTOUTS_PER_PANO):
            result_img.save(OUTPUT_PATH + base_name + "_" + str(i) +  "_query.jpg")
            
            pitch, yaw = get_biased_cutout_center(pitch_gt, yaw_gt, fov_y, fov_x)
        
            params = {
                'roll': 0.,
                'pitch': pitch,
                'yaw': yaw
            }
            
            tile = get_tile(img, params, cutout_h, cutout_w, fov_x, fov_y)
            tile = Image.fromarray(tile)
            tile.save(OUTPUT_PATH + base_name + "_" + str(i) + "_cutout.jpg")
            
            ### get cutout - query prediction 
            # _, _, _, _, _, _, matches_im0, matches_im1 = get_mast3r_output(OUTPUT_PATH + base_name + "_cutout.jpg", OUTPUT_PATH + base_name + "_query.jpg")

            # save 
            with open(OUTPUT_PATH + base_name + "_" + str(i) + "_gt.csv", "w") as f_out:
                extra_values = f"{pitch_gt},{yaw_gt},{roll_gt},{fov_gt},{yaw},{pitch},{cutout_w},{cutout_h},{query_w},{query_h},{fov_x},{fov_y}"
                f_out.write(extra_values + "\n")
            
            # with open(OUTPUT_PATH + base_name + "_matches.csv", "w") as f:
            #     f.write(json.dumps(matches_im0.tolist()) + "\n")
            #     f.write(json.dumps(matches_im1.tolist()) + "\n")
