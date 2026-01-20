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

DATASET_PATH = "./out_first_step/"
OUTPUT_PATH = "./out_second_step/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

INPUT_HEIGHT = 2048
INPUT_WIDTH = 4096

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "../models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

#### cutout utils

def get_tile(img, params, height, width, fov_x, fov_y):
    return py360convert.e2p(img, (fov_x, fov_y), params['yaw'], params['pitch'], (height, width))

def xy_to_yaw_pitch(x, y):
    yaw = (x / INPUT_WIDTH) * 360.0 - 180.0
    pitch = 90.0 - (y / INPUT_HEIGHT) * 180.0
    return yaw, pitch

#### mast3r prediction
def get_mast3r_output(first, second):
    images = load_images([first, second], size=512)
    
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    # if "standing" image we want standing cutout
    shp = view2['true_shape'].cpu().numpy()
    query_h = shp[0, 0]
    query_w = shp[0, 1]
    
    cutout_h = 512
    cutout_w = 256
    fov_x = 45
    fov_y = 90
    
    if query_h < query_w:
        cutout_h = 256
        cutout_w = 512
        fov_x = 90
        fov_y = 45
    
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                device=device, dist='dot', block_size=2**13)

    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    
    ### ransac
    try:
        _, inliers = cv2.findHomography(matches_im1,
                                    matches_im0,
                                    method=cv2.RANSAC,
                                    ransacReprojThreshold=3.0)


        inlier_mask = inliers.ravel().astype(bool)
        matches_im0 = matches_im0[inlier_mask]
        matches_im1  = matches_im1[inlier_mask]
    except: # if there are less then 4 pairs
        pass

    return cutout_h, cutout_w, query_h, query_w, fov_x, fov_y, matches_im0, matches_im1

#### main
if __name__ == '__main__':
    for panorama_file in os.listdir(DATASET_PATH):
        if not "_panorama.jpg" in panorama_file:
            continue
        
        base_name = panorama_file.replace("_panorama.jpg", "")
        
        cutout_h, cutout_w, query_h, query_w, fov_x, fov_y, matches_im0, matches_im1 = get_mast3r_output(DATASET_PATH + base_name + "_panorama.jpg", DATASET_PATH + base_name + "_query.jpg")
        
        ##### if we predict matches between whole panorama and query - median calculation
        median_x = np.median(matches_im0[:, 0])
        median_y = np.median(matches_im0[:, 1])
        
        ##### getting cutout based on median
        x, y = median_x * (INPUT_WIDTH/512) , median_y * (INPUT_HEIGHT/256)  # coordinates on whole panorama but it is resized by mast3r to thats why it is too small number

        img = Image.open(DATASET_PATH + panorama_file)
        img = np.asarray(img)

        yaw, pitch = xy_to_yaw_pitch(x, y)
        params = {
            'roll': 0.,
            'pitch': pitch,
            'yaw': yaw
        }
        tile = get_tile(img, params, cutout_h, cutout_w, fov_x, fov_y)
        tile = Image.fromarray(tile)
        tile.save(OUTPUT_PATH + base_name + "_cutout.jpg")
        
        # copy query
        shutil.copy(DATASET_PATH + base_name + "_query.jpg" , OUTPUT_PATH + base_name + "_query.jpg")
        
        ### get cutout - query prediction 
        _, _, _, _, _, _, matches_im0, matches_im1 = get_mast3r_output(OUTPUT_PATH + base_name + "_cutout.jpg", OUTPUT_PATH + base_name + "_query.jpg")

        # save 
        with open(DATASET_PATH + base_name + ".csv", "r") as f_in, open(OUTPUT_PATH + base_name + "_gt.csv", "w") as f_out:
            original_line = f_in.read().strip()
            extra_values = f"{yaw},{pitch},{cutout_w},{cutout_h},{query_w},{query_h},{fov_x},{fov_y}"
            f_out.write(original_line + "," + extra_values + "\n")
        
        with open(OUTPUT_PATH + base_name + "_matches.csv", "w") as f:
            f.write(json.dumps(matches_im0.tolist()) + "\n")
            f.write(json.dumps(matches_im1.tolist()) + "\n")
