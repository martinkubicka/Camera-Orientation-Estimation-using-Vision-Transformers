import sys
import os
# we can use main out of mast3r folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'mast3r')))
import numpy as np
import cv2
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from py360convert.utils import xyzpers, xyz2uv, uv2coor
import numpy as np
import torch
import torchvision.transforms.functional
from scipy.spatial.transform import Rotation as R

### PERSP point x,y TO EQ x,y (based on py360convert)
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

DATASET_PATH = "../dataset/out_second_step/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "../models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

count = 0
rot_err_sum = 0
min_dif_num = np.inf
max_dif_num = -np.inf
min_dif = [] # base_name, gt_pitch, gt_yaw, gt_roll ,predicted_pitch, preditcted_yaw, predicted_roll
max_dif = [] # base_name, gt_pitch, gt_yaw, gt_roll ,predicted_pitch, preditcted_yaw, predicted_roll

####### MAIN
if __name__ == '__main__':
    
    for panorama_file in os.listdir(DATASET_PATH):
        if not "cutout.jpg" in panorama_file:
            continue
        
        base_name = panorama_file.replace("_cutout.jpg", "")
        
        with open(DATASET_PATH + base_name + "_gt.csv", "r") as gt:
            pitch_gt, yaw_gt, roll_gt, fov_gt, yaw_cutout, pitch_cutout, cutout_width, cutout_height, query_h, query_w, cutout_fox_x, cutout_fov_y = gt.read().strip().split(",")
            
        images = load_images([DATASET_PATH + base_name + "_cutout.jpg", DATASET_PATH + base_name + "_query.jpg"], size=512)
        
        output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        
        pts3d_im0 = pred1['pts3d'].squeeze(0).detach().cpu().numpy() 
        pts3d_im1 = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy() 

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
        
        ## filtering 3d points
        if pts3d_im0.shape[2] != 3:
            pts3d_im0 = np.moveaxis(pts3d_im0, 0, -1)
            pts3d_im1 = np.moveaxis(pts3d_im1, 0, -1)

        cols = matches_im0[:, 0].astype(int)
        rows = matches_im0[:, 1].astype(int) 
        pts3d_im0 = pts3d_im0[rows, cols]  
        pts3d_im1 = pts3d_im1[rows, cols]  
        ###
        
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
        
        pts3d_im0 = pts3d_im0[valid_matches]
        pts3d_im1 = pts3d_im1[valid_matches]

        image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

        viz_imgs = []
        for i, view in enumerate([view1, view2]):
            rgb_tensor = view['img'] * image_std + image_mean
            viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        ### ransac
        try:
            M, inliers = cv2.findHomography(matches_im1,
                                        matches_im0,
                                        method=cv2.RANSAC,
                                        ransacReprojThreshold=3.0)
        except:
            print(base_name)
            print("Less then 4 points matched. \n --------")
            continue

        inlier_mask = inliers.ravel().astype(bool)
        matches_im0 = matches_im0[inlier_mask]
        matches_im1  = matches_im1[inlier_mask]
        
        pts3d_im0 = pts3d_im0[inlier_mask]
        pts3d_im1 = pts3d_im1[inlier_mask]
        
        ##### kabsch 
        # taken from: https://hunterheidenreich.com/posts/kabsch_algorithm/
        def kabsch_numpy(P, Q):
            """
            Computes the optimal rotation and translation to align two sets of points (P -> Q),
            and their RMSD.

            :param P: A Nx3 matrix of points
            :param Q: A Nx3 matrix of points
            :return: A tuple containing the optimal rotation matrix, the optimal
                    translation vector, and the RMSD.
            """
            assert P.shape == Q.shape, "Matrix dimensions must match"

            # Compute centroids
            centroid_P = np.mean(P, axis=0)
            centroid_Q = np.mean(Q, axis=0)

            # Optimal translation
            t = centroid_Q - centroid_P

            # Center the points
            p = P - centroid_P
            q = Q - centroid_Q

            # Compute the covariance matrix
            H = np.dot(p.T, q)

            # SVD
            U, S, Vt = np.linalg.svd(H)

            # Validate right-handed coordinate system
            if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
                Vt[-1, :] *= -1.0

            # Optimal rotation
            R = np.dot(Vt.T, U.T)

            # RMSD
            rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])

            return R, t, rmsd
        
        R_k, t, rmsd = kabsch_numpy(pts3d_im0, pts3d_im1)
        yaw, pitch, roll = R.from_matrix(R_k).as_euler('zyx', degrees=True)
        
        # to pixel
        x = (yaw + 180) / 360 * int(cutout_width)
        y = (90 - pitch) / 180 * int(cutout_height)
        
        # to equirectangular pixel
        x, y = get_eq(int(x), int(y), int(cutout_width), int(cutout_height), int(cutout_fox_x), int(cutout_fov_y), float(yaw_cutout), float(pitch_cutout))
        
        # to equirectangular pitch, yaw
        yaw = (x / 4096) * 360.0 - 180.0
        pitch = 90.0 - (y / 2048) * 180.0

        R_gt = R.from_euler('zyx', [yaw_gt, pitch_gt, roll_gt], degrees=True).as_matrix()
        R_est = R.from_euler('zyx', [pitch[0], yaw[0], roll], degrees=True).as_matrix()

        R_diff = R_gt.T @ R_est
        cos_angle = (np.trace(R_diff) - 1) / 2
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        error_rad = np.arccos(cos_angle)
        error_deg = np.degrees(error_rad)

        count += 1
        rot_err_sum += error_deg
        
        if rot_err_sum < min_dif_num:
            min_dif_num = rot_err_sum
            min_dif = [base_name, pitch_gt, yaw_gt, roll_gt, pitch[0], yaw[0], roll]
            
        if rot_err_sum > max_dif_num:
            max_dif_num = rot_err_sum
            max_dif = [base_name, pitch_gt, yaw_gt, roll_gt, pitch[0], yaw[0], roll]

        # print(f"Rotation error (deg): {error_deg:.4f}")


print("---- RESULTS ----")
print("min err", min_dif_num, min_dif)
print("max err", max_dif_num, max_dif)
print("mean rot err", rot_err_sum / count)    
