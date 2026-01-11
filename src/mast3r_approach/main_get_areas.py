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
from numpy.typing import NDArray
from postprocessing.coord_p2e import persp_to_equirect_deg
from py360convert.utils import xyzpers, xyz2uv, uv2coor
import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

### PERSP point x,y TO EQ x,y (based on py360convert)
def get_eq(px, py):
    W, H = 4096, 2048 
    out_w, out_h = 256, 512 # TODO can be swapped (cutout)
    fovx_deg, fovy_deg = 45, 90 # TODO can be dynamic (y)
    yaw_deg, pitch_deg = 53.7890625, 8.4375 # TODO change me based estimated pitch, yaw in first step

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

### GMS FUNCTIONS
def single_scale(s, d, g):
    N = s.shape[0]
    s_norm = s / (s.max(axis=0) + 1e-7)
    d_norm = d / (d.max(axis=0) + 1e-7)

    idx_s = (np.floor(s_norm * g)).astype(int)
    idx_d = (np.floor(d_norm * g)).astype(int)
    flat = lambda x: x[:, 1] * g + x[:, 0]
    flat_s, flat_d = flat(idx_s), flat(idx_d)

    hist_s, hist_d = {}, {}
    for i in flat_s: hist_s[i] = hist_s.get(i, 0) + 1
    for i in flat_d: hist_d[i] = hist_d.get(i, 0) + 1

    offs = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]
    def neigh(i):
        x, y = i % g, i // g
        return [ (y+dy)*g + (x+dx)
                    for dx,dy in offs
                    if 0 <= x+dx < g and 0 <= y+dy < g ]

    support = np.array([min(sum(hist_s.get(j,0) for j in neigh(flat_s[k])),
                            sum(hist_d.get(j,0) for j in neigh(flat_d[k])))
                        for k in range(N)])
    return support >= np.median(support)

def gms_filter(src, dst, grid=20, multiscale=True):
    mask = single_scale(src, dst, grid)
    if multiscale:
        for g in (grid//2, grid//4):
            mask |= single_scale(src, dst, g)
    return mask

####### MAIN
if __name__ == '__main__':
    ##### get predictions based on example
    device = 'cuda'

    model_name = "../models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
        
    images = load_images(["../dataset/1_median.jpg", "../dataset/1_query.png"], size=512) # TODO paths
    
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
    
    # GMS
    # src_tile = matches_im0.astype(np.float32)
    # dst_tile = matches_im1.astype(np.float32)

    # mask_gms = gms_filter(src_tile, dst_tile, grid=20, multiscale=True)

    # matches_im0 = matches_im0[mask_gms]
    # matches_im1 = matches_im1[mask_gms]
    
    ### ransac
    # M, inliers = cv2.findHomography(matches_im1,
    #                             matches_im0,
    #                             method=cv2.RANSAC,
    #                             ransacReprojThreshold=3.0)
    M, inliers = cv2.estimateAffinePartial2D(
                            matches_im1, matches_im0,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=3
                        )


    inlier_mask = inliers.ravel().astype(bool)
    matches_im0 = matches_im0[inlier_mask]
    matches_im1  = matches_im1[inlier_mask]
    
    pts3d_im0 = pts3d_im0[inlier_mask]
    pts3d_im1 = pts3d_im1[inlier_mask]
    
    ##### if we predicate matches between whole panorama and query - median calculation
    # median_x = np.median(matches_im0[:, 0])
    # median_y = np.median(matches_im0[:, 1])
    # print(median_x, median_y)
    
    ###### visualize matches between cutout and query
    num_matches = matches_im0.shape[0]
    n_viz = num_matches
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    plt.figure()
    plt.imshow(img)
    cmap = plt.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        
    save_path = "../outputs_areas_testing/" + "cutout_query_matches.jpg"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    ##### recalculate matches on cutout to equirectangular panorama
    matches_im0_orig = []
    for x, y in matches_im0:
        matches_im0_orig.append(get_eq(x,y))
        
    matches_im0_orig = np.array(matches_im0_orig, dtype=np.float32)
    
    ##### homography on recalculated points
    # M, _ = cv2.findHomography(matches_im1, matches_im0_orig, 0) # homography w/o ransac on panorama
    M, _ = cv2.estimateAffinePartial2D(matches_im1, matches_im0_orig, 0)
    
    ##### save recalcuated points on panorama
    pan_img = cv2.imread("../dataset/1_panorama.jpg") # TODO path
    sec_img = cv2.imread("../dataset/1_query.png") # TODO path
    
    panorama = cv2.cvtColor(pan_img, cv2.COLOR_BGR2RGB)
    persp = cv2.cvtColor(sec_img, cv2.COLOR_BGR2RGB)
    persp = cv2.resize(persp, (416, 512), interpolation=cv2.INTER_LINEAR) # TODO change me based on resized query (query)
    
    h1, w1 = panorama.shape[:2]
    H = h1
    W = w1
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:h1, :w1] = panorama

    plt.figure(figsize=(12, 6))
    plt.imshow(canvas)
    plt.axis('off')

    for (x1, y1) in matches_im0_orig:
        plt.scatter([x1], [y1], color='yellow', s=2)

    plt.tight_layout()
    plt.show()

    save_path = "../outputs_areas_testing/" + "recalculated_matches_panorama_points.jpg"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    #################### pano to query recalculated visualization
    h1, w1 = panorama.shape[:2]
    h2, w2 = persp.shape[:2]
    H = max(h1, h2)
    W = w1 + w2

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:h1, :w1] = panorama
    canvas[:h2, w1:w1+w2] = persp

    persp_off = matches_im1.copy()
    persp_off[:, 0] += w1

    plt.figure(figsize=(12, 6))
    plt.imshow(canvas)
    plt.axis('off')

    for (x1, y1), (x2, y2) in zip(matches_im0_orig, persp_off):
        plt.plot([x1, [x2]], [y1, [y2]], '-', color='cyan', linewidth=1)
        plt.scatter([x1], [y1], color='yellow', s=2)

    plt.tight_layout()
    plt.show()

    save_path = "../outputs_areas_testing/" + "mapped_panorama_query.jpg"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    ########### apply homography and save resulting img
    frame_bgr = cv2.imread("../dataset/1_panorama.jpg") # TODO path
    sec_img = cv2.imread("../dataset/1_query.png") # TODO path

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    persp = cv2.cvtColor(sec_img, cv2.COLOR_BGR2RGB)
    persp = cv2.resize(persp, (416, 512), interpolation=cv2.INTER_LINEAR)  # TODO change me based on resized query

    H, W = frame_rgb.shape[:2]
    # persp_warped = cv2.warpPerspective(persp, M, (W, H))
    persp_warped = cv2.warpAffine(persp, M, (W, H))
    
    # # transparent query on panorama    
    alpha = 0.7
    mask = (persp_warped > 0).any(axis=2)
    blended = frame_rgb.copy()
    blended[mask] = cv2.addWeighted(persp_warped[mask], alpha, frame_rgb[mask], 1 - alpha, 0)

    # no transparency
    # mask = (persp_warped > 0).any(axis=2)
    # blended = frame_rgb.copy()
    # blended[mask] = persp_warped[mask]

    # save mapped query to panorama based on homography
    plt.figure(figsize=(10, 5))
    plt.axis('off')
    plt.imshow(blended)
    plt.tight_layout()
    save_path = "../outputs_areas_testing/applied_homography_query_on_panorama.jpg"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    ###### get pich, yaw based on original center of the query image and homography matrix    
    # from scipy.spatial.transform import Rotation as R

    # W, H = 4096, 2048
    # w_persp, h_persp = 416, 512 # TODO query dims
    
    # cx, cy = w_persp // 2, h_persp // 2
    # point = np.array([[[cx, cy]]], dtype=np.float32)

    # # transformed_point = cv2.perspectiveTransform(point, M)
    # transformed_point = cv2.transform(point, M)
    # x_new, y_new = transformed_point[0, 0]
    
    # yaw   = (x_new / W) * 360.0 - 180.0
    # pitch = 90.0 - (y_new / H) * 180.0
    # print(f"Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°")

    # ###### get roll based on three points
    # offset = 20

    # points = np.array([
    #     [cx, cy],
    #     [cx + offset, cy],
    #     [cx, cy + offset],
    # ], dtype=np.float32).reshape(-1, 1, 2)

    # transformed_points = cv2.perspectiveTransform(points, M).reshape(-1, 2)
    # transformed_points = cv2.transform(points, M).reshape(-1, 2)
    # center_eq, right_eq, down_eq = transformed_points

    # def equirectangular_to_vector(x, y, width, height):
    #     yaw_rad = (x / width) * 2 * np.pi - np.pi
    #     pitch_rad = np.pi / 2 - (y / height) * np.pi
    #     vx = np.cos(pitch_rad) * np.sin(yaw_rad)
    #     vy = np.sin(pitch_rad)
    #     vz = np.cos(pitch_rad) * np.cos(yaw_rad)
    #     return np.array([vx, vy, vz])

    # c_vec = equirectangular_to_vector(*center_eq, W, H)
    # r_vec = equirectangular_to_vector(*right_eq, W, H)
    # d_vec = equirectangular_to_vector(*down_eq, W, H)

    # z_axis = c_vec / np.linalg.norm(c_vec)
    # x_axis = r_vec - c_vec
    # x_axis -= np.dot(x_axis, z_axis) * z_axis
    # x_axis /= np.linalg.norm(x_axis)
    # y_axis = np.cross(z_axis, x_axis)

    # R_mat = np.stack([x_axis, y_axis, z_axis], axis=-1)
    # if np.linalg.det(R_mat) < 0:
    #     R_mat[:, 1] *= -1

    # rot = R.from_matrix(R_mat)
    # _, _, roll = rot.as_euler('YXZ', degrees=True)

    # print(f"Roll: {roll:.2f}°")
    
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
    x = (yaw + 180) / 360 * 256   # TODO cutout width
    y = (90 - pitch) / 180 * 512   # TODO cutout height
    
    # to equirectangular pixel
    x, y = get_eq(int(x), int(y))
    
    # to equirectangular pitch, yaw
    yaw = (x / 4096) * 360.0 - 180.0 # TODO panorama width
    pitch = 90.0 - (y / 2048) * 180.0 # TODO panorama height
    
    print(pitch, yaw, roll) # we dont have to recalculate roll
    
    # formula error calculation
    pitch_gt, yaw_gt, roll_gt = 9.4, 41.5, 0.7

    R_gt = R.from_euler('zyx', [yaw_gt, pitch_gt, roll_gt], degrees=True).as_matrix()
    R_est = R.from_euler('zyx', [yaw[0], pitch[0], roll], degrees=True).as_matrix()

    R_diff = R_gt.T @ R_est
    cos_angle = (np.trace(R_diff) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    error_rad = np.arccos(cos_angle)
    error_deg = np.degrees(error_rad)

    print(f"Rotation error (rad): {error_rad:.4f}")
    print(f"Rotation error (deg): {error_deg:.4f}")

    