import os
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from py360convert.utils import xyzpers, xyz2uv, uv2coor
from model import PoseRegressor
from model_first_look import PoseRegressor as PoseSecond
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import py360convert

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

def angles_to_eq_batch(pred,
                       yaw_cutout, pitch_cutout,
                       cutout_w, cutout_h,
                       query_w, query_h,
                       cutout_fov_x, cutout_fov_y,
                       pano_W=4096, pano_H=2048):
    out = []

    for i in range(pred.size(0)):
        pitch, yaw, roll = pred[i].tolist()

        w_c, h_c   = int(cutout_w),  int(cutout_h)
        w_q, h_q   = query_w,   query_h
        fovx, fovy = cutout_fov_x, cutout_fov_y
        yaw_c, pitch_c = yaw_cutout, pitch_cutout

        pitch = ((pitch +  90) % 180) -  90  
        yaw = ((yaw + 180) % 360) - 180
        roll = roll % 360 

        px, py      = pitch_yaw_to_pixel(pitch, yaw, w_c, h_c)
        
        px = int(px)
        py = int(py)
        px = max(0, min(px, w_c - 1))
        py = max(0, min(py, h_c - 1))
        
        u_eq, v_eq  = get_eq(px, py, w_c, h_c, fovx, fovy, yaw_c, pitch_c)
        pitch_eq, yaw_eq = pixel_to_pitch_yaw(u_eq, v_eq, pano_W, pano_H)

        row_t = torch.tensor(
            [pitch_eq[0], yaw_eq[0], roll],
            dtype=pred.dtype,
            device=pred.device,
        )

        out.append(row_t)
        
    return torch.stack(out, dim=0)

def angles_normalize(pred):
    out = []

    for i in range(pred.size(0)):
        pitch, yaw = pred[i].tolist()

        pitch = ((pitch +  90) % 180) -  90  
        yaw = ((yaw + 180) % 360) - 180


        row_t = torch.tensor(
            [pitch, yaw],
            dtype=pred.dtype,
            device=pred.device,
        )

        out.append(row_t)
        
    return torch.stack(out, dim=0)

def get_tile(img, params, height, width, fov_x, fov_y):
    return py360convert.e2p(img, (fov_x, fov_y), params['yaw'], params['pitch'], (height, width))

DATASET_PATH = "./dataset/out_test_first_look/"

counter = 0
rot_err = 0
rot_errors = []

device  = "cuda" if torch.cuda.is_available() else "cpu"

model_second = PoseRegressor("dinov2_vitl14").to(device)
ckpt_path = "trained_models/model_just_geo_2_aug.pth"
checkpoint = torch.load(ckpt_path, map_location=device)
model_second.load_state_dict(checkpoint["model_state"])
model_second.to(device)
model_second.eval()

model_first = PoseSecond("dinov2_vitl14").to(device)
ckpt_path = "trained_models/model_geo_first_look_1_aug.pth"
checkpoint = torch.load(ckpt_path, map_location=device)
model_first.load_state_dict(checkpoint["model_state"])
model_first.to(device)
model_first.eval()

tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

for fname in os.listdir(DATASET_PATH):
    if not fname.endswith("_gt.csv"):
        continue

    base = fname.replace("_gt.csv", "")
    panorama_path = os.path.join(DATASET_PATH, f"{base}_panorama.jpg")
    query_path  = os.path.join(DATASET_PATH, f"{base}_query.jpg")
    csv_path    = os.path.join(DATASET_PATH, f"{base}_gt.csv")

    if not (os.path.isfile(query_path) and os.path.isfile(csv_path)):
        print(f"Missing query/csv for {base} – skipping")
        continue

    with open(csv_path, "r") as f:
        row = f.readline().strip().split(",")
    pitch_gt, yaw_gt, roll_gt, _ = map(float, row)
    
    query_img  = Image.open(query_path).convert("RGB")
    query_img  = tf(query_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model_first(query_img, [panorama_path])
    
    pred_eq_nd_first = angles_normalize(pred.cpu())
    pitch_first = pred_eq_nd_first[0, 0].item()
    yaw_first   = pred_eq_nd_first[0, 1].item()
    
    params = {
        'roll': 0.,
        'pitch': pitch_first,
        'yaw': yaw_first
    }
            
    pano_img  = Image.open(panorama_path)
    pano_img = np.asarray(pano_img)
    cutout_img = get_tile(pano_img, params, 518, 518, 45, 45)
    cutout_img = Image.fromarray(cutout_img).convert("RGB")
    cutout_img = tf(cutout_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model_second(query_img, cutout_img)
    
    pred_eq_nd = angles_to_eq_batch(
                pred.cpu(),
                yaw_first, pitch_first,
                int(518),  int(518),
                int(518),   int(518),
                45, 45,
            ).to(device)
    
    pitch = pred_eq_nd[0, 0].item()
    yaw   = pred_eq_nd[0, 1].item()
    roll  = pred_eq_nd[0, 2].item()
    
    R_gt = R.from_euler('zyx', [yaw_gt, pitch_gt, roll_gt], degrees=True).as_matrix()
    R_est = R.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_matrix()

    R_diff = R_gt.T @ R_est
    cos_angle = (np.trace(R_diff) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    error_rad = np.arccos(cos_angle)
    error_deg = np.degrees(error_rad)
    
    rot_err += error_deg
    rot_errors.append(error_deg)
    counter += 1
    
print("Mean rot error (deg):", rot_err / counter)

rot_errors = np.array(rot_errors)

bins = np.arange(0, 181, 1)
fractions = [(rot_errors <= t).sum() / len(rot_errors) for t in bins]

x_norm = bins / 180.0
auc = np.trapz(fractions, x_norm)
print(f"AUC (normalized to [0, 1]): {auc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(bins, fractions, label=f"AUC = {auc:.3f}", linewidth=2)
plt.xticks(np.arange(0, 181, 20))
plt.xlim(0, 180)
plt.ylim(0, 1.01)
plt.xlabel("Orientation error (°)")
plt.ylabel("Fraction of images")
plt.legend()
plt.tight_layout()
plt.savefig("orientation_error_pipeline.png", dpi=300)

