import os
import sys
sys.path.append(os.path.abspath(".."))
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from model.model import OrientationEstimator
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F

DATASET_PATH = "../geopose_test/"
DATASET_8K_PATH = "../geopose_test_8k/"
CHECKPOINT = "../checkpoints/best_model.pth"

def crop_from_yaw_with_padding(pano, yaw_deg, out_hw):
    H, W = pano.shape[:2]
    out_h, out_w = out_hw

    mask = np.any(pano > 0, axis=-1)
    ys, xs = np.where(mask)

    if len(xs) == 0:
        center_x = W / 2
    else:
        min_x = xs.min()
        max_x = xs.max()
        content_width = max_x - min_x

        yaw_norm = (yaw_deg + 180.0) / 360.0
        center_x = min_x + yaw_norm * content_width

    center_y = H / 2

    x0 = int(center_x - out_w / 2)
    x1 = int(center_x + out_w / 2)
    y0 = int(center_y - out_h / 2)
    y1 = int(center_y + out_h / 2)

    y0 = max(0, y0)
    y1 = min(H, y1)

    if x0 < 0 or x1 > W:
        pano_wrap = np.concatenate([pano, pano, pano], axis=1)
        x0 += W
        x1 += W
        crop = pano_wrap[y0:y1, x0:x1]
    else:
        crop = pano[y0:y1, x0:x1]

    return crop

def angles_normalize(pred):
    out = []
    for i in range(pred.size(0)):
        pitch, yaw, roll = pred[i].tolist()
        pitch = ((pitch +  90) % 180) -  90
        yaw = ((yaw  + 180) % 360) - 180
        roll = ((roll + 180) % 360) - 180
        row_t = torch.tensor([pitch, yaw, roll], dtype=pred.dtype, device=pred.device)
        out.append(row_t)
    
    return torch.stack(out, dim=0)

def angles_to_classes(pitch_deg, yaw_deg, roll_deg):
    pitch_idx = torch.clamp(torch.floor(pitch_deg + 90.0), 0, 179).long()
    yaw_idx = torch.remainder(torch.floor(yaw_deg + 180.0), 360).long()
    roll_idx = torch.remainder(torch.floor(roll_deg + 180.0), 360).long()
    
    return pitch_idx, yaw_idx, roll_idx

def classes_to_angles(pitch_idx, yaw_idx, roll_idx):
    pitch_deg = -90.0 + (pitch_idx.float() + 0.5)
    yaw_deg = -180.0 + (yaw_idx.float() + 0.5)
    roll_deg = -180.0 + (roll_idx.float() + 0.5)
    
    return pitch_deg, yaw_deg, roll_deg

def circ_expect_deg(prob, centers_deg):
    theta = torch.deg2rad(centers_deg).to(prob.device)
    x = torch.matmul(prob, torch.cos(theta))
    y = torch.matmul(prob, torch.sin(theta))
    ang_rad = torch.atan2(y, x)
    
    return torch.rad2deg(ang_rad)

def class_centers_deg(C, start_deg):
    idx = torch.arange(C, dtype=torch.float32)
    
    return start_deg + (idx + 0.5)

def ang_diff_deg(a, b, period):
    d = (a - b + period/2) % period - period/2
    
    return torch.abs(d)

counter = 0
rot_err = 0
rot_errors = []

device  = "cuda" if torch.cuda.is_available() else "cpu"

tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

model = OrientationEstimator(tf=tf, device=device).to(device)

checkpoint = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

pitch_errors = []
yaw_errors = []
roll_errors = []

Cp, Cy, Cr = 180, 360, 360
cp = class_centers_deg(Cp, -90.0).to(device)
cy = class_centers_deg(Cy, -180.0).to(device)
cr = class_centers_deg(Cr, -180.0).to(device)

for fname in os.listdir(DATASET_PATH):
    if not fname.endswith("_gt.csv"):
        continue

    base = fname.replace("_gt.csv", "")
    panorama_path = os.path.join(DATASET_PATH, f"{base}_panorama.jpg")
    query_path = os.path.join(DATASET_PATH, f"{base}_query.jpg")
    csv_path = os.path.join(DATASET_PATH, f"{base}_gt.csv")

    if not (os.path.isfile(query_path) and os.path.isfile(csv_path) and os.path.isfile(panorama_path)):
        print(f"Missing query/csv for {base} - skipping")
        continue

    with open(csv_path, "r") as f:
        row = f.readline().strip().split(",")
    pitch_gt, yaw_gt, roll_gt, _ = map(float, row)

    query_img = Image.open(query_path).convert("RGB")
    query_img = tf(query_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_p, logits_y, logits_r = model(query_img, [panorama_path])

        prob_p = F.softmax(logits_p, dim=1)
        prob_y = F.softmax(logits_y, dim=1)
        prob_r = F.softmax(logits_r, dim=1)
        pred_p_deg_exp = circ_expect_deg(prob_p, cp).cpu()
        pred_y_deg_exp = circ_expect_deg(prob_y, cy).cpu()
        pred_r_deg_exp = circ_expect_deg(prob_r, cr).cpu()
        
    with torch.no_grad():
        # first pass
        logits_p, logits_y, logits_r = model(query_img, [panorama_path])

        prob_p = F.softmax(logits_p, dim=1)
        prob_y = F.softmax(logits_y, dim=1)
        prob_r = F.softmax(logits_r, dim=1)

        pred_p_deg_exp = circ_expect_deg(prob_p, cp).cpu()
        pred_y_deg_exp = circ_expect_deg(prob_y, cy).cpu()
        pred_r_deg_exp = circ_expect_deg(prob_r, cr).cpu()

        pitch_first = float(pred_p_deg_exp[0])
        yaw_first   = float(pred_y_deg_exp[0])
        roll_first  = float(pred_r_deg_exp[0])

        # second pass
        panorama_path = DATASET_8K_PATH + panorama_path.split("/")[-1]
        pano_np = np.array(Image.open(panorama_path).convert("RGB"))

        tile = crop_from_yaw_with_padding(
            pano_np,
            yaw_first,
            (2048, 4096)
        )

        tile_pil = Image.fromarray(tile)
        tile_pil.save("./tmp.jpg")

        tile_tensor = tf(tile_pil).unsqueeze(0).to(device)

        logits_p2, logits_y2, logits_r2 = model(
            query_img,
            ["./tmp.jpg"]
        )

        prob_p2 = F.softmax(logits_p2, dim=1)
        prob_y2 = F.softmax(logits_y2, dim=1)
        prob_r2 = F.softmax(logits_r2, dim=1)

        pred_p2 = circ_expect_deg(prob_p2, cp).cpu()
        pred_y2 = circ_expect_deg(prob_y2, cy).cpu()
        pred_r2 = circ_expect_deg(prob_r2, cr).cpu()

        pitch_pred_deg = pitch_first
        yaw_pred_deg = yaw_first + float(pred_y2[0])
        roll_pred_deg = roll_first
        
        if yaw_pred_deg > 180:
            yaw_pred_deg -= 360
        if yaw_pred_deg < -180:
            yaw_pred_deg += 360
            
    yaw_gt_t   = torch.tensor(yaw_gt, dtype=torch.float32)
    pitch_gt_t = torch.tensor(pitch_gt, dtype=torch.float32)
    roll_gt_t  = torch.tensor(roll_gt, dtype=torch.float32)

    yaw_err   = ang_diff_deg(yaw_pred_deg, yaw_gt_t, 360.0)
    pitch_err = ang_diff_deg(pitch_pred_deg, pitch_gt_t, 180.0)
    roll_err  = ang_diff_deg(roll_pred_deg, roll_gt_t, 360.0)

    yaw_errors.append(yaw_err.item())
    pitch_errors.append(pitch_err.item())
    roll_errors.append(roll_err.item())

    err_for_acc = torch.stack([
        ang_diff_deg(yaw_pred_deg,   yaw_gt_t,   360.0),
        ang_diff_deg(pitch_pred_deg, pitch_gt_t, 180.0),
        ang_diff_deg(roll_pred_deg,  roll_gt_t,  360.0),
    ], dim=0)

    R_gt  = R.from_euler('zyx', [yaw_gt,  pitch_gt,  roll_gt ], degrees=True).as_matrix()
    R_est = R.from_euler('zyx', [float(yaw_pred_deg), float(pitch_pred_deg), float(roll_pred_deg)], degrees=True).as_matrix()

    R_diff = R_gt.T @ R_est
    cos_angle = (np.trace(R_diff) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    error_rad = np.arccos(cos_angle)
    error_deg = np.degrees(error_rad)

    rot_err += error_deg
    rot_errors.append(error_deg)
    counter += 1
    print(counter, error_deg, panorama_path)

print("Mean rot error (deg):", rot_err / counter)

pitch_errors = torch.tensor(pitch_errors)
avg_pitch_error = pitch_errors.mean().item()
median_pitch_error = pitch_errors.median().item()
print(f"Average Pitch Error: {avg_pitch_error:.2f}°")
print(f"Median Pitch Error: {median_pitch_error:.2f}°")

yaw_errors = torch.tensor(yaw_errors)
avg_yaw_error = yaw_errors.mean().item()
median_yaw_error = yaw_errors.median().item()
print(f"Average yaw Error: {avg_yaw_error:.2f}°")
print(f"Median Yaw Error: {median_yaw_error:.2f}°")

roll_errors = torch.tensor(roll_errors)
avg_roll_error = roll_errors.mean().item()
median_roll_error = roll_errors.median().item()
print(f"Average roll Error: {avg_roll_error:.2f}°")
print(f"Median Roll Error: {median_roll_error:.2f}°")

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
plt.savefig("auc.png", dpi=300)
