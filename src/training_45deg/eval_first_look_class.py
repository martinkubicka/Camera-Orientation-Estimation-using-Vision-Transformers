import os
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from py360convert.utils import xyzpers, xyz2uv, uv2coor
from main_first_look_regsmallclass.model import PoseRegressor # TODO
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F  # CHANGED: for softmax

def angles_normalize(pred):
    out = []
    for i in range(pred.size(0)):
        pitch, yaw, roll = pred[i].tolist()
        pitch = ((pitch +  90) % 180) -  90
        yaw   = ((yaw  + 180) % 360) - 180
        roll  = roll % 360
        row_t = torch.tensor([pitch, yaw, roll], dtype=pred.dtype, device=pred.device)
        out.append(row_t)
    return torch.stack(out, dim=0)

# ADDED: class-index <-> degree helpers for classification heads
def angles_to_classes(pitch_deg, yaw_deg, roll_deg):
    pitch_idx = torch.clamp(torch.floor(pitch_deg + 90.0), 0, 179).long()
    yaw_idx   = torch.remainder(torch.floor(yaw_deg + 180.0), 360).long()
    roll_idx  = torch.remainder(torch.floor(roll_deg), 360).long()
    return pitch_idx, yaw_idx, roll_idx

def classes_to_angles(pitch_idx, yaw_idx, roll_idx):
    pitch_deg = -90.0  + (pitch_idx.float() + 0.5)   # [B]
    yaw_deg   = -180.0 + (yaw_idx.float()   + 0.5)   # [B]
    roll_deg  = 0.0    + (roll_idx.float()  + 0.5)   # [B]
    return pitch_deg, yaw_deg, roll_deg

# ADDED: circular expected angle from probabilities
def circ_expect_deg(prob: torch.Tensor, centers_deg: torch.Tensor) -> torch.Tensor:
    theta = torch.deg2rad(centers_deg).to(prob.device)   # [C]
    x = torch.matmul(prob, torch.cos(theta))             # [B]
    y = torch.matmul(prob, torch.sin(theta))             # [B]
    ang_rad = torch.atan2(y, x)                          # [-pi, pi)
    return torch.rad2deg(ang_rad)                        # (-180, 180]

# ADDED: centers of 1° bins (bin centers end with .5°)
def class_centers_deg(C, start_deg):
    idx = torch.arange(C, dtype=torch.float32)
    return start_deg + (idx + 0.5)

# Same as before
def ang_diff_deg(a, b, period):
    d = (a - b + period/2) % period - period/2
    return torch.abs(d)

DATASET_PATH = "/storage/brno2/home/xkubic45/DP/src/training/dataset/out_test_first_look_45/"

counter = 0
rot_err = 0
rot_errors = []

acc22 = 0
acc35 = 0
acc45 = 0

device  = "cuda" if torch.cuda.is_available() else "cpu"
model = PoseRegressor("dinov2_vits14_reg").to(device) # TODO 2x - model name (niekedy large) + import

# TODO
ckpt_path = "/storage/brno2/home/xkubic45/DP/src/training_45deg/main_first_look_regsmallclass/model_geo_first_look_45.pth"
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

pitch_errors = []
yaw_errors = []
roll_errors = []

# ADDED: bin centers for decoding logits -> angles
Cp, Cy, Cr = 180, 360, 360
cp = class_centers_deg(Cp, -90.0).to(device)    # [-89.5 .. +89.5]
cy = class_centers_deg(Cy, -180.0).to(device)   # [-179.5 .. +179.5]
cr = class_centers_deg(Cr,   0.0).to(device)    # [ +0.5 .. +359.5]

# ADDED: temperature used only for decoding expected angle (optional)
TAU_EXPECT = 2.0

for fname in os.listdir(DATASET_PATH):
    if not fname.endswith("_gt.csv"):
        continue

    base = fname.replace("_gt.csv", "")
    panorama_path = os.path.join(DATASET_PATH, f"{base}_panorama.jpg")
    query_path    = os.path.join(DATASET_PATH, f"{base}_query.jpg")
    csv_path      = os.path.join(DATASET_PATH, f"{base}_gt.csv")

    if not (os.path.isfile(query_path) and os.path.isfile(csv_path) and os.path.isfile(panorama_path)):
        print(f"Missing query/csv for {base} – skipping")
        continue

    with open(csv_path, "r") as f:
        row = f.readline().strip().split(",")
    pitch_gt, yaw_gt, roll_gt, _ = map(float, row)

    query_img = Image.open(query_path).convert("RGB")
    query_img = tf(query_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_p, logits_y, logits_r = model(query_img, [panorama_path])

        # TODO logis / TAU_EXPECT
        prob_p = F.softmax(logits_p / TAU_EXPECT, dim=1)     # [1,180]
        prob_y = F.softmax(logits_y / TAU_EXPECT, dim=1)     # [1,360]
        prob_r = F.softmax(logits_r / TAU_EXPECT, dim=1)     # [1,360]
        pred_p_deg_exp = circ_expect_deg(prob_p, cp).cpu()   # [1]
        pred_y_deg_exp = circ_expect_deg(prob_y, cy).cpu()
        pred_r_deg_exp = circ_expect_deg(prob_r, cr).cpu()

    pitch_pred_deg = pred_p_deg_exp[0]
    yaw_pred_deg   = pred_y_deg_exp[0]
    roll_pred_deg  = pred_r_deg_exp[0]

    yaw_gt_t   = torch.tensor(yaw_gt,   dtype=torch.float32)
    pitch_gt_t = torch.tensor(pitch_gt, dtype=torch.float32)
    roll_gt_t  = torch.tensor(roll_gt,  dtype=torch.float32)

    yaw_err   = ang_diff_deg(yaw_pred_deg,   yaw_gt_t,   360.0)
    pitch_err = ang_diff_deg(pitch_pred_deg, pitch_gt_t, 180.0)
    roll_err  = ang_diff_deg(roll_pred_deg,  roll_gt_t,  360.0)

    yaw_errors.append(yaw_err.item())
    pitch_errors.append(pitch_err.item())
    roll_errors.append(roll_err.item())

    err_for_acc = torch.stack([
        ang_diff_deg(yaw_pred_deg,   yaw_gt_t,   360.0),
        ang_diff_deg(pitch_pred_deg, pitch_gt_t, 180.0),
        ang_diff_deg(roll_pred_deg,  roll_gt_t,  360.0),
    ], dim=0)  # [3]

    correct_mask = (err_for_acc <= 22.5).all(dim=0)
    acc22 += int(correct_mask.item())

    correct_mask = (err_for_acc <= 35).all(dim=0)
    acc35 += int(correct_mask.item())

    correct_mask = (err_for_acc <= 45).all(dim=0)
    acc45 += int(correct_mask.item())

    # Geodesic rotation error (use the *same* angles as chosen above for errors)
    # Build R from yaw(Z), pitch(Y), roll(X) in degrees
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
    print(counter, error_deg)

print("Mean rot error (deg):", rot_err / counter)
print("Accuacy all angles to 22.5 deg:", acc22 / counter)
print("Accuacy all angles to 35 deg:",   acc35 / counter)
print("Accuacy all angles to 45 deg:",   acc45 / counter)

# Per-axis error summaries (using USE_EXPECTED choice above)
pitch_errors = torch.tensor(pitch_errors)
avg_pitch_error = pitch_errors.mean().item()
pitch_accuracy_5deg  = (pitch_errors <= 5).float().mean().item()
pitch_accuracy_10deg = (pitch_errors <= 10).float().mean().item()
print(f"Average Pitch Error: {avg_pitch_error:.2f}°")
print(f"Accuracy pitch within 5°: {pitch_accuracy_5deg * 100:.2f}%")
print(f"Accuracy pitch within 10°: {pitch_accuracy_10deg * 100:.2f}%")

yaw_errors = torch.tensor(yaw_errors)
avg_yaw_error = yaw_errors.mean().item()
yaw_accuracy_5deg  = (yaw_errors <= 5).float().mean().item()
yaw_accuracy_10deg = (yaw_errors <= 10).float().mean().item()
print(f"Average yaw Error: {avg_yaw_error:.2f}°")
print(f"Accuracy yaw within 5°: {yaw_accuracy_5deg * 100:.2f}%")
print(f"Accuracy yaw within 10°: {yaw_accuracy_10deg * 100:.2f}%")

roll_errors = torch.tensor(roll_errors)
avg_roll_error = roll_errors.mean().item()
roll_accuracy_5deg  = (roll_errors <= 5).float().mean().item()
roll_accuracy_10deg = (roll_errors <= 10).float().mean().item()
print(f"Average roll Error: {avg_roll_error:.2f}°")
print(f"Accuracy roll within 5°: {roll_accuracy_5deg * 100:.2f}%")
print(f"Accuracy roll within 10°: {roll_accuracy_10deg * 100:.2f}%")

# AUC (same as before)
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
plt.savefig(ckpt_path.replace("model_geo_first_look_45.pth", "") + "orientation_error_first_look.png", dpi=300)
