import os
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from py360convert.utils import xyzpers, xyz2uv, uv2coor
from main_first_look.model import PoseRegressor
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

def angles_normalize(pred):
    out = []

    for i in range(pred.size(0)):
        pitch, yaw, roll = pred[i].tolist()

        pitch = ((pitch +  90) % 180) -  90  
        yaw = ((yaw + 180) % 360) - 180
        roll = roll % 360

        row_t = torch.tensor(
            [pitch, yaw, roll],
            dtype=pred.dtype,
            device=pred.device,
        )

        out.append(row_t)
        
    return torch.stack(out, dim=0)

DATASET_PATH = "/storage/brno2/home/xkubic45/DP/src/training/dataset/out_test_first_look_45/"

counter = 0
rot_err = 0
rot_errors = []

acc22 = 0
acc35 = 0
acc45 = 0

device  = "cuda" if torch.cuda.is_available() else "cpu"
model = PoseRegressor("dinov2_vitb14").to(device)
ckpt_path = "/storage/brno2/home/xkubic45/DP/src/training_45deg/main_first_look/model_geo_first_look_45.pth"

checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

pitch_errors = []
yaw_errors = []
roll_errors = []

for fname in os.listdir(DATASET_PATH):
    if not fname.endswith("_gt.csv"):
        continue

    base = fname.replace("_gt.csv", "")
    panorama_path = os.path.join(DATASET_PATH, f"{base}_panorama.jpg")
    query_path  = os.path.join(DATASET_PATH, f"{base}_query.jpg")
    csv_path    = os.path.join(DATASET_PATH, f"{base}_gt.csv")

    if not (os.path.isfile(query_path) and os.path.isfile(csv_path) and os.path.isfile(panorama_path)):
        print(f"Missing query/csv for {base} – skipping")
        continue

    with open(csv_path, "r") as f:
        row = f.readline().strip().split(",")
    pitch_gt, yaw_gt, roll_gt, _= map(float, row)
    
    query_img  = Image.open(query_path).convert("RGB")
    # cutout_img = Image.open(cutout_path).convert("RGB")

    query_img  = tf(query_img).unsqueeze(0).to(device)
    # cutout_img = tf(cutout_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(query_img, [panorama_path])
    
    pred_eq_nd = angles_normalize(pred.cpu())
    
    pitch = pred_eq_nd[0, 0]
    yaw   = pred_eq_nd[0, 1]
    roll  = pred_eq_nd[0, 2]
    
    pred = torch.stack([pitch, yaw, roll], dim=0)
    target = torch.stack([torch.tensor(pitch_gt, device="cpu", dtype=torch.float32), torch.tensor(yaw_gt, device="cpu", dtype=torch.float32), torch.tensor(roll_gt, device="cpu", dtype=torch.float32)], dim=0)
    
    def ang_diff_deg(a, b, period):
        d = (a - b + period/2) % period - period/2
        return torch.abs(d)

    yaw_err   = ang_diff_deg(yaw,   yaw_gt,   360.0)
    pitch_err = ang_diff_deg(pitch, pitch_gt, 180.0)
    roll_err  = ang_diff_deg(roll,  roll_gt,  360.0)

    yaw_errors.append(yaw_err.item())
    pitch_errors.append(pitch_err.item())
    roll_errors.append(roll_err.item())

    err = torch.stack([yaw_err, pitch_err, roll_err], dim=0)
    
    correct_mask = (err <= 22.5).all(dim=0) 
    acc22 += correct_mask.sum().item()
    
    correct_mask = (err <= 35).all(dim=0) 
    acc35 += correct_mask.sum().item()
    
    correct_mask = (err <= 45).all(dim=0) 
    acc45 += correct_mask.sum().item()
    
    
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
    print(counter, error_deg)
    
print("Mean rot error (deg):", rot_err / counter)
print("Accuacy all angles to 22.5 deg:", acc22 / counter)
print("Accuacy all angles to 35 deg:", acc35 / counter)
print("Accuacy all angles to 45 deg:", acc45 / counter)

pitch_errors = torch.tensor(pitch_errors)
avg_pitch_error = pitch_errors.mean().item()
pitch_accuracy_5deg = (pitch_errors <= 5).float().mean().item()
pitch_accuracy_10deg = (pitch_errors <= 10).float().mean().item()

print(f"Average Pitch Error: {avg_pitch_error:.2f}°")
print(f"Accuracy pitch within 5°: {pitch_accuracy_5deg * 100:.2f}%")
print(f"Accuracy pitch within 10°: {pitch_accuracy_10deg * 100:.2f}%")

yaw_errors = torch.tensor(yaw_errors)
avg_yaw_error = yaw_errors.mean().item()
yaw_accuracy_5deg = (yaw_errors <= 5).float().mean().item()
yaw_accuracy_10deg = (yaw_errors <= 10).float().mean().item()

print(f"Average yaw Error: {avg_yaw_error:.2f}°")
print(f"Accuracy yaw within 5°: {yaw_accuracy_5deg * 100:.2f}%")
print(f"Accuracy yaw within 10°: {yaw_accuracy_10deg * 100:.2f}%")

roll_errors = torch.tensor(roll_errors)
avg_roll_error = roll_errors.mean().item()
roll_accuracy_5deg = (roll_errors <= 5).float().mean().item()
roll_accuracy_10deg = (roll_errors <= 10).float().mean().item()

print(f"Average roll Error: {avg_roll_error:.2f}°")
print(f"Accuracy roll within 5°: {roll_accuracy_5deg * 100:.2f}%")
print(f"Accuracy roll within 10°: {roll_accuracy_10deg * 100:.2f}%")

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
plt.savefig("orientation_error_first_look.png", dpi=300)

