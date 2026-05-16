import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
from scipy.spatial.transform import Rotation as R
from model import OrientationEstimator
from vis_utils import visualize
import torch.nn.functional as F

CHECKPOINT = "../checkpoints/best_model.pth"
QUERY_IMG = "../example/query.jpg"
PANO_IMG = "../example/panorama.jpg"
GT_PATH = "../example/gt.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

def circ_expect_deg(prob, centers_deg):
    theta = torch.deg2rad(centers_deg).to(prob.device)
    x = torch.matmul(prob, torch.cos(theta))
    y = torch.matmul(prob, torch.sin(theta))
    ang_rad = torch.atan2(y, x)
    
    return torch.rad2deg(ang_rad)

def class_centers_deg(C, start_deg):
    idx = torch.arange(C, dtype=torch.float32)
    return start_deg + (idx + 0.5)

with open(GT_PATH, "r") as f:
    row = f.readline().strip().split(",")
pitch_gt, yaw_gt, roll_gt, _ = map(float, row)

model = OrientationEstimator(transform, device).to(device)

ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

query_pil = Image.open(QUERY_IMG).convert("RGB")
query_orig = np.array(query_pil)

query = transform(query_pil).unsqueeze(0).to(device)

with torch.no_grad():
    attn_qp, attn_pq, pano_tiles, orig_tiles, Np, Nq, logits_p, logits_y, logits_r = model(
        query,
        [PANO_IMG],
        return_attention=True
    )

visualize(
    query_orig,
    orig_tiles,
    attn_qp,
    attn_pq,
    Np,
    Nq,
    model.num_rows,
    model.num_cols
)

Cp, Cy, Cr = 180, 360, 360
cp = class_centers_deg(Cp, -90.0).to(device)
cy = class_centers_deg(Cy, -180.0).to(device)
cr = class_centers_deg(Cr, -180.0).to(device)
prob_p = F.softmax(logits_p, dim=1)
prob_y = F.softmax(logits_y, dim=1)
prob_r = F.softmax(logits_r, dim=1)
pred_p_deg_exp = circ_expect_deg(prob_p, cp).cpu()
pred_y_deg_exp = circ_expect_deg(prob_y, cy).cpu()
pred_r_deg_exp = circ_expect_deg(prob_r, cr).cpu()

pitch_pred_deg = pred_p_deg_exp[0]
yaw_pred_deg   = pred_y_deg_exp[0]
roll_pred_deg  = pred_r_deg_exp[0]

R_gt  = R.from_euler('zyx', [yaw_gt,  pitch_gt,  roll_gt ], degrees=True).as_matrix()
R_est = R.from_euler('zyx', [float(yaw_pred_deg), float(pitch_pred_deg), float(roll_pred_deg)], degrees=True).as_matrix()

R_diff = R_gt.T @ R_est
cos_angle = (np.trace(R_diff) - 1) / 2
cos_angle = np.clip(cos_angle, -1.0, 1.0)

error_rad = np.arccos(cos_angle)
error_deg = np.degrees(error_rad)

print("Error:", error_deg)
