import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from model import PoseRegressor
from dataset import OrientationDataset
import torchvision.transforms as T
from py360convert.utils import xyzpers, xyz2uv, uv2coor

def angular_diff(pred, target):
    diff360 = torch.remainder(pred - target + 180, 360) - 180 
    diff360 = diff360.abs()

    err = diff360.clone()

    pitch_err = torch.minimum(diff360[:,1], (180 - diff360[:,1]).abs())
    err[:,1] = pitch_err
    return err

class AngularHuberLoss(nn.Module):
    def __init__(self, delta=1.0, reduction="mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.loss = nn.SmoothL1Loss(reduction=reduction, beta=delta)

    def forward(self, pred, target):
        err = angular_diff(pred, target)
        
        # normalize
        div = torch.tensor([90.0, 180.0, 360.0], device=err.device).view(1, 3)  # shape [1, 3]
        err = err / div
        
        loss = self.loss(err, torch.zeros_like(err))
        
        return loss
    
# def canonicalize_pitch(p: torch.Tensor) -> torch.Tensor:
#     return ((p + 90) % 180) - 90

# def canonicalize_yaw(y: torch.Tensor) -> torch.Tensor:
#     return ((y + 180) % 360) - 180

# def canonicalize_roll(r: torch.Tensor) -> torch.Tensor:
#     return r % 360

def euler_to_quat(e):
    y, p, r = e[:,0], e[:,1], e[:,2]
    cy, sy = torch.cos(y/2), torch.sin(y/2)
    cp, sp = torch.cos(p/2), torch.sin(p/2)
    cr, sr = torch.cos(r/2), torch.sin(r/2)
    q0 = cy*cp*cr + sy*sp*sr
    q1 = cy*cp*sr - sy*sp*cr
    q2 = sy*cp*sr + cy*sp*cr
    q3 = sy*cp*cr - cy*sp*sr
    q = torch.stack([q0,q1,q2,q3], dim=-1)
    return q / q.norm(dim=-1, keepdim=True)

class GeodesicLoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_angles, gt_angles):          
        pred = torch.deg2rad(pred_angles)
        gt   = torch.deg2rad(gt_angles)
        q_pred = euler_to_quat(pred)
        q_gt   = euler_to_quat(gt)
        dot = torch.abs((q_pred * q_gt).sum(dim=-1))
        dot = torch.clamp(dot, -1 + self.eps, 1 - self.eps)
        loss = torch.acos(dot)
        
        return loss.mean()

def accuracy10(pred, target):
    err = angular_diff(pred, target)
    correct = (err <= 10).all(dim=1).float()
    return correct.mean()

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
        px, py, roll = pred[i].tolist()

        w_c, h_c   = int(cutout_w[i].item()),  int(cutout_h[i].item())
        w_q, h_q   = query_w[i].item(),   query_h[i].item()
        fovx, fovy = cutout_fov_x[i].item(), cutout_fov_y[i].item()
        yaw_c, pitch_c = yaw_cutout[i].item(), pitch_cutout[i].item()

        # pitch = ((pitch +  90) % 180) -  90  
        # yaw = ((yaw + 180) % 360) - 180
        px = float(px) % w_c
        py = float(py) % h_c
        roll = roll % 360 

        # px, py      = pitch_yaw_to_pixel(pitch, yaw, w_c, h_c)
        
        px = max(0, min(px, w_c - 1))
        py = max(0, min(py, h_c - 1))
        
        px = int(round(px))
        py = int(round(py))
        
        u_eq, v_eq  = get_eq(px, py, w_c, h_c, fovx, fovy, yaw_c, pitch_c)
        pitch_eq, yaw_eq = pixel_to_pitch_yaw(u_eq, v_eq, pano_W, pano_H)

        row_t = torch.tensor(
            [pitch_eq[0], yaw_eq[0], roll],
            dtype=pred.dtype,
            device=pred.device,
        )

        out.append(row_t)
        
    return torch.stack(out, dim=0)

def run_epoch(model, loader, crit_angular, crit_geo, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)

    total_loss, total_acc, n = 0.0, 0.0, 0

    for (query_img, cutout_img,
         pitch_gt, yaw_gt, roll_gt,
         yaw_cutout, pitch_cutout,
         cutout_w, cutout_h,
         query_h, query_w,
         cutout_fov_x, cutout_fov_y) in loader:

        query_img  = query_img.to(device, non_blocking=True)
        cutout_img = cutout_img.to(device, non_blocking=True)
        
        pitch_gt   = pitch_gt.float()
        yaw_gt     = yaw_gt.float()
        roll_gt    = roll_gt.float()
        yaw_cutout = yaw_cutout.float()
        pitch_cutout = pitch_cutout.float()
        cutout_fov_x = cutout_fov_x.float()
        cutout_fov_y = cutout_fov_y.float()

        tgt = torch.stack((pitch_gt, yaw_gt, roll_gt), dim=1).to(device, non_blocking=True)

        pred = model(query_img, cutout_img)
        
        with torch.no_grad():   
            pred_eq_nd = angles_to_eq_batch(
                pred.cpu(),
                yaw_cutout, pitch_cutout,
                cutout_w,  cutout_h,
                query_w,   query_h,
                cutout_fov_x, cutout_fov_y,
            ).to(device)
            
        pred_eq = pred + (pred_eq_nd - pred).detach()
        
        # loss_ang = crit_angular(pred_eq, tgt)
        loss = crit_geo(pred_eq, tgt)
        # loss = loss_ang + 0.01 * loss_geo

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            acc = accuracy10(pred, tgt)

        bs = query_img.size(0)
        total_loss += loss.item() * bs
        total_acc  += acc.item()  * bs
        n += bs

    return total_loss / n, total_acc / n

def main():
    log_file = open("output_second_look_45.txt", "a") 
    log_file.write("Training started." + "\n")
    log_file.flush()
    
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    epochs  = 100
    batchsz = 64
    lr      = 1e-4

    tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

    ds = OrientationDataset(transform=tf)
    idx_train, idx_val = train_test_split(range(len(ds)),
                                          test_size=0.2,
                                          random_state=42,
                                          shuffle=True)
    dl_train = DataLoader(Subset(ds, idx_train), batch_size=batchsz,
                          shuffle=True,  num_workers=2, pin_memory=True)
    dl_val   = DataLoader(Subset(ds, idx_val),   batch_size=batchsz,
                          shuffle=False, num_workers=2, pin_memory=True)

    model = PoseRegressor("dinov2_vitl14").to(device)
    crit_angular  = AngularHuberLoss()
    crit_geo = GeodesicLoss()
    opt   = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=lr)

    best_val_loss = float("inf")
    ckpt_path     = "model_geo_second_look_45.pth"

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, dl_train, crit_angular, crit_geo, opt, device)
        va_loss, va_acc = run_epoch(model, dl_val,   crit_angular, crit_geo, None, device)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({
                "model_state": model.state_dict(),
                "optim_state": opt.state_dict(),
            }, ckpt_path)

        msg = f"Epoch {ep} | "f"train loss {tr_loss:.3f}  acc {tr_acc*100:.3f}% | "f"val loss {va_loss:.3f}  acc {va_acc*100:.3f}%"
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
            
if __name__ == "__main__":
    main()
