import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from model import PoseRegressor
from dataset import OrientationDataset
import torchvision.transforms as T
from py360convert.utils import xyzpers, xyz2uv, uv2coor
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
    
def euler_to_quat(e):
    y, p, = e[:,0], e[:,1]
    r = torch.zeros_like(y)
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

def run_epoch(model, loader, crit_geo, optimizer=None, device="cpu", scaler=None):
    train = optimizer is not None
    model.train(train)

    total_loss, total_acc, n = 0.0, 0.0, 0

    pbar = tqdm(loader, desc="Training" if train else "Validation", unit="batch")

    for batch_idx, (query_img, panorama_path, pitch_gt, yaw_gt, roll_gt) in enumerate(pbar):

        query_img  = query_img.to(device, non_blocking=True)
        
        pitch_gt   = pitch_gt.float()
        yaw_gt     = yaw_gt.float()
        roll_gt    = roll_gt.float()

        tgt = torch.stack((pitch_gt, yaw_gt), dim=1).to(device, non_blocking=True)

        with autocast():
            pred = model(query_img, panorama_path)
            
            with torch.no_grad():   
                pred_eq_nd = angles_normalize(pred.cpu()).to(device)
                
            pred_eq = pred + (pred_eq_nd - pred).detach()

            loss = crit_geo(pred_eq, tgt)

        if train:            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            err = torch.abs(pred_eq - tgt)
            correct_mask = (err <= 45).all(dim=1) # FOV / 2 -> FOV for both is 45 deg
            acc = correct_mask.sum().item()

        bs = query_img.size(0)
        total_loss += loss.item() * bs
        total_acc  += acc
        n += bs
        
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "acc": f"{acc/bs:.3f}",
            "seen": f"{n}/{len(loader.dataset)}"
        })

    return total_loss / n, total_acc / n

scaler = GradScaler()

def main():
    log_file = open("output_first_look_45.txt", "a") 
    log_file.write("Training started." + "\n")
    log_file.flush()
    
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    epochs  = 100
    batchsz = 5
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
    crit_geo = GeodesicLoss()
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.SGD(
        trainable_params,
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )

    best_val_loss = float("inf")
    ckpt_path     = "model_geo_sgd_first_look45.pth"

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, dl_train, crit_geo, opt, device, scaler)
        va_loss, va_acc = run_epoch(model, dl_val,   crit_geo, None, device)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({
                "model_state": model.state_dict(),
                "optim_state": opt.state_dict(),
            }, ckpt_path)

        msg = f"Epoch {ep} | "f"train loss {tr_loss:.3f}  acc {tr_acc*100:.3f}% || "f"val loss {va_loss:.3f}  acc {va_acc*100:.3f}%"
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        
if __name__ == "__main__":
    main()

# Epoch 1 | train loss 0.573  acc 21.975% || val loss 0.471  acc 27.599%.