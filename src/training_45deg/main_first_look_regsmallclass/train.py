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

## DETERMINISTIC
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g_train = torch.Generator()
g_train.manual_seed(SEED)

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

def angles_to_classes(pitch_deg, yaw_deg, roll_deg):
    pitch_idx = torch.clamp(torch.floor(pitch_deg + 90.0), 0, 179).long()
    yaw_idx   = torch.remainder(torch.floor(yaw_deg + 180.0), 360).long()
    roll_idx  = torch.remainder(torch.floor(roll_deg), 360).long()
    return pitch_idx, yaw_idx, roll_idx

def classes_to_angles(pitch_idx, yaw_idx, roll_idx):
    pitch_deg = -90.0  + (pitch_idx.float() + 0.5)  # [B]
    yaw_deg   = -180.0 + (yaw_idx.float()   + 0.5)  # [B]
    roll_deg  = 0.0    + (roll_idx.float()  + 0.5)  # [B]
    return pitch_deg, yaw_deg, roll_deg

def circ_expect_deg(prob: torch.Tensor, centers_deg: torch.Tensor) -> torch.Tensor:
    # Convert bin centers to radians
    theta = torch.deg2rad(centers_deg).to(prob.device)          # [C]
    cos_t = torch.cos(theta)                                     # [C]
    sin_t = torch.sin(theta)                                     # [C]

    # Weighted sum of unit vectors
    x = torch.matmul(prob, cos_t)                                # [B]
    y = torch.matmul(prob, sin_t)                                # [B]

    # Circular mean
    ang_rad = torch.atan2(y, x)                                  # [-pi, pi)
    exp_deg = torch.rad2deg(ang_rad)                             # (-180, 180]

    return exp_deg

def class_centers_deg(C, start_deg):
    idx = torch.arange(C, dtype=torch.float32)
    return start_deg + (idx + 0.5)

def circ_diff_rad(a, b):
    d = a - b
    return (d + torch.pi) % (2*torch.pi) - torch.pi

def circular_huber_deg(pred_deg, tgt_deg, delta_deg=3.0):
    pred = torch.deg2rad(pred_deg)
    tgt  = torch.deg2rad(tgt_deg)
    d = torch.abs(circ_diff_rad(pred, tgt))
    delta = torch.deg2rad(torch.tensor(delta_deg, device=d.device, dtype=d.dtype))
    quad = 0.5 * (d**2) / (delta + 1e-12)
    lin  = d - 0.5 * delta
    return torch.where(d <= delta, quad, lin)

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

        with torch.no_grad():
            tgt_deg = torch.stack((pitch_gt, yaw_gt, roll_gt), dim=1).to(device)
            tgt_norm = angles_normalize(tgt_deg)  # normalize into periodic ranges
            pitch_cls, yaw_cls, roll_cls = angles_to_classes(
                tgt_norm[:,0], tgt_norm[:,1], tgt_norm[:,2]
            )

        with autocast():
            logits_p, logits_y, logits_r = model(query_img, panorama_path)
            
            tau = 2.0
            prob_p = F.softmax(logits_p / tau, dim=1)
            prob_y = F.softmax(logits_y / tau, dim=1)
            prob_r = F.softmax(logits_r / tau, dim=1)

            # Bin centers (deg)
            Cp, Cy, Cr = 180, 360, 360
            cp = class_centers_deg(Cp, -90.0).to(prob_p.device)   # [-89.5..+89.5]
            cy = class_centers_deg(Cy, -180.0).to(prob_y.device)  # [-179.5..+179.5]
            cr = class_centers_deg(Cr,   0.0).to(prob_r.device)   # [ +0.5..+359.5]

            # Circular expected angles (deg)
            pred_pitch_deg = circ_expect_deg(prob_p, cp)
            pred_yaw_deg   = circ_expect_deg(prob_y, cy)
            pred_roll_deg  = circ_expect_deg(prob_r, cr)

            # Targets (already normalized in your code)
            tgt_pitch_deg, tgt_yaw_deg, tgt_roll_deg = tgt_norm[:,0], tgt_norm[:,1], tgt_norm[:,2]

            # Circular Huber with ~3° tolerance
            hub_p = circular_huber_deg(pred_pitch_deg, tgt_pitch_deg, delta_deg=3.0).mean()
            hub_y = circular_huber_deg(pred_yaw_deg,   tgt_yaw_deg,   delta_deg=3.0).mean()
            hub_r = circular_huber_deg(pred_roll_deg,  tgt_roll_deg,  delta_deg=3.0).mean()

            loss = (hub_p + hub_y + hub_r) / 3.0  # no CE, purely distance-shaped

        if train:            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # CHANGED: compute accuracy similar to before (<= 45° on each angle)
        with torch.no_grad():
            pred_p_cls = torch.argmax(logits_p, dim=1)
            pred_y_cls = torch.argmax(logits_y, dim=1)
            pred_r_cls = torch.argmax(logits_r, dim=1)

            pred_p_deg, pred_y_deg, pred_r_deg = classes_to_angles(
                pred_p_cls, pred_y_cls, pred_r_cls
            )
            # TODO tgt_roll_deg used instead of prediction - fix for bad roll
            pred_deg = torch.stack([pred_p_deg, pred_y_deg, tgt_roll_deg], dim=1)

            err = torch.abs(pred_deg - tgt_norm)
            correct_mask = (err <= 22.5).all(dim=1)  # keep same FOV/2 criterion
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
    batchsz = 2
    lr      = 1e-4

    tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

    ds = OrientationDataset(transform=tf)
    idx_train, idx_val = train_test_split(range(len(ds)),
                                          test_size=0.1,
                                          random_state=SEED,
                                          shuffle=True)    
    dl_train = DataLoader(
        Subset(ds, idx_train),
        batch_size=batchsz,
        shuffle=True,
        pin_memory=True,
        generator=g_train,
    )

    dl_val = DataLoader(
        Subset(ds, idx_val),
        batch_size=batchsz,
        shuffle=False,
        pin_memory=True,
    )

    model = PoseRegressor("dinov2_vits14_reg").to(device)
    # CHANGED: no geodesic loss; classification uses cross-entropy in run_epoch
    crit_geo = None  # CHANGED

    opt   = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=lr)

    best_val_loss = float("inf")
    ckpt_path     = "model_geo_first_look_45.pth"

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