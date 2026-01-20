import time
import torch
from main_first_look.model import PoseRegressor
from PIL import Image
import torchvision.transforms as T

def angles_normalize(pred):
    out = []
    for i in range(pred.size(0)):
        pitch, yaw, roll = pred[i].tolist()
        pitch = ((pitch +  90) % 180) -  90
        yaw   = ((yaw   + 180) % 360) - 180
        roll  = roll % 360
        row_t = torch.tensor([pitch, yaw, roll], dtype=pred.dtype, device=pred.device)
        out.append(row_t)
    return torch.stack(out, dim=0)

tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PoseRegressor("dinov2_vitb14").to(device)
model.eval()

dummy_paths = ["/storage/brno2/home/xkubic45/DP/src/training/dataset/out_test_first_look_45/flickr_5845318147_13f4e26e89_2446_81035653@N00_panorama.jpg"]
dummy_query = Image.open("/storage/brno2/home/xkubic45/DP/src/training/dataset/out_test_first_look_45/flickr_5845318147_13f4e26e89_2446_81035653@N00_query.jpg").convert("RGB")
dummy_query = tf(dummy_query).unsqueeze(0).to(device)

for _ in range(50):
    with torch.no_grad():
        pred = model(dummy_query, dummy_paths)
        pred_eq_nd = angles_normalize(pred.cpu())

torch.cuda.synchronize() if device == "cuda" else None
start = time.time()

pred = model(dummy_query, dummy_paths)
pred_eq_nd = angles_normalize(pred.cpu())

torch.cuda.synchronize() if device == "cuda" else None
end = time.time()

avg_time_per_frame = (end - start)
print(f"Average inference time: {avg_time_per_frame:.6f} seconds per frame")
