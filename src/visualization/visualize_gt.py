# fov: (degrees)
# pitch: [-90;+90], 0 -> middle, -90 -> down, +90 -> up (degrees)
# yaw: [-180;+180] - 0 -> center, -180 -> left, +180 -> right (degrees)
# roll: [-180;+180] - 0 -> aligned with horizon, + -> clockwise, - conter clockwise (degrees)

import cv2 
import py360convert_lib.Perspec2Equirec as P2E
import numpy as np
from PIL import Image

IN_PATH_PANO = "../example/panorama.jpg"
IN_PATH_QUERY = "../example/query.jpg"
IN_PATH_GT = "../example/gt.csv"

with open(IN_PATH_GT, "r") as f:
    row = f.readline().strip().split(",")
pitch, yaw, roll, fov = map(float, row)

pano = Image.open(IN_PATH_PANO).convert("RGB")
W, H = pano.size

equ = P2E.Perspective(IN_PATH_QUERY, FOV=fov, THETA=yaw, PHI=pitch, ROLL=roll)
img, mask = equ.GetEquirec(height=H, width=W)

pano_np = np.array(pano)
pano_np = cv2.cvtColor(pano_np, cv2.COLOR_RGB2BGR)

proj_img = cv2.resize(img, (W, H))

mask = np.any(proj_img > 10, axis=2)
mask3 = mask[:, :, None]

alpha = 0.5
overlay = pano_np.copy()
overlay[mask] = (
    pano_np[mask] * (1 - alpha) +
    proj_img[mask] * alpha
)

overlay = overlay.astype(np.uint8)
cv2.imshow("", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
    