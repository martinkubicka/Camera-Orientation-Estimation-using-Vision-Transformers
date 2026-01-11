# from PIL import Image
# import py360convert
# import os

# # grid params
# H_TILES = 4
# W_TILES = 8
# fov = (45.0, 45.0)          # (hFOV, vFOV) in degrees
# out_hw = (518, 518)         # output size (H, W)
# img_path = "./data/3/pano.jpg"

# j = np.arange(H_TILES, dtype=np.float32)
# pitch_vals = 90.0 - (j + 0.5) * (180.0 / H_TILES)
# i = np.arange(W_TILES, dtype=np.float32)
# yaw_vals = -180.0 + (i + 0.5) * (360.0 / W_TILES)
# img_np = np.array(Image.open(img_path).convert("RGB"))

# counter = 0
# for r, pv in enumerate(pitch_vals):
#     for c, yv in enumerate(yaw_vals):
#         counter += 1
#         tile = py360convert.e2p(img_np, fov, float(yv), float(pv), out_hw)        
#         Image.fromarray(tile).save(f"./tile_{str(counter)}.jpg")