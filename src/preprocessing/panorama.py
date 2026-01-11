import numpy as np
from PIL import Image
import py360convert
import os
from edge import has_horizon

# HEIGHT = 512 # TODO swapped
# WIDTH = 256 # TODO swapped
# FOV_X = 45
# FOV_Y = 90

HEIGHT = 518 # TODO swapped
WIDTH = 518 # TODO swapped
FOV_X = 45
FOV_Y = 45

INPUT_HEIGHT = 2048
INPUT_WIDTH = 4096

# 183 136 - 2_panorama (x,y)
# 332.5 116.0 - 1_panorama (x,y) - p,y 8.4375 53.7890625

def get_tile(img, params):
    return py360convert.e2p(img, (FOV_X, FOV_Y), params['yaw'], params['pitch'], (HEIGHT, WIDTH))

def get_panorama_tiles(img_path, output_folder):
    img = Image.open(img_path)
    img = np.asarray(img)
    
    image_name = img_path.split("/")[-1].split(".")[0].split("_")[0]
    os.makedirs(f"{output_folder}/{image_name}", exist_ok=True)
    
    y = 0
    for pitch in np.arange(-90, 90, 22.5):
        x = 0
        for yaw in np.arange(-180, 180, 45):
            print(pitch + (22.5/2), yaw+ 45/2, f"{image_name}/{image_name}_{x}_{y}")
            params = {
                'roll': 0.,
                'pitch': pitch + (22.5 / 2),
                'yaw': yaw + (45 / 2)
            }
            tile = get_tile(img, params)
            
            has_hor, edges = has_horizon(tile)
            
            x += 1
            
            if not has_hor:
                continue
            
            tile = Image.fromarray(edges) # todo edges instead of tile
            tile.save(f"{output_folder}/{image_name}/{image_name}_{x-1}_{y}.jpg")

        y += 1

#######

def xy_to_yaw_pitch(x, y, input_width, input_height):
    yaw = (x / input_width) * 360.0 - 180.0
    pitch = 90.0 - (y / input_height) * 180.0
    return yaw, pitch

# TODO change me based on median and resized panorama resolution
x, y = 332.5 * (4096/512) , 116 * (2048/256) # coordinates on whole panorama but it is resized by mast3r to thats why it is too small number

img = Image.open("../../dataset/1_panorama.jpg")
img = np.asarray(img)

yaw, pitch = xy_to_yaw_pitch(x, y, INPUT_WIDTH, INPUT_HEIGHT)
print(pitch, yaw)
params = {
    'roll': 0.,
    'pitch': pitch,
    'yaw': yaw
}
tile = get_tile(img, params)
tile = Image.fromarray(tile)
tile.save(f"../../dataset/test45.jpg")
