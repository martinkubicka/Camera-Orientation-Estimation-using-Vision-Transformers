from PIL import Image
import os

LONGER_EDGE_SIZE = 512

# inspired by https://github.com/naver/dust3r/blob/c9e9336a6ba7c1f1873f9295852cea6dffaf770d/dust3r/utils/image.py
# _resize_pil_image function
def resize_query_image(img_path, output_folder):
    img = Image.open(img_path).convert('RGB')
    S = max(img.size)
    
    if S > LONGER_EDGE_SIZE:
        interp = Image.LANCZOS
    elif S <= LONGER_EDGE_SIZE:
        interp = Image.BICUBIC
    
    new_size = tuple(int(round(x * LONGER_EDGE_SIZE / S)) for x in img.size)
    resized_img = img.resize(new_size, interp)
    
    image_name = img_path.split("/")[-1].split(".")[0].split("_")[0]
    os.makedirs(f"{output_folder}/{image_name}", exist_ok=True)
    resized_img.save(f"{output_folder}/{image_name}/{image_name}_query.jpg")
