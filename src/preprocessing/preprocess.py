import os
from panorama import get_panorama_tiles
from query import resize_query_image

INPUT_PATH = "../../dataset/"
OUTPUT_PATH = "../../output/preprocessed"

panorama_paths = [file for file in os.listdir(INPUT_PATH) if file.endswith("_panorama.jpg")]

pairs = []

for panorama in panorama_paths:
    query_image = panorama.replace("_panorama.jpg", "_query.jpeg") # TODO jpeg any type..
    query_image_path = os.path.join(INPUT_PATH, query_image)
    panorama_image_path = os.path.join(INPUT_PATH, panorama)

    if os.path.exists(query_image_path):
        pairs.append((panorama_image_path, query_image_path))
        
for panorama, query in pairs:
    resize_query_image(query, OUTPUT_PATH)
    get_panorama_tiles(panorama, OUTPUT_PATH)
