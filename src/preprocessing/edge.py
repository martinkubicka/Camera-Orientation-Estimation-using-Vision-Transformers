import cv2
import numpy as np

def has_horizon(np_img, num_of_edges = 3) -> bool:
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(np_img, 50, 150)

    num_labels, _ = cv2.connectedComponents(edges)
    edge_count = num_labels - 1
    
    return edge_count >= num_of_edges, edges

# from PIL import Image
# img = Image.open("../../dataset/2_query.jpeg")
# img = np.asarray(img)
# has_hor, edges = has_horizon(img)
# tile = Image.fromarray(edges)
# tile.save(f"../../dataset/2_query_edges.jpg")
