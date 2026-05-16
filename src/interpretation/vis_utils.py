import numpy as np
import matplotlib.pyplot as plt
import cv2

def normalize_global(x, vmin, vmax):
    return (x - vmin) / (vmax - vmin + 1e-8)

def visualize(query_img, orig_tiles, attn_qp, attn_pq, Np, Nq, rows, cols):
    tile_size = int(np.sqrt(Np))
    q_size = int(np.sqrt(Nq))

    # query to pano
    tile_h, tile_w, _ = orig_tiles[0].shape

    pano_h = rows * tile_h
    pano_w = cols * tile_w

    pano_img = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
    pano_attn = np.zeros((pano_h, pano_w), dtype=np.float32)

    all_vals = []

    for idx in range(rows * cols):
        start = idx * Np
        end = start + Np

        patch = attn_qp[0, :, start:end].mean(0)
        patch = patch.view(tile_size, tile_size).cpu().numpy()

        all_vals.append(patch)

    all_vals = np.concatenate([p.flatten() for p in all_vals])
    vmin, vmax = all_vals.min(), all_vals.max()

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c

            img = orig_tiles[idx] / 255.0

            start = idx * Np
            end = start + Np

            patch = attn_qp[0, :, start:end].mean(0)
            patch = patch.view(tile_size, tile_size).cpu().numpy()

            patch = normalize_global(patch, vmin, vmax)

            patch = cv2.resize(patch, (tile_w, tile_h))

            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w

            pano_img[y0:y1, x0:x1] = img
            pano_attn[y0:y1, x0:x1] = patch

            cv2.rectangle(
                pano_img,
                (x0, y0),
                (x1 - 1, y1 - 1),
                color=(1, 1, 1),
                thickness=1
            )

    plt.figure(figsize=(12, 6))
    plt.title("Query to Panorama")
    plt.imshow(pano_img)
    plt.imshow(pano_attn  ** 0.4 , cmap="jet", alpha=0.5)
    plt.axis("off")

    # pano to query
    q_img = query_img / 255.0
    full_attn = np.zeros((q_size, q_size), dtype=np.float32)

    for idx in range(rows * cols):
        start = idx * Np
        end = start + Np

        patch = attn_pq[0, start:end, :].mean(0)
        patch = patch.view(q_size, q_size).cpu().numpy()

        full_attn += patch

    full_attn = (full_attn - full_attn.min()) / (full_attn.max() - full_attn.min() + 1e-8)
    full_attn = cv2.resize(full_attn, (q_img.shape[1], q_img.shape[0]))

    plt.figure(figsize=(6, 6))
    plt.title("Panorama to Query")
    plt.imshow(q_img)
    plt.imshow(full_attn, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.show()
