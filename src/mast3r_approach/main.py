import sys
import os
# we can use main out of mast3r folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'mast3r')))
import numpy as np
import cv2

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

from postprocessing.coord_p2e import persp_to_equirect_deg

def gms_filter(src, dst, grid=20, multiscale=True):
    def single_scale(s, d, g):
        N = s.shape[0]
        # normalizácia <0,1>
        s_norm = s / (s.max(axis=0) + 1e-7)
        d_norm = d / (d.max(axis=0) + 1e-7)

        # index bunky
        idx_s = (np.floor(s_norm * g)).astype(int)
        idx_d = (np.floor(d_norm * g)).astype(int)
        flat = lambda x: x[:, 1] * g + x[:, 0]
        flat_s, flat_d = flat(idx_s), flat(idx_d)

        # hustoty
        hist_s, hist_d = {}, {}
        for i in flat_s: hist_s[i] = hist_s.get(i, 0) + 1
        for i in flat_d: hist_d[i] = hist_d.get(i, 0) + 1

        # 3×3 okolie
        offs = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]
        def neigh(i):
            x, y = i % g, i // g
            return [ (y+dy)*g + (x+dx)
                     for dx,dy in offs
                     if 0 <= x+dx < g and 0 <= y+dy < g ]

        support = np.array([min(sum(hist_s.get(j,0) for j in neigh(flat_s[k])),
                                sum(hist_d.get(j,0) for j in neigh(flat_d[k])))
                            for k in range(N)])
        return support >= np.median(support)   # mask

    # multi-scale: 20×20, 10×10, 5×5
    mask = single_scale(src, dst, grid)
    if multiscale:
        for g in (grid//2, grid//4):
            mask |= single_scale(src, dst, g)
    return mask

if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "../models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    
    
    INPUT_PATH = "../output/preprocessed/2"

    panorama_paths = [file for file in os.listdir(INPUT_PATH) if file.startswith("2_") and not "_query" in file]

    pairs = []

    for panorama in panorama_paths:
        panorama_image_path = os.path.join(INPUT_PATH, panorama)
        query_image_path = INPUT_PATH + "/2_query.jpg"

        if os.path.exists(query_image_path):
            pairs.append((panorama_image_path, query_image_path))
            
    # print("len pairs", len(pairs))
    # print(pairs)
    
    all_conf_pairs = []
            
    for panorama, query in pairs:
        images = load_images([panorama, query], size=512)
        
        output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                    device=device, dist='dot', block_size=2**13)

        # ignore small border around the edge
        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

        # print(matches_im0, len(matches_im0))
        # print("-----")
        # print(matches_im1)
        
        
        ##### conf
        
        

        conf_im0 = pred1['conf'].squeeze(0).detach().cpu().numpy() #confidence 
        conf_im1 = pred2['conf'].squeeze(0).detach().cpu().numpy()
        
        # Extract confidence scores for the matches
        match_conf_im0 = conf_im0[matches_im0[:, 1], matches_im0[:, 0]]
        match_conf_im1 = conf_im1[matches_im1[:, 1], matches_im1[:, 0]]
        
        #print(match_conf_im0)
        
        # filtered_conf_im0 = match_conf_im0[match_conf_im0 >= 1.0001]
        
        # print(filtered_conf_im0)
        # print("*****")
    
        # if len(filtered_conf_im0) > 0:
        all_conf_pairs.append((match_conf_im0, match_conf_im1, matches_im0, matches_im1, str(panorama.split("/")[-1]), view1, view2))
    
    all_conf_pairs = sorted(all_conf_pairs, key=lambda x: len(x[0]), reverse=True)

    #print(len(all_conf_pairs))

    top_3_pairs = all_conf_pairs[:3]

    matches_equir = []
    persp_points = []
    # print("Top 3 pairs based on array length (after filtering):")
    for i, (im0, im1, matches_im0, matches_im1, name, view1, view2) in enumerate(top_3_pairs):
        # print(f"Pair {i+1} - Length: {len(im0)}")
        # print("Image 1 coefs:", im0)
        # print("Image 2 coefs:", im1)
        # print(matches_im0, matches_im1)
            
        # print("CONF:", match_conf_im0, match_conf_im1, len(match_conf_im0))        
        # #########
        
        # visualize a few matches
        import numpy as np
        import torch
        import torchvision.transforms.functional
        from matplotlib import pyplot as plt

        image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

        viz_imgs = []
        for i, view in enumerate([view1, view2]):
            rgb_tensor = view['img'] * image_std + image_mean
            viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
        

        # removing non white (edge) pointings
        white_mask = np.all(viz_imgs[0] >= [0.9, 0.9, 0.9], axis=-1).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        white_mask_dilated = cv2.dilate(white_mask, kernel, iterations=1)

        x_coords = matches_im0[:, 0].astype(int)
        y_coords = matches_im0[:, 1].astype(int)

        keep_mask = white_mask_dilated[y_coords, x_coords].astype(bool)

        filtered_matches_im0 = matches_im0[keep_mask]
        filtered_matches_im1 = matches_im1[keep_mask]
        
        matches_im0 = filtered_matches_im0
        matches_im1 = filtered_matches_im1
        ### end of removing
        
        # GMS
        src_tile = matches_im0.astype(np.float32)
        dst_tile = matches_im1.astype(np.float32)

        mask_gms = gms_filter(src_tile, dst_tile, grid=20, multiscale=True)

        matches_im0 = matches_im0[mask_gms]
        matches_im1 = matches_im1[mask_gms]
        # end of gms
        
        num_matches = matches_im0.shape[0]
        n_viz = num_matches
        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
        img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        plt.figure()
        plt.imshow(img)
        cmap = plt.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        
        save_path = "../outputs_third_testing/" + name
        #print(save_path)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # presp coordinates to equirectangular
        yy = -180 + (int(name.split("_")[1])) * 45 + 45/2 # X
        pp = -90 + (7 - int(name.split("_")[2].replace(".jpg", ""))) * 22.5 + 22.5/2# Y - 8- because panorma is made from BOTTOM left
       
        # print(name,matches_im0[0] ,yy, pp, persp_to_equirect_deg(matches_im0[0][0], matches_im0[0][1], 512, 256, 45.0, 22.5, yy, pp, 4096, 2048))
        x, _ = matches_im0.shape
        for i in range(0, x):
            matches_equir.append(persp_to_equirect_deg(matches_im0[i][0], matches_im0[i][1], 512, 256, 45, 22.5, yy, pp, 4096, 2048))
            persp_points.append(matches_im1[i])
            
        ### end of pers to equir
        
    matches_equir = np.array(matches_equir)
    persp_points = np.array(persp_points)
    
        
        
    # ### ransac
    # _, inliers = cv2.estimateAffine2D(persp_points, matches_equir)

    # inlier_mask = inliers.ravel().astype(bool)
    # matches_equir = matches_equir[inlier_mask]
    # persp_points  = persp_points[inlier_mask]
    # ### end of ransac 
        
    # plt      
    
    pan_img = cv2.imread("../dataset/2_panorama.jpg")        # BGR
    sec_img = cv2.imread("../dataset/2_query.jpeg")
    
    panorama = cv2.cvtColor(pan_img, cv2.COLOR_BGR2RGB)
    persp = cv2.cvtColor(sec_img, cv2.COLOR_BGR2RGB)

    h1, w1 = panorama.shape[:2]
    h2, w2 = persp.shape[:2]
    H = h1 # max(h1, h2)
    W = w1 # + w2

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:h1, :w1] = panorama
    # canvas[:h2, w1:w1+w2] = persp

    # persp_off = persp_points.copy()
    # persp_off[:, 0] += w1

    plt.figure(figsize=(12, 6))
    plt.imshow(canvas)
    plt.axis('off')

    for (x1, y1) in matches_equir: # , persp_off):
        # line
        # plt.plot([x1, x2], [y1, y2], '-', color='cyan', linewidth=1)
        # endpoints
        plt.scatter([x1], [y1], color='yellow', s=2)

    plt.tight_layout()
    plt.show()

    save_path = "../outputs_third_testing/" + "out.jpg"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
        
            
    