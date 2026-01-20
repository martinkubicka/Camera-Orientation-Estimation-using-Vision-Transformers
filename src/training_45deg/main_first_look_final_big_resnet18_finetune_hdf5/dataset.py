import os
from torch.utils.data import Dataset
import numpy as np
import h5py
import torch

DATASET_PATH = "../../training/dataset/geopose_lar_augmented512_tiled.h5" # TODO

class OrientationDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.hdf5_path = DATASET_PATH
        self.transform = transform

        # Open & keep file open for entire lifetime
        self.h5f = h5py.File(self.hdf5_path, "r")

        # List all sample keys once
        self.keys = list(self.h5f["panoramas"].keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        base = self.keys[idx]
        grp = self.h5f["panoramas"][base]

        # Load query image (stored as [3,H,W] float32)
        query_img = torch.from_numpy(grp["query_image"][:])
        
        mean = torch.tensor([0.485, 0.456, 0.406], device=query_img.device).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=query_img.device).view(3, 1, 1)

        query_img = (query_img - mean) / std

        # Load all tiles (stored as [32,3,512,512])
        tiles = torch.from_numpy(grp["tiles"][:])

        tiles = (tiles - mean) / std

        # Load GT
        gt = torch.from_numpy(grp["ground_truth"][:])
        pitch_gt, yaw_gt, roll_gt, _ = gt.tolist()

        return query_img, tiles, pitch_gt, yaw_gt, roll_gt

    def close(self):
        self.h5f.close()
