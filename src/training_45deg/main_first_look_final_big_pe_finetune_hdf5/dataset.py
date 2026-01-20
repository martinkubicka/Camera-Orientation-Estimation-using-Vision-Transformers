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

        # Apply your transform only to query image
        # if self.transform:
        #     query_img = self.transform(query_img)
        
        query_img = (query_img - 0.5) / 0.5 # TODO

        # Load all tiles (stored as [32,3,512,512])
        tiles = torch.from_numpy(grp["tiles"][:])

        tiles = (tiles - 0.5) / 0.5 # TODO

        # Load GT
        gt = torch.from_numpy(grp["ground_truth"][:])
        pitch_gt, yaw_gt, roll_gt, _ = gt.tolist()

        return query_img, tiles, pitch_gt, yaw_gt, roll_gt

    def close(self):
        self.h5f.close()
