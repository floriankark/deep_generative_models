import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd

class HDF5Dataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None):
        self.file_path = file_path
        self.transform = transform
        self.target_transform = target_transform
    
    def _open_hdf5(self):
        self._hf = h5py.File(self.file_path, 'r')

    def __getitem__(self, brain, image, row, column, tile_size):
        if not hasattr(self, '_hf'):
            self._open_hdf5()
        
        img = self._hf[brain][image][row:row+tile_size, column:column+tile_size]
        glob_stats = pd.read_csv("global_stats.csv")
        glob_mean = np.mean(glob_stats["mean"])
        glob_std = np.mean(glob_stats["std"])
        
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        img = (img - glob_mean) / glob_std
        
        if self.transform:
            img = self.transform(img)
            
        assert img.shape == (1, tile_size, tile_size), f"Shape is {img.shape} and must be (1, {tile_size}, {tile_size})"
        return img