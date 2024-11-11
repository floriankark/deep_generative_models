import torch
import h5py
import numpy as np
import pandas as pd
from typing import Iterator
from torch.utils.data import Dataset
from torch.utils.data import Sampler

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

# NOTE: make sure index is not out of bounds
class HDF5Sampler(Sampler):
    def __init__(self, data: HDF5Dataset):
        self.data = data
        
    # NOTE: Overwriting the len method sets the number of example tiles you draw per epoch
    def __len__(self) -> int:
        return len(self.data)
        
    # NOTE: Overwrite the iter method such that it yields random tuples of 
    # (brain, image, row, column, tile size) based on your train brains, 
    # their image shapes and the selected tile size
    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data)))