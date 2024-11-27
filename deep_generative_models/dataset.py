import torch
import h5py
import random
import numpy as np
import pandas as pd
from typing import Iterator
from torchvision.transforms import Normalize
from torch.utils.data import Dataset, Sampler, DataLoader
from config.paths import GLOBAL_STATS


class HDF5Dataset(Dataset):
    def __init__(self, file_path: str, transform=None) -> None:
        self.file_path = file_path
        self.glob_stats = pd.read_csv(GLOBAL_STATS)
        self.glob_mean = np.mean(self.glob_stats["mean"])
        self.glob_std = np.mean(self.glob_stats["std"])
        self.transform = transform

    def _open_hdf5(self) -> None:
        self._hf = h5py.File(self.file_path, "r")

    def __len__(self) -> int:
        # placeholder as it will be overwritten by sampler
        return 1

    def __getitem__(self, idx: tuple) -> torch.Tensor:
        # NOTE: Method __getitem__ should have 2 parameters. sonarlint(ipython:S5722)
        brain, image, row, column, tile_size = idx

        # load the image
        if not hasattr(self, "_hf"):
            self._open_hdf5()
        image = self._hf[brain][image]

        # get the tile
        tile = image[row : row + tile_size, column : column + tile_size]

        # ensure the tile is in the required shape (1, H, W) and type (float32)
        tile = torch.tensor(tile[None, :, :], dtype=torch.float32)

        # normalize
        tile = Normalize(mean=[self.glob_mean], std=[self.glob_std])(tile)

        # TODO: You might want to experiment with several
        # data augmentations in your training Dataset
        if self.transform:
            tile = self.transform(tile)

        return tile


class HDF5Sampler(Sampler):
    def __init__(
        self, file_path: str, brains: list[str], tile_size: int, tiles_per_epoch: int
    ) -> None:
        self.file_path = file_path
        self.brains = brains
        self.tile_size = tile_size
        self.tiles_per_epoch = tiles_per_epoch
        self.max_retries = 20

    def _open_hdf5(self):
        self._hf = h5py.File(self.file_path, "r")

    # NOTE: Overwriting the len method sets the number of example tiles you draw per epoch
    def __len__(self) -> int:
        return self.tiles_per_epoch

    def _is_outlier(self, image, lower_threshold, upper_threshold, percentage=0.1):
        below_lower = image < lower_threshold
        above_upper = image > upper_threshold

        below_lower_count = below_lower.sum()
        above_upper_count = above_upper.sum()

        below_condition = below_lower_count > percentage * image.size
        above_condition = above_upper_count > percentage * image.size

        return below_condition | above_condition

    # NOTE: Overwrite the iter method such that it yields random tuples of
    # (brain, image, row, column, tile size) based on your train brains,
    # their image shapes and the selected tile size
    def __iter__(self) -> Iterator[int]:
        if not hasattr(self, "_hf"):
            self._open_hdf5()

        tiles = []
        for _ in range(self.tiles_per_epoch):
            retries = 0
            while retries < self.max_retries:
                brain = random.choice(self.brains)
                image = random.choice(list(self._hf[brain].keys()))

                row_len, column_len = self._hf[brain][image].shape
                row_range = row_len - self.tile_size
                column_range = column_len - self.tile_size

                row = random.randint(0, row_range)
                column = random.randint(0, column_range)

                tile = self._hf[brain][image][
                    row : row + self.tile_size, column : column + self.tile_size
                ]

                if not self._is_outlier(tile, 20, 230):
                    tiles.append((brain, image, row, column, self.tile_size))
                    break
                retries += 1

            if retries == self.max_retries:
                print(
                    "Warning: Maximum retries reached for tile sampling. Skipping this tile."
                )

        return iter(tiles)


def create_dataloader(
    hdf5_file_path: str,
    brains: list[str],
    tile_size: int = 64,
    batch_size: int = 8,
    tiles_per_epoch: int = 1000,
    num_workers: int = 0,
) -> DataLoader:
    r"""
    Create a DataLoader for the given HDF5 file and brains.

    Args:
        hdf5_file_path (str): location of the HDF5 file
        brains (list[str]): list of brain names in the HDF5 file e.g. 'B01', 'B02'
        tile_size (int, optional): Size of one square cutout of one brain image. Defaults to 64.
        batch_size (int, optional): How many samples per batch to load. Defaults to 8.
        tiles_per_epoch (int, optional): How many sample tiles to draw per epoch. Defaults to 1000.

    Returns:
        DataLoader: Pytorch DataLoader object for the given HDF5 file and brains.

    Example:
        >>> hdf5_file_path = 'cell_data.h5'
        >>> train_brains = ['B01', 'B02', 'B05']
        >>> test_brains = ['B20']
        >>> tile_size = 64
        >>> batch_size = 8
        >>> tiles_per_epoch = 1000
        >>> train_loader = create_dataloader(hdf5_file_path, train_brains, tile_size, batch_size, tiles_per_epoch)
        >>> test_loader = create_dataloader(hdf5_file_path, test_brains, tile_size, batch_size, tiles_per_epoch)
    """
    dataset = HDF5Dataset(hdf5_file_path)
    sampler = HDF5Sampler(hdf5_file_path, brains, tile_size, tiles_per_epoch)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    return dataloader
