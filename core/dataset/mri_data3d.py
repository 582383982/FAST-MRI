import pathlib
import random

import h5py
from torch.utils.data import Dataset
class Data3D(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, crop_size=None):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        self.files = list(pathlib.Path(root).iterdir())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace']
            target = data[self.recons_key] if self.recons_key in data else None
            return self.transform(kspace, target, data.attrs, fname.name)