from core.dataset.transforms import DataTransform, DataTransform_Test, DataTransform_3D, DataTransform_3DTest
from core.dataset.subsample import MaskFunc
from core.dataset.mri_data import SliceData
from core.dataset.mri_data3d import Data3D
from torch.utils.data import DataLoader
import json
import h5py
def create_datasets(cfg):
    mask_func = MaskFunc(cfg.center_fractions, cfg.accelerations)
    if cfg.type=='slice':
        data = SliceData(
            root=cfg.data_path,
            transform=DataTransform(mask_func, cfg.resolution, cfg.challenge, cfg.use_seed, cfg.crop, cfg.crop_size),
            sample_rate=cfg.sample_rate,
            challenge=cfg.challenge,
        )
    elif cfg.type=='3d':
        data = Data3D(
            root=cfg.data_path,
            transform=DataTransform_3D(mask_func, cfg.resolution, cfg.challenge, cfg.use_seed, cfg.crop, cfg.crop_size),
            challenge=cfg.challenge,
        )
    return data

def create_data_loaders(cfg):
    train_data = create_datasets(cfg.data.train)
    # train_data = [train_data[i] for i in range(320)]
    dev_data = create_datasets(cfg.data.val)
    # dev_data = [dev_data[i] for i in range(cfg.data.val.val_count)]
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.data.train.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=cfg.data.val.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=cfg.data.val.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader

def create_loader_for_infer(cfg):
    data_type = cfg.data.val.type
    cfg = cfg.infer_cfg
    mask_func = None
    if cfg.mask_kspace:
        mask_func = MaskFunc(cfg.center_fractions, cfg.accelerations)
    if data_type == 'slice':
        data = SliceData(
            root=cfg.data_path,
            transform=DataTransform_Test(cfg.resolution, cfg.challenge, mask_func),
            sample_rate=1.,
            challenge=cfg.challenge
        )
    else:
        data = Data3D(
            root=cfg.data_path,
            transform=DataTransform_3DTest(mask_func, cfg.resolution, cfg.challenge),
            challenge=cfg.challenge
        )
    data_loader = DataLoader(
        dataset=data,
        batch_size=cfg.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    return data_loader

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.
    Args:
        data (torch.Tensor): Input data to be converted to numpy.

    Returns:
        np.array: Complex numpy version of data
    """
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]