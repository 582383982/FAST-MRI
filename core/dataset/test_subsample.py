import pytest
import h5py
import matplotlib.pyplot as plt
from core.dataset.subsample import MaskFunc
from core.dataset import common
from core.dataset.transforms import DataTransform
import os

def to_img(kspace, target, which_challenge, mask_func, resolution):
    kspace = common.to_tensor(kspace)
    seed = tuple(map(ord, '233'))
    masked_kspace = kspace
    if mask_func is not None:
        masked_kspace, mask = common.apply_mask(kspace, mask_func, seed)
    # Inverse Fourier Transform to get zero filled solution
    image = common.ifft2(masked_kspace)
    # Crop input image
    image = common.complex_center_crop(image, (resolution, resolution))
    # Absolute value
    image = common.complex_abs(image)
    # Apply Root-Sum-of-Squares if multicoil data
    if which_challenge == 'multicoil':
        image = common.root_sum_of_squares(image)
    image, mean, std = common.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)
    target = common.to_tensor(target)
    image, target = common.random_crop(image, target, 196)
    # Normalize target
    target = common.normalize(target, mean, std, eps=1e-11)
    target = target.clamp(-6, 6)
    return image.numpy(), target.numpy()

# @pytest.mark.parametrize('data_path, which_challenge', [
#     ('data/singlecoil_train/file1000000.h5', 'singlecoil'),
#     ('data/singlecoil_train/file1000000.h5', 'singlecoil')
# ])
def test_data(data_path, which_challenge):
    print(data_path)
    sample = h5py.File(os.path.abspath(data_path))
    kspace = sample['kspace']
    mask_func_x4 = MaskFunc(center_fractions=[0.08], accelerations=[4]) 
    mask_func_x8 = MaskFunc(center_fractions=[0.04], accelerations=[8])
    mask_func_x16 = MaskFunc(center_fractions=[0.02], accelerations=[16])
    target_esc = sample['reconstruction_esc']
    target_rss = sample['reconstruction_rss']

    for i, s in enumerate(kspace):
        y_esc = target_esc[i]
        y_rss = target_rss[i]

        image, _ = to_img(s, y_esc, which_challenge, None, 320)
        image_x4, y_esc = to_img(s, y_esc, which_challenge, mask_func_x4, 320)
        image_x8, y_rss = to_img(s, y_rss, which_challenge, mask_func_x8, 320)
        image_x16, _ = to_img(s, y_rss, which_challenge, mask_func_x16, 320)

        plt.subplot(231)
        plt.imshow(y_esc, cmap='gray')
        plt.subplot(232)
        plt.imshow(y_rss, cmap='gray')
        plt.subplot(233)
        plt.imshow(image, cmap='gray')
        plt.subplot(234)
        plt.imshow(image_x4, cmap='gray')
        plt.subplot(235)
        plt.imshow(image_x8, cmap='gray')
        plt.subplot(236)
        plt.imshow(image_x16, cmap='gray')
        plt.show()
    assert 1==1
