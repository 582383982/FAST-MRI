from core.dataset import common
import numpy as np
class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True, crop=False, crop_size=48):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.crop = crop
        self.crop_size = crop_size

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        kspace = common.to_tensor(kspace)
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        masked_kspace, mask = common.apply_mask(kspace, self.mask_func, seed)
        # Inverse Fourier Transform to get zero filled solution
        image = common.ifft2(masked_kspace)
        # Crop input image
        image = common.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        image = common.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = common.root_sum_of_squares(image)
        # Normalize input
        image, mean, std = common.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)
        target = common.to_tensor(target)
        # Normalize target
        target = common.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
        # Apply random crop
        if self.crop:
            image, target = common.random_crop(image, target, self.crop_size)
        return image, target, mean, std, attrs['norm'].astype(np.float32)

class DataTransform_Patch:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True, patch_size=48):
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        kspace = common.to_tensor(kspace)
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        masked_kspace, mask = common.apply_mask(kspace, self.mask_func, seed)
        # Inverse Fourier Transform to get zero filled solution
        image = common.ifft2(masked_kspace)
        # Crop input image
        image = common.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        image = common.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = common.root_sum_of_squares(image)
        # Normalize input
        image, mean, std = common.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        target = common.to_tensor(target)
        # Normalize target
        target = common.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
        return image, target, mean, std, attrs['norm'].astype(np.float32)

class DataTransform_Test:
    """
    Data Transformer for running U-Net models on a test dataset.
    """

    def __init__(self, resolution, which_challenge, mask_func=None):
        """
        Args:
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.mask_func = mask_func

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.Array): k-space measurements
            target (numpy.Array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object
            fname (pathlib.Path): Path to the input file
            slice (int): Serial number of the slice
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Normalized zero-filled input image
                mean (float): Mean of the zero-filled image
                std (float): Standard deviation of the zero-filled image
                fname (pathlib.Path): Path to the input file
                slice (int): Serial number of the slice
        """
        kspace = common.to_tensor(kspace)
        if self.mask_func is not None:
            seed = tuple(map(ord, fname))
            masked_kspace, _ = common.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
        # Inverse Fourier Transform to get zero filled solution
        image = common.ifft2(masked_kspace)
        # Crop input image
        image = common.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        image = common.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = common.root_sum_of_squares(image)
        # Normalize input
        image, mean, std = common.normalize_instance(image)
        image = image.clamp(-6, 6)
        return image, mean, std, fname, slice