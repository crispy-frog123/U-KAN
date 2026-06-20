"""Dataset utilities for complex-valued inverse-scattering data.

The dataset loader reads real/imaginary inputs and labels from MAT files and
applies z-score normalization to inputs while keeping labels in physical scale.
"""

import json
import os

import h5py
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset


class ComplexMatDataset(Dataset):
    """PyTorch dataset for paired real/imaginary input-label MAT tensors."""

    def __init__(self, real_img_path, imag_img_path, real_label_path, imag_label_path,
                 img_size=(64, 64), transform=None, normalize_method='z-score'):
        """
        Dataset for complex-valued tomography data.
        Input (chi0) is z-score normalized, whereas labels (chi) remain unnormalized.
        """
        self.transform = transform
        self.normalize_method = normalize_method

        # Load raw arrays from MAT files.
        self.real_img = self._load_mat(real_img_path)
        self.imag_img = self._load_mat(imag_img_path)
        self.real_label = self._load_mat(real_label_path)
        self.imag_label = self._load_mat(imag_label_path)

        # Enforce sample-major layout: (N, H, W).
        self.real_img = self._check_dims(self.real_img)
        self.imag_img = self._check_dims(self.imag_img)
        self.real_label = self._check_dims(self.real_label)
        self.imag_label = self._check_dims(self.imag_label)

        self.img_size = img_size

        # Initialize input normalization statistics.
        self.inp_real_mean = 0.0
        self.inp_real_std = 1.0
        self.inp_imag_mean = 0.0
        self.inp_imag_std = 1.0

        # Keep label range metadata for analysis.
        self.lbl_real_max = 1.0
        self.lbl_real_min = 0.0

        self.stats_calculated = False

    @staticmethod
    def _load_mat(path):
        """Load the first non-private variable from a MATLAB file."""
        try:
            data = sio.loadmat(path)
            keys = [k for k in data.keys() if not k.startswith('__')]
            return data[keys[0]]
        except NotImplementedError:
            with h5py.File(path, 'r') as f:
                keys = list(f.keys())
                return f[keys[0]][:]

    @staticmethod
    def _check_dims(data):
        # Convert flattened representations into (N, 64, 64) when needed.
        if data.ndim == 2:
            # Heuristic: 4096 = 64 x 64.
            if data.shape[0] == 4096:
                # (4096, N) -> (N, 64, 64)
                N = data.shape[1]
                data = data.T.reshape(N, 64, 64)
            elif data.shape[1] == 4096:
                # (N, 4096) -> (N, 64, 64)
                N = data.shape[0]
                data = data.reshape(N, 64, 64)
        return data

    def calculate_normalization_stats(self, indices):
        """
        Compute input normalization statistics from training samples only.
        Labels are intentionally kept in physical scale.
        """
        print(f"Computing Input normalization stats on {len(indices)} training samples...")

        # Select training subset.
        train_real = self.real_img[indices]
        train_imag = self.imag_img[indices]

        # Compute z-score parameters for the input channels.
        self.inp_real_mean = float(np.mean(train_real))
        self.inp_real_std = float(np.std(train_real)) + 1e-8  # Numerical safeguard.

        self.inp_imag_mean = float(np.mean(train_imag))
        self.inp_imag_std = float(np.std(train_imag)) + 1e-8

        self.stats_calculated = True
        print(f"  [Input Real] Mean: {self.inp_real_mean:.4f}, Std: {self.inp_real_std:.4f}")
        print(f"  [Input Imag] Mean: {self.inp_imag_mean:.4f}, Std: {self.inp_imag_std:.4f}")
        print("  [Label] skipped (keeping raw values for physics consistency).")

    def save_stats(self, save_path):
        """Persist input statistics to JSON for reproducible evaluation."""
        stats = {
            'inp_real_mean': self.inp_real_mean,
            'inp_real_std': self.inp_real_std,
            'inp_imag_mean': self.inp_imag_mean,
            'inp_imag_std': self.inp_imag_std,
            'method': self.normalize_method
        }
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Normalization stats saved to {save_path}")

    def load_stats(self, load_path):
        """Load previously saved input statistics from JSON."""
        if not os.path.exists(load_path):
            print(f"Warning: Stats file {load_path} not found! Using Identity norm.")
            return

        with open(load_path, 'r') as f:
            stats = json.load(f)

        self.inp_real_mean = stats['inp_real_mean']
        self.inp_real_std = stats['inp_real_std']
        self.inp_imag_mean = stats['inp_imag_mean']
        self.inp_imag_std = stats['inp_imag_std']
        self.stats_calculated = True
        print(f"Normalization stats loaded from {load_path}")

    def __len__(self):
        return self.real_img.shape[0]

    def __getitem__(self, idx):
        # Retrieve raw sample.
        input_r = self.real_img[idx]
        input_i = self.imag_img[idx]
        label_r = self.real_label[idx]
        label_i = self.imag_label[idx]

        # Apply z-score normalization to input channels only.
        input_r = (input_r - self.inp_real_mean) / self.inp_real_std
        input_i = (input_i - self.inp_imag_mean) / self.inp_imag_std

        # Labels remain unnormalized in physical units.

        # Convert to channel-first tensors: (C, H, W).
        input_tensor = torch.stack([
            torch.from_numpy(input_r).float(),
            torch.from_numpy(input_i).float()
        ], dim=0)

        label_tensor = torch.stack([
            torch.from_numpy(label_r).float(),
            torch.from_numpy(label_i).float()
        ], dim=0)

        return input_tensor, label_tensor, idx


class TransformSubset(Dataset):
    """
    Subset wrapper that applies transforms at retrieval time.
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        input_tensor, label_tensor, _ = self.dataset[real_idx]

        # Apply transform in HWC format for Albumentations compatibility.
        if self.transform:
            input_np = input_tensor.permute(1, 2, 0).numpy()  # (H, W, C)
            label_np = label_tensor.permute(1, 2, 0).numpy()

            # Apply joint transform to preserve input-label alignment.
            augmented = self.transform(image=input_np, mask=label_np)
            input_np = augmented['image']
            label_np = augmented['mask']

            input_tensor = torch.from_numpy(input_np).permute(2, 0, 1).float()
            label_tensor = torch.from_numpy(label_np).permute(2, 0, 1).float()

        return input_tensor, label_tensor, real_idx

    def __len__(self):
        return len(self.indices)
