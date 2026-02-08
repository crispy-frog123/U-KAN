import os
import json
import torch
import numpy as np
import scipy.io as sio
import h5py
from torch.utils.data import Dataset


class ComplexMatDataset(Dataset):
    def __init__(self, real_img_path, imag_img_path, real_label_path, imag_label_path,
                 img_size=(64, 64), transform=None, normalize_method='z-score'):
        """
        Modified Dataset for DeepNIS:
        1. Input (chi0): Uses Z-Score Normalization.
        2. Label (chi):  USES NO NORMALIZATION (Raw values [-2, 0]).
        """
        self.transform = transform
        self.normalize_method = normalize_method

        # 1. Load Data
        self.real_img = self._load_mat(real_img_path)
        self.imag_img = self._load_mat(imag_img_path)
        self.real_label = self._load_mat(real_label_path)
        self.imag_label = self._load_mat(imag_label_path)

        # Ensure correct shape (N, H, W) or (H, W, N) -> (N, H, W)
        self.real_img = self._check_dims(self.real_img)
        self.imag_img = self._check_dims(self.imag_img)
        self.real_label = self._check_dims(self.real_label)
        self.imag_label = self._check_dims(self.imag_label)

        self.img_size = img_size

        # 2. Initialize Normalization Stats (Default to Identity/No-op)
        # Input Stats (will be updated by calculate_normalization_stats)
        self.inp_real_mean = 0.0
        self.inp_real_std = 1.0
        self.inp_imag_mean = 0.0
        self.inp_imag_std = 1.0

        # Label Stats (Not used for normalization, but kept for reference)
        self.lbl_real_max = 1.0
        self.lbl_real_min = 0.0

        self.stats_calculated = False

    def _load_mat(self, path):
        try:
            data = sio.loadmat(path)
            keys = [k for k in data.keys() if not k.startswith('__')]
            return data[keys[0]]
        except NotImplementedError:
            with h5py.File(path, 'r') as f:
                keys = list(f.keys())
                return f[keys[0]][:]

    def _check_dims(self, data):
        # Assuming data is (H*W, N) or (N, H*W) or similar flat structure based on previous context
        # Adjust this if your raw .mat is already (N, 64, 64)
        if data.ndim == 2:
            # Heuristic: usually N is 2000 or 4000, 4096 is 64*64
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
        Calculate stats ONLY for Inputs using training indices.
        Labels are intentionally NOT normalized to preserve sparsity (0 background).
        """
        print(f"Computing Input normalization stats on {len(indices)} training samples...")

        # Select training subset
        train_real = self.real_img[indices]
        train_imag = self.imag_img[indices]

        # Calculate Z-Score params for INPUT
        self.inp_real_mean = float(np.mean(train_real))
        self.inp_real_std = float(np.std(train_real)) + 1e-8  # avoid div by zero

        self.inp_imag_mean = float(np.mean(train_imag))
        self.inp_imag_std = float(np.std(train_imag)) + 1e-8

        self.stats_calculated = True
        print(f"  [Input Real] Mean: {self.inp_real_mean:.4f}, Std: {self.inp_real_std:.4f}")
        print(f"  [Input Imag] Mean: {self.inp_imag_mean:.4f}, Std: {self.inp_imag_std:.4f}")
        print(f"  [Label] skipped (keeping raw values for physics consistency).")

    def save_stats(self, save_path):
        """Save input stats to JSON for consistency in Testing."""
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
        """Load input stats from JSON."""
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
        # 1. Get Data
        input_r = self.real_img[idx]
        input_i = self.imag_img[idx]
        label_r = self.real_label[idx]
        label_i = self.imag_label[idx]

        # 2. Normalize INPUT ONLY (Z-Score)
        # (input - mean) / std
        input_r = (input_r - self.inp_real_mean) / self.inp_real_std
        input_i = (input_i - self.inp_imag_mean) / self.inp_imag_std

        # 3. Handle LABEL (Do NOT normalize)
        # Keep them as raw float values.
        # If you want to scale them slightly (e.g. /2.0), do it here, but raw is fine.

        # 4. To Tensor (C, H, W)
        input_tensor = torch.stack([
            torch.from_numpy(input_r).float(),
            torch.from_numpy(input_i).float()
        ], dim=0)

        label_tensor = torch.stack([
            torch.from_numpy(label_r).float(),
            torch.from_numpy(label_i).float()
        ], dim=0)

        # 5. Apply Transforms (if any, usually Resize)
        # Note: Transform logic usually expects HWC or CHW.
        # Since we manually stacked CHW, make sure transform handles it or apply before stack.
        # Assuming transform is just Resize which works on Tensor or PIL.
        # If your transform is complex (albumentations), it needs numpy HWC.
        # Keeping it simple for now based on your code:
        if self.transform:
            # Naive transform application (assuming simple torch transforms)
            # If using Albumentations, strict adaptation is needed.
            # Here we return tensor directly as your code seemed to handle transforms externally or minimally.
            pass

        return input_tensor, label_tensor, idx


class TransformSubset(Dataset):
    """
    Subset wrapper to apply transforms dynamically
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        input_tensor, label_tensor, _ = self.dataset[real_idx]

        # Apply Albumentations or Resize here if needed
        # Convert to numpy HWC for Albumentations
        if self.transform:
            input_np = input_tensor.permute(1, 2, 0).numpy()  # (H, W, C)
            label_np = label_tensor.permute(1, 2, 0).numpy()

            # Apply same transform to input and label? usually inputs only for aug,
            # but resize must be both.
            # Assuming 'transform' is just Resize for now based on context.
            augmented = self.transform(image=input_np, mask=label_np)
            input_np = augmented['image']
            label_np = augmented['mask']

            input_tensor = torch.from_numpy(input_np).permute(2, 0, 1).float()
            label_tensor = torch.from_numpy(label_np).permute(2, 0, 1).float()

        return input_tensor, label_tensor, real_idx

    def __len__(self):
        return len(self.indices)