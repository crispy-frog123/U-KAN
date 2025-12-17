import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


class ComplexMatDataset(Dataset):
    """
    读取复数数据（实部+虚部）的.mat文件，合并为2通道

    数据格式：
        - 实部图片：(N, H*W) 即 (20, 4096)
        - 虚部图片：(N, H*W) 即 (20, 4096)
        - 合并成：(N, 2, H, W) 即 (20, 2, 64, 64)

    Args:
        real_img_path (str): 实部图片的.mat文件路径
        imag_img_path (str): 虚部图片的.mat文件路径
        real_mask_path (str): 实部掩码的.mat文件路径
        imag_mask_path (str): 虚部掩码的.mat文件路径
        img_var_name (str): 图片变量名
        mask_var_name (str): 掩码变量名
        img_size (tuple): 图片尺寸 (H, W)，默认(64, 64)
        transform (callable, optional): 数据增强
    """

    def __init__(self,
                 real_img_path,
                 imag_img_path,
                 real_mask_path,
                 imag_mask_path,
                 img_var_name='chi_all_real',
                 mask_var_name='chi_all_real_mask',
                 img_size=(64, 64),
                 transform=None):

        self.transform = transform
        self.img_size = img_size
        self.H, self.W = img_size

        print("=" * 60)
        print("Loading Complex Data (Real + Imaginary)")
        print("=" * 60)

        # 读取实部数据
        print("\n[1/4] Loading Real part images...")
        real_img_data = self._load_mat(real_img_path, img_var_name)
        print(f"  Shape: {real_img_data.shape}, dtype: {real_img_data.dtype}")

        print("[2/4] Loading Real part masks...")
        real_mask_data = self._load_mat(real_mask_path, mask_var_name)
        print(f"  Shape: {real_mask_data.shape}, dtype: {real_mask_data.dtype}")

        # 读取虚部数据
        print("[3/4] Loading Imaginary part images...")
        imag_img_data = self._load_mat(imag_img_path, img_var_name)
        print(f"  Shape: {imag_img_data.shape}, dtype: {imag_img_data.dtype}")

        print("[4/4] Loading Imaginary part masks...")
        imag_mask_data = self._load_mat(imag_mask_path, mask_var_name)
        print(f"  Shape: {imag_mask_data.shape}, dtype: {imag_mask_data.dtype}")

        # Reshape数据
        print("\nReshaping data...")
        real_img = self._reshape_data(real_img_data)  # (20, 64, 64, 1)
        imag_img = self._reshape_data(imag_img_data)  # (20, 64, 64, 1)
        real_mask = self._reshape_data(real_mask_data)  # (20, 64, 64, 1)
        imag_mask = self._reshape_data(imag_mask_data)  # (20, 64, 64, 1)

        # 合并实部和虚部为2通道
        self.images = np.concatenate([real_img, imag_img], axis=-1)
        # (20, 64, 64, 2) - 2通道：[实部, 虚部]

        self.masks = np.concatenate([real_mask, imag_mask], axis=-1)
        # (20, 64, 64, 2) - 2通道：[实部mask, 虚部mask]

        print(f"  Images combined: {self.images.shape}")
        print(f"  Masks combined: {self.masks.shape}")

        # 检查
        assert self.images.shape[0] == self.masks.shape[0], \
            f"Images and masks count mismatch!"

        # 统计信息
        print("\n" + "=" * 60)
        print("Dataset loaded successfully!")
        print("=" * 60)
        print(f"  Total samples: {len(self)}")
        print(f"  Image shape per sample: {self.images.shape[1:]}")
        print(f"  Channels: 2 (Real + Imaginary)")
        print(f"  Value range: [{self.images.min():.4f}, {self.images.max():.4f}]")
        print(f"  Mean: {self.images.mean():.4f}, Std: {self.images.std():.4f}")
        print("=" * 60 + "\n")

    def _load_mat(self, mat_path, var_name):
        """加载.mat文件"""
        try:
            import scipy.io as sio
            data = sio.loadmat(mat_path)
            return data[var_name]
        except:
            with h5py.File(mat_path, 'r') as f:
                data = f[var_name][:]
                if data.shape[0] != self.H * self.W:
                    data = data.T
                return data

    def _reshape_data(self, data):
        """
        将展平的数据reshape回图片

        输入：(N, H*W) 即 (20, 4096)
        输出：(N, H, W, 1)
        """
        N = data.shape[0]

        # 如果是 (H*W, N)，转置
        if data.shape[1] != self.H * self.W:
            data = data.T
            N = data.shape[0]

        # 现在应该是 (N, H*W) = (20, 4096)
        assert data.shape[1] == self.H * self.W, \
            f"Expected {self.H * self.W} pixels per image, got {data.shape[1]}"

        # Reshape: (N, H*W) -> (N, H, W)
        data = data.reshape(N, self.H, self.W)

        # 增加通道维：(N, H, W) -> (N, H, W, 1)
        data = np.expand_dims(data, axis=-1)

        return data.astype('float32')

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        """获取合并的数据（2通道）"""
        # 获取图片和掩码
        img = self.images[idx].copy()  # (64, 64, 2)
        mask = self.masks[idx].copy()  # (64, 64, 2)

        # 归一化到[0, 1]
        img_min = self.images.min()
        img_max = self.images.max()

        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        # 掩码归一化
        if mask.max() > 1.0:
            mask = mask / mask.max()

        # 二值化掩码
        mask = (mask > 0.5).astype('float32')

        # 数据增强
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # 转置：(H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))  # (2, 64, 64)
        mask = np.transpose(mask, (2, 0, 1))  # (2, 64, 64)

        # 转tensor
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return img, mask, {'img_id': f'sample_{idx}'}


class TransformSubset(torch.utils.data.Subset):
    """支持transform的Subset"""

    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, mask, meta = self.dataset[self.indices[idx]]

        if self.transform is not None:
            # 转回numpy做增强
            img_np = img.numpy().transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
            mask_np = mask.numpy().transpose(1, 2, 0)

            # 数据增强
            augmented = self.transform(image=img_np, mask=mask_np)
            img_np = augmented['image']
            mask_np = augmented['mask']

            # 转回tensor
            img = torch.from_numpy(img_np.transpose(2, 0, 1))  # (H,W,C) -> (C,H,W)
            mask = torch.from_numpy(mask_np.transpose(2, 0, 1))

        return img, mask, meta
