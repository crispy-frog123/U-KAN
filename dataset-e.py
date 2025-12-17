import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


class MatDatasetFlattened(Dataset):
    """
    读取展平的.mat数据（每行是一张图片）

    数据格式：
        - 输入：(N, H*W) 即 (20, 4096)
        - 需要reshape成 (N, H, W) 即 (20, 64, 64)

    Args:
        img_mat_path (str): 输入图片的.mat文件路径
        mask_mat_path (str): 掩码的.mat文件路径（可选）
        img_var_name (str): 图片变量名
        mask_var_name (str): 掩码变量名
        img_size (tuple): 图片尺寸 (H, W)，默认(64, 64)
        transform (callable, optional): 数据增强
    """

    def __init__(self,
                 img_mat_path,
                 mask_mat_path=None,
                 img_var_name='chi_all_imag',
                 mask_var_name='chi_all_mask',
                 img_size=(64, 64),
                 transform=None):

        self.transform = transform
        self.img_size = img_size
        self.H, self.W = img_size

        # 读取图片数据
        print(f"Loading images from {img_mat_path}...")
        img_data = self._load_mat(img_mat_path, img_var_name)

        # 显示原始形状
        print(f"  Raw data shape: {img_data.shape}")
        print(f"  Data type: {img_data.dtype}")

        # 转换形状：(20, 4096) -> (20, 64, 64, 1)
        self.images = self._reshape_data(img_data)
        print(f"  Reshaped to: {self.images.shape}")

        # 读取掩码数据
        if mask_mat_path is not None:
            print(f"Loading masks from {mask_mat_path}...")
            mask_data = self._load_mat(mask_mat_path, mask_var_name)
            print(f"  Raw data shape: {mask_data.shape}")
            self.masks = self._reshape_data(mask_data)
            print(f"  Reshaped to: {self.masks.shape}")
        else:
            # 如果没有掩码，创建全0掩码
            print("No mask file provided, creating dummy masks...")
            self.masks = np.zeros_like(self.images)

        # 检查
        assert self.images.shape[0] == self.masks.shape[0], \
            f"Images and masks count mismatch!"

        # 统计信息
        print(f"\nDataset loaded successfully!")
        print(f"  Total samples: {len(self)}")
        print(f"  Image shape per sample: {self.images.shape[1:]}")
        print(f"  Value range: [{self.images.min():.4f}, {self.images.max():.4f}]")
        print(f"  Mean: {self.images.mean():.4f}, Std: {self.images.std():.4f}")

    def _load_mat(self, mat_path, var_name):
        """
        加载.mat文件
        """
        try:
            import scipy.io as sio
            data = sio.loadmat(mat_path)
            return data[var_name]
        except:
            with h5py.File(mat_path, 'r') as f:
                data = f[var_name][:]
                # h5py可能需要转置
                if data.shape[0] != self.H * self.W:
                    data = data.T
                return data

    def _reshape_data(self, data):
        """
        将展平的数据reshape回图片

        输入：(N, H*W) 即 (20, 4096)
        输出：(N, H, W, 1)
        """
        # 确保形状正确
        N = data.shape[0]

        # 如果是 (H*W, N)，转置
        if data.shape[1] != self.H * self.W:
            data = data.T
            N = data.shape[0]

        # 现在应该是 (N, H*W) = (20, 4096)
        assert data.shape[1] == self.H * self.W, \
            f"Expected {self.H * self.W} pixels per image, got {data.shape[1]}"

        # Reshape成图片：(N, H*W) -> (N, H, W)
        data = data.reshape(N, self.H, self.W)

        # 增加通道维：(N, H, W) -> (N, H, W, 1)
        data = np.expand_dims(data, axis=-1)

        return data.astype('float32')

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # 获取单张图片和掩码
        img = self.images[idx].copy()  # (H, W, 1)
        mask = self.masks[idx].copy()  # (H, W, 1)

        # 归一化到[0, 1]
        img_min = self.images.min()
        img_max = self.images.max()

        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        # 掩码归一化（如果不是0-1）
        if mask.max() > 1.0:
            mask = mask / mask.max()

        # 确保掩码是二值的
        mask = (mask > 0.5).astype('float32')

        # 数据增强
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # 转置：(H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        # 转tensor
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return img, mask, {'img_id': f'sample_{idx}'}


class TransformSubset(torch.utils.data.Subset):
    """
    支持transform的Subset
    """

    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        # 获取原始数据（已经是tensor）
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
