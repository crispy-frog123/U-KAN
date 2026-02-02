import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


class ComplexMatDataset(Dataset):
    """
    读取复数数据（实部+虚部）的.mat文件，合并为2通道
    """

    def __init__(self,
                 real_img_path,
                 imag_img_path,
                 real_mask_path,
                 imag_mask_path,
                 img_var_name=None,
                 mask_var_name=None,
                 img_size=(64, 64),
                 transform=None,
                 normalize_method='z-score'):

        self.transform = transform
        self.img_size = img_size
        self.H, self.W = img_size
        self.normalize_method = normalize_method

        # ========== 变量名处理 ==========
        if img_var_name in [None, 'None', '']:
            img_var_name = None
        if mask_var_name in [None, 'None', '']:
            mask_var_name = None

        real_img_var = img_var_name
        imag_img_var = img_var_name
        real_mask_var = mask_var_name
        imag_mask_var = mask_var_name
        # ===============================

        print("=" * 60)
        print("Loading Complex Data (Real + Imaginary)")
        print("=" * 60)

        # 加载数据
        print("[1/4] Loading Real part images...")
        real_img_data = self._load_mat(real_img_path, real_img_var)

        print("[2/4] Loading Imaginary part images...")
        imag_img_data = self._load_mat(imag_img_path, imag_img_var)

        print("[3/4] Loading Real part masks...")
        real_mask_data = self._load_mat(real_mask_path, real_mask_var)

        print("[4/4] Loading Imaginary part masks...")
        imag_mask_data = self._load_mat(imag_mask_path, imag_mask_var)

        # Reshape数据
        print("\nReshaping data...")
        real_img = self._reshape_data(real_img_data)
        imag_img = self._reshape_data(imag_img_data)
        real_mask = self._reshape_data(real_mask_data)
        imag_mask = self._reshape_data(imag_mask_data)

        # 合并实部和虚部为2通道
        self.images = np.concatenate([real_img, imag_img], axis=-1)
        self.masks = np.concatenate([real_mask, imag_mask], axis=-1)

        print(f"  Images combined: {self.images.shape}")
        print(f"  Masks combined: {self.masks.shape}")

        # 检查
        assert self.images.shape[0] == self.masks.shape[0], \
            f"Images and masks count mismatch!"

        # ========== 修改点 1: 初始化归一化参数默认值 ==========
        # 不再自动计算，而是先给默认值（防止未调用计算方法时报错）
        self.real_mean, self.real_std = 0.0, 1.0
        self.imag_mean, self.imag_std = 0.0, 1.0
        self.real_min, self.real_max = 0.0, 1.0
        self.imag_min, self.imag_max = 0.0, 1.0

        self.stats_calculated = False  # 标记位
        # ====================================================

        # 统计信息
        print("\n" + "=" * 60)
        print("Dataset loaded successfully!")
        print("=" * 60)
        print(f"  Total samples: {len(self)}")
        print(f"  Image shape per sample: {self.images.shape[1:]}")
        print(f"  Channels: 2 (Real + Imaginary)")
        print(f"  Normalize method: {self.normalize_method}")
        print(
            "  (Note: Normalization stats not calculated yet. Call 'calculate_normalization_stats' with training indices.)")
        print("=" * 60 + "\n")

    def _load_mat(self, mat_path, var_name):
        """加载.mat文件，支持自动推断变量名"""
        try:
            import scipy.io as sio
            data = sio.loadmat(mat_path)
            if var_name is None:
                available_vars = [k for k in data.keys() if not k.startswith('__')]
                if not available_vars:
                    raise ValueError(f"No valid variables in {mat_path}")
                var_name = available_vars[0]
            return data[var_name]
        except:
            with h5py.File(mat_path, 'r') as f:
                if var_name is None:
                    available_vars = list(f.keys())
                    if not available_vars:
                        raise ValueError(f"No valid variables in {mat_path}")
                    var_name = available_vars[0]
                data = f[var_name][:]
                if data.shape[0] != self.H * self.W:
                    data = data.T
                return data

    def _reshape_data(self, data):
        """将展平的数据reshape回图片"""
        N = data.shape[0]
        if data.shape[1] != self.H * self.W:
            data = data.T
            N = data.shape[0]
        assert data.shape[1] == self.H * self.W, \
            f"Expected {self.H * self.W} pixels per image, got {data.shape[1]}"
        data = data.reshape(N, self.H, self.W)
        data = np.expand_dims(data, axis=-1)
        return data.astype('float32')

    # ========== 修改点 2: 将私有方法改为公开方法，并支持 indices 参数 ==========
    def calculate_normalization_stats(self, indices=None):
        """
        计算归一化参数。
        为了避免数据泄露，应该只传入训练集的 indices。
        """
        if self.normalize_method == 'none':
            return

        # 根据 indices 选择数据子集
        if indices is None:
            subset_images = self.images
            print("Warning: Calculating normalization stats on ALL data (Validation data leakage risk!)")
        else:
            subset_images = self.images[indices]
            print(f"Calculating normalization stats based on {len(indices)} training samples...")

        if self.normalize_method == 'z-score':
            # 分通道计算
            self.real_mean = subset_images[:, :, :, 0].mean()
            self.real_std = subset_images[:, :, :, 0].std()
            self.imag_mean = subset_images[:, :, :, 1].mean()
            self.imag_std = subset_images[:, :, :, 1].std()

            print(f"\n[Stats Updated] Z-Score Parameters:")
            print(f"  Real channel - mean: {self.real_mean:.4f}, std: {self.real_std:.4f}")
            print(f"  Imag channel - mean: {self.imag_mean:.4f}, std: {self.imag_std:.4f}")

        elif self.normalize_method == 'min-max':
            self.real_min = subset_images[:, :, :, 0].min()
            self.real_max = subset_images[:, :, :, 0].max()
            self.imag_min = subset_images[:, :, :, 1].min()
            self.imag_max = subset_images[:, :, :, 1].max()

            print(f"\n[Stats Updated] Min-Max Parameters:")
            print(f"  Real channel - min: {self.real_min:.4f}, max: {self.real_max:.4f}")
            print(f"  Imag channel - min: {self.imag_min:.4f}, max: {self.imag_max:.4f}")

        self.stats_calculated = True

    def _normalize(self, img):
        """对图像进行归一化"""
        # 防止分母为0
        epsilon = 1e-8

        if self.normalize_method == 'z-score':
            img[:, :, 0] = (img[:, :, 0] - self.real_mean) / (self.real_std + epsilon)
            img[:, :, 1] = (img[:, :, 1] - self.imag_mean) / (self.imag_std + epsilon)

        elif self.normalize_method == 'min-max':
            img[:, :, 0] = (img[:, :, 0] - self.real_min) / (self.real_max - self.real_min + epsilon)
            img[:, :, 1] = (img[:, :, 1] - self.imag_min) / (self.imag_max - self.imag_min + epsilon)

        return img

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # 获取图片和掩码
        img = self.images[idx].copy()
        mask = self.masks[idx].copy()

        # 应用归一化 (使用之前计算好的参数)
        img = self._normalize(img)

        # 数据增强
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # 转置
        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        # 转tensor
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        return img, mask, {'img_id': f'sample_{idx}'}

    def denormalize(self, img_tensor):
        """
        将归一化后的 Tensor (C, H, W) 还原回原始物理数值
        """
        # 确保统计数据已加载 (如果是在 test.py 里，可能需要手动设或者从文件读)
        # 临时方案：为了跑通，先假设你知道这些值。
        # 在实际训练后，你应该把 mean/std 保存到了 config 或者 txt 里。
        # 比如：
        # real_mean = 1.05, real_std = 0.2
        # imag_mean = 0.0, imag_std = 0.1

        # 这里先写逻辑：
        device = img_tensor.device
        img_denorm = img_tensor.clone()

        if self.normalize_method == 'z-score':
            # 实部 (Channel 0)
            img_denorm[0, :, :] = img_tensor[0, :, :] * self.real_std + self.real_mean
            # 虚部 (Channel 1)
            img_denorm[1, :, :] = img_tensor[1, :, :] * self.imag_std + self.imag_mean

        return img_denorm


class TransformSubset(torch.utils.data.Subset):
    """支持transform的Subset (无需修改)"""

    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, mask, meta = self.dataset[self.indices[idx]]

        if self.transform is not None:
            # 转回numpy做增强
            img_np = img.numpy().transpose(1, 2, 0)
            mask_np = mask.numpy().transpose(1, 2, 0)

            # 数据增强
            augmented = self.transform(image=img_np, mask=mask_np)
            img_np = augmented['image']
            mask_np = augmented['mask']

            # 转回tensor
            img = torch.from_numpy(img_np.transpose(2, 0, 1))
            mask = torch.from_numpy(mask_np.transpose(2, 0, 1))

        return img, mask, meta