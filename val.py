import sys

sys.path.insert(0, r'E:\U-KAN')

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import archs
from dataset_e import ComplexMatDataset
from albumentations import Resize
from albumentations.core.composition import Compose

# ========== 直接在这里改路径 ==========
MODEL_PATH = r'outputs\test2_UKAN_woDS\model.pth'
CONFIG_PATH = r'outputs\test2_UKAN_woDS\config.yml'
NUM_SAMPLES = 3
SAVE_PATH = 'predictions.png'
# ====================================

print("=" * 60)
print("生成预测对比图")
print("=" * 60)

# 加载配置
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}\n")

# 加载数据
print("加载数据集...")


def build_path(base_dir, file_path):
    if os.path.isabs(file_path):
        return file_path
    else:
        return os.path.normpath(os.path.join(base_dir, file_path))


real_img_path = build_path(config['data_dir'], config['real_img_file'])
imag_img_path = build_path(config['data_dir'], config['imag_img_file'])
real_mask_path = build_path(config['data_dir'], config['real_mask_file'])
imag_mask_path = build_path(config['data_dir'], config['imag_mask_file'])

val_transform = Compose([Resize(config['input_h'], config['input_w'])])

dataset = ComplexMatDataset(
    real_img_path=real_img_path,
    imag_img_path=imag_img_path,
    real_mask_path=real_mask_path,
    imag_mask_path=imag_mask_path,
    img_size=(config['original_img_size'], config['original_img_size']),
    transform=val_transform,
    normalize_method='z-score'
)

print(f"✓ 数据集大小: {len(dataset)}\n")


# 创建模型
# 创建模型
print("\n创建模型...")
model = archs.__dict__[config['arch']](
    config['num_classes'],
    config['input_channels'],
    config['deep_supervision'],
    embed_dims=config['input_list'],
    no_kan=config['no_kan']
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("✓ 模型加载成功\n")

# 预测
print(f"生成 {NUM_SAMPLES} 个样本的预测...")

fig, axes = plt.subplots(NUM_SAMPLES, 6, figsize=(18, 3 * NUM_SAMPLES))
if NUM_SAMPLES == 1:
    axes = axes.reshape(1, -1)

with torch.no_grad():
    for i in range(NUM_SAMPLES):
        # 随机选择样本
        idx = np.random.randint(0, len(dataset))
        img, mask_gt, _ = dataset[idx]

        # 预测
        img_input = img.unsqueeze(0).to(device)

        if config['deep_supervision']:
            outputs = model(img_input)
            mask_pred = outputs[-1]
        else:
            mask_pred = model(img_input)

        # 转numpy
        img_np = img.cpu().numpy()
        mask_gt_np = mask_gt.cpu().numpy()
        mask_pred_np = mask_pred[0].cpu().numpy()

        # 实部
        axes[i, 0].imshow(img_np[0], cmap='seismic')
        axes[i, 0].set_title(f'Sample {idx}\nInput Real', fontsize=9)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_gt_np[0], cmap='seismic')
        axes[i, 1].set_title('Mask Real', fontsize=9)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(mask_pred_np[0], cmap='seismic')
        axes[i, 2].set_title('Pred Real', fontsize=9)
        axes[i, 2].axis('off')

        # 虚部
        axes[i, 3].imshow(img_np[1], cmap='seismic')
        axes[i, 3].set_title('Input Imag', fontsize=9)
        axes[i, 3].axis('off')

        axes[i, 4].imshow(mask_gt_np[1], cmap='seismic')
        axes[i, 4].set_title('Mask Imag', fontsize=9)
        axes[i, 4].axis('off')

        axes[i, 5].imshow(mask_pred_np[1], cmap='seismic')
        axes[i, 5].set_title('Pred Imag', fontsize=9)
        axes[i, 5].axis('off')

        # 计算MSE
        mse_real = np.mean((mask_pred_np[0] - mask_gt_np[0]) ** 2)
        mse_imag = np.mean((mask_pred_np[1] - mask_gt_np[1]) ** 2)

        print(f"  样本 {i + 1}: MSE_real={mse_real:.6f}, MSE_imag={mse_imag:.6f}")

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')

print(f"\n{'=' * 60}")
print(f"✓ 已保存到: {SAVE_PATH}")
print(f"{'=' * 60}")
