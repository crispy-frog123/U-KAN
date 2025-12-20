import sys

sys.path.insert(0, r'E:\U-KAN')

import torch
import torch.nn as nn
import archs
from dataset_e import ComplexMatDataset
import os

print("=" * 60)
print("训练诊断")
print("=" * 60)

device = torch.device('cuda')

# 1. 加载数据集
print("\n[1] 加载数据集...")
base_dir = r'E:\U-KAN\inputs'

dataset = ComplexMatDataset(
    real_img_path=os.path.join(base_dir, 'input/chi_all_real_64.mat'),
    imag_img_path=os.path.join(base_dir, 'input/chi_all_imag_64.mat'),
    real_mask_path=os.path.join(base_dir, 'label/chi0_all_real_64.mat'),
    imag_mask_path=os.path.join(base_dir, 'label/chi0_all_imag_64.mat'),
    img_size=(64, 64),
    transform=None,
    normalize_method='z-score'
)

# 获取一个样本
img, mask, meta = dataset[0]

print(f"\n图像信息:")
print(f"  Shape: {img.shape}")
print(f"  Type: {img.dtype}")
print(f"  Range: [{img.min():.4f}, {img.max():.4f}]")
print(f"  Mean: {img.mean():.4f}, Std: {img.std():.4f}")

print(f"\nMask信息:")
print(f"  Shape: {mask.shape}")
print(f"  Type: {mask.dtype}")
print(f"  Range: [{mask.min():.4f}, {mask.max():.4f}]")
print(f"  Unique values: {torch.unique(mask)}")
print(f"  Non-zero ratio: {(mask > 0).float().mean():.4f}")

# 2. 创建模型
print("\n[2] 创建模型...")
model = archs.UKAN(
    num_classes=2,
    input_channels=2,
    deep_supervision=False
).to(device)

print(f"✓ 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# 3. 测试前向传播
print("\n[3] 测试前向传播...")
model.eval()

input = img.unsqueeze(0).to(device)  # (1, 2, 64, 64)
target = mask.unsqueeze(0).to(device)

print(f"  Input shape: {input.shape}")
print(f"  Target shape: {target.shape}")

with torch.no_grad():
    output = model(input)

print(f"  Output shape: {output.shape}")
print(f"  Output range (raw): [{output.min():.4f}, {output.max():.4f}]")

# 应用sigmoid
output_prob = torch.sigmoid(output)
print(f"  Output range (sigmoid): [{output_prob.min():.4f}, {output_prob.max():.4f}]")

# 4. 测试损失函数
print("\n[4] 测试损失函数...")
criterion = nn.BCEWithLogitsLoss().to(device)

loss = criterion(output, target)
print(f"  Loss: {loss.item():.4f}")

# 5. 测试IoU计算
print("\n[5] 测试IoU计算...")
from metrics import iou_score

iou, dice, detail = iou_score(output, target)
print(f"  IoU: {iou:.6f}")
print(f"  Dice: {dice:.6f}")

# 详细分析
print(f"\n详细分析:")
output_binary = (torch.sigmoid(output) > 0.5).float()
target_binary = (target > 0.5).float()

for c in range(2):
    pred = output_binary[0, c]
    gt = target_binary[0, c]

    intersection = (pred * gt).sum().item()
    union = pred.sum().item() + gt.sum().item() - intersection

    print(f"\n  通道 {c}:")
    print(f"    Prediction pixels: {pred.sum().item()}")
    print(f"    Ground truth pixels: {gt.sum().item()}")
    print(f"    Intersection: {intersection}")
    print(f"    Union: {union}")

    if union > 0:
        channel_iou = intersection / union
        print(f"    IoU: {channel_iou:.6f}")
    else:
        print(f"    IoU: N/A (both empty)")

# 6. 可视化
print("\n[6] 保存可视化...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for c in range(2):
    # 输入
    axes[c, 0].imshow(input[0, c].cpu().numpy(), cmap='gray')
    axes[c, 0].set_title(f'Input Ch{c}')
    axes[c, 0].axis('off')

    # 预测
    axes[c, 1].imshow(output_prob[0, c].cpu().numpy(), cmap='gray')
    axes[c, 1].set_title(f'Prediction Ch{c}')
    axes[c, 1].axis('off')

    # 真值
    axes[c, 2].imshow(target[0, c].cpu().numpy(), cmap='gray')
    axes[c, 2].set_title(f'Ground Truth Ch{c}')
    axes[c, 2].axis('off')

plt.tight_layout()
plt.savefig('debug_output.png', dpi=150)
print("✓ 已保存到 debug_output.png")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
