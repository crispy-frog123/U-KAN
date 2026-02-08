import argparse
import os
import random
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations.core.composition import Compose
from albumentations import Resize

# 导入高级评价指标库
from skimage.metrics import structural_similarity as ssim_func

# 导入项目模块
import archs
from dataset_e import ComplexMatDataset, TransformSubset
from utils import AverageMeter

# 防止无关警告
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    # 指定 output 文件夹路径 (与你提供的 test.py 格式一致)
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Experiment directory (e.g., outputs/2000_[...])')
    parser.add_argument('--checkpoint', type=str, default='model.pth',
                        help='Model filename to load (default: model.pth)')
    parser.add_argument('--save_dir', type=str, default='test_results_deepnis',
                        help='Directory to save visualization results')
    return parser.parse_args()


def calculate_deepnis_metrics(pred, target):
    """
    计算 DeepNIS 论文核心指标: MSE 和 SSIM
    pred, target: (C, H, W) numpy arrays
    """
    # 1. MSE (均方误差) - DeepNIS 论文中的 "Euclidean cost"
    mse_val = np.mean((pred - target) ** 2)

    # 2. SSIM (结构相似性)
    # 分别计算实部和虚部，然后取平均
    # data_range 动态计算，确保准确
    range_real = target[0].max() - target[0].min()
    range_imag = target[1].max() - target[1].min()

    # 处理纯色背景 range=0 的情况，防止除零
    if range_real == 0: range_real = 1.0
    if range_imag == 0: range_imag = 1.0

    ssim_real = ssim_func(target[0], pred[0], data_range=range_real)
    ssim_imag = ssim_func(target[1], pred[1], data_range=range_imag)

    avg_ssim = (ssim_real + ssim_imag) / 2.0

    return mse_val, avg_ssim


def main():
    args = parse_args()

    # 1. 加载 Config (配置部分完全按照你的 test.py 格式)
    config_path = os.path.join(args.exp_dir, 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 60}")
    print(f" DeepNIS Evaluation Mode")
    print(f" Experiment: {config.get('name', 'Unknown')}")
    print(f" Device: {device}")
    print(f"{'=' * 60}\n")

    # 2. 准备数据集
    print("Loading Dataset...")
    val_transform = Compose([Resize(config['input_h'], config['input_w'])])

    def build_path(base_dir, file_path):
        if os.path.isabs(file_path): return file_path
        return os.path.normpath(os.path.join(base_dir, file_path))

    # [自动适配] 读取训练时的归一化方法，如果没有记录则默认 'none'
    norm_method = config.get('normalize_method', 'none')  # <--- 关键修改，自动适配 none
    print(f"Normalization Method: {norm_method}")

    full_dataset = ComplexMatDataset(
        real_img_path=build_path(config['data_dir'], config['real_img_file']),
        imag_img_path=build_path(config['data_dir'], config['imag_img_file']),
        real_label_path=build_path(config['data_dir'], config['real_label_file']),
        imag_label_path=build_path(config['data_dir'], config['imag_label_file']),
        img_size=(config['original_img_size'], config['original_img_size']),
        transform=None,
        normalize_method=norm_method
    )

    # === 复现 8:1:1 划分逻辑 (与 train.py 完全一致) ===
    indices = list(range(len(full_dataset)))
    np.random.seed(config['dataseed'])
    np.random.shuffle(indices)

    test_split = int(np.floor(0.1 * len(full_dataset)))
    test_indices = indices[:test_split]  # 取前 10% 作为测试集

    test_dataset = TransformSubset(full_dataset, test_indices, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Test Set Size: {len(test_dataset)} samples (Independent 10%)")

    # 3. 加载模型
    print("Loading Model...")
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config.get('no_kan', False)  # 兼容旧 config
    ).to(device)

    model_path = os.path.join(args.exp_dir, args.checkpoint)
    print(f"Loading weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. 随机抽取 10 个样本用于画图
    random.seed(config['dataseed'])  # 保证可复现
    num_vis = 10
    vis_indices = set(random.sample(range(len(test_dataset)), min(num_vis, len(test_dataset))))
    saved_samples = []

    # 5. 开始测试循环
    metrics = {
        'mse': AverageMeter(),
        'ssim': AverageMeter()
    }

    print("\nRunning DeepNIS Metric Evaluation...")
    with torch.no_grad():
        for i, (input_tensor, target, _) in enumerate(tqdm(test_loader)):
            input_tensor = input_tensor.to(device)
            target = target.to(device)

            # 推理
            if config['deep_supervision']:
                outputs = model(input_tensor)
                pred = outputs[-1]
            else:
                pred = model(input_tensor)

            # [重要] 如果训练时用了无归一化+ReLU，这里理论上不需要再处理
            # 但为了安全，如果发现预测值有微小负数，可以在此截断
            # pred = torch.relu(pred)

            # 转为 numpy (C, H, W)
            pred_np = pred.cpu().numpy().squeeze(0)
            target_np = target.cpu().numpy().squeeze(0)

            # 计算指标
            mse_val, ssim_val = calculate_deepnis_metrics(pred_np, target_np)

            metrics['mse'].update(mse_val)
            metrics['ssim'].update(ssim_val)

            # 如果是被选中的样本，保存数据用于后续画图
            if i in vis_indices:
                saved_samples.append({
                    'id': i,
                    'pred': pred_np,
                    'target': target_np,
                    'ssim': ssim_val,
                    'mse': mse_val
                })

    # 6. 打印最终结果
    print(f"\n{'=' * 60}")
    print(f"FINAL TEST RESULTS (DeepNIS Benchmark)")
    print(f"{'=' * 60}")
    print(f"Avg MSE:  {metrics['mse'].avg:.6f}  (Target: < 0.05)")
    print(f"Avg SSIM: {metrics['ssim'].avg:.4f}  (Target: > 0.85)")
    print(f"{'=' * 60}\n")

    # 7. 绘制随机抽取的 10 张图
    save_full_path = os.path.join(args.exp_dir, args.save_dir)
    os.makedirs(save_full_path, exist_ok=True)

    print(f"Saving visualization of {len(saved_samples)} random samples to {save_full_path}...")

    # 创建大图：行数=样本数，列数=4 (GT实, Pred实, GT虚, Pred虚)
    fig, axes = plt.subplots(len(saved_samples), 4, figsize=(12, 3 * len(saved_samples)))

    # 设置列标题
    cols = ['GT Real', 'Pred Real', 'GT Imag', 'Pred Imag']
    if len(saved_samples) > 1:
        for ax, col in zip(axes[0], cols):
            ax.set_title(col, fontsize=14, fontweight='bold')
    else:
        # 如果只有1个样本，axes是一维数组
        for ax, col in zip(axes, cols):
            ax.set_title(col, fontsize=14, fontweight='bold')

    # 排序方便查看 (按ID排序)
    saved_samples.sort(key=lambda x: x['id'])

    for idx, sample in enumerate(saved_samples):
        # 获取当前的 axes 行
        curr_axes = axes[idx] if len(saved_samples) > 1 else axes

        target = sample['target']
        pred = sample['pred']

        # Real Part
        im0 = curr_axes[0].imshow(target[0], cmap='jet')
        curr_axes[0].axis('off')
        plt.colorbar(im0, ax=curr_axes[0], fraction=0.046, pad=0.04)
        curr_axes[0].text(-5, 32, f"ID: {sample['id']}", fontsize=12, rotation=90, va='center')

        im1 = curr_axes[1].imshow(pred[0], cmap='jet')
        curr_axes[1].axis('off')
        curr_axes[1].text(0, -2, f"SSIM: {sample['ssim']:.2f}", fontsize=10, color='black')
        plt.colorbar(im1, ax=curr_axes[1], fraction=0.046, pad=0.04)

        # Imag Part
        im2 = curr_axes[2].imshow(target[1], cmap='jet')
        curr_axes[2].axis('off')
        plt.colorbar(im2, ax=curr_axes[2], fraction=0.046, pad=0.04)

        im3 = curr_axes[3].imshow(pred[1], cmap='jet')
        curr_axes[3].axis('off')
        plt.colorbar(im3, ax=curr_axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plot_file = os.path.join(save_full_path, 'random_samples_vis.png')
    plt.savefig(plot_file, dpi=300)
    print(f"✓ Visualization saved to {plot_file}")

    # 保存指标到文本
    txt_file = os.path.join(save_full_path, 'metrics.txt')
    with open(txt_file, 'w') as f:
        f.write("Metrics\n")
        f.write("=" * 30 + "\n")
        f.write(f"Avg MSE:  {metrics['mse'].avg:.6f}\n")
        f.write(f"Avg SSIM: {metrics['ssim'].avg:.4f}\n")

    print("✓ Metrics saved to text file.")


if __name__ == '__main__':
    main()