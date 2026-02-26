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

import warnings

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Experiment directory (e.g., outputs/DualStream_Weight...)')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='Model filename to load (default: best_model.pth)')
    parser.add_argument('--save_dir', type=str, default='test_results_flexible',
                        help='Directory to save visualization results')

    # [新增可选项]
    parser.add_argument('--new_data_dir', type=str, default=None,
                        help='[Optional] Path to a folder containing NEW dataset .mat files.')

    return parser.parse_args()


def calculate_metrics_detailed(pred, target):
    """分别计算实部和虚部的 MSE 和 SSIM"""
    # MSE
    mse_real = np.mean((pred[0] - target[0]) ** 2)
    mse_imag = np.mean((pred[1] - target[1]) ** 2)

    # SSIM
    range_real = target[0].max() - target[0].min()
    if range_real == 0: range_real = 1e-6
    ssim_real = ssim_func(target[0], pred[0], data_range=range_real)

    range_imag = target[1].max() - target[1].min()
    if range_imag == 0: range_imag = 1e-6
    ssim_imag = ssim_func(target[1], pred[1], data_range=range_imag)

    return mse_real, mse_imag, ssim_real, ssim_imag


def main():
    args = parse_args()

    # 1. 加载 Config
    config_path = os.path.join(args.exp_dir, 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 60}")
    print(f" Evaluation: {config.get('name', 'Unknown')}")
    print(f" Model Arch: {config['arch']}")
    print(f"{'=' * 60}\n")

    # ================= 2. 数据集路径选择逻辑 =================
    print("Configuring Dataset Paths...")

    val_transform = Compose([Resize(config['input_h'], config['input_w'])])

    def build_path(base, fname):
        if os.path.isabs(fname): return fname
        return os.path.normpath(os.path.join(base, fname))

    if args.new_data_dir is not None:
        # --- 分支 A: 使用新路径 (New Data Mode) ---
        print(f"✅ MODE: Using NEW Dataset from: {args.new_data_dir}")
        print("   (Assuming filenames are: chi0_all_real_64.mat, etc.)")

        path_real_img = os.path.join(args.new_data_dir, 'chi0_all_real_64.mat')
        path_imag_img = os.path.join(args.new_data_dir, 'chi0_all_imag_64.mat')
        path_real_lbl = os.path.join(args.new_data_dir, 'chi_all_real_64.mat')
        path_imag_lbl = os.path.join(args.new_data_dir, 'chi_all_imag_64.mat')

        use_subset_split = False
    else:
        # --- 分支 B: 使用 Config 默认路径 (Default Mode) ---
        print("🔹 MODE: Using CONFIG Dataset (Validation Split)")

        path_real_img = build_path(config['data_dir'], config['real_img_file'])
        path_imag_img = build_path(config['data_dir'], config['imag_img_file'])
        path_real_lbl = build_path(config['data_dir'], config['real_label_file'])
        path_imag_lbl = build_path(config['data_dir'], config['imag_label_file'])

        use_subset_split = True

        # =========================================================

    # 3. 构建数据集
    if not os.path.exists(path_real_img):
        raise FileNotFoundError(f"Dataset file not found: {path_real_img}")

    full_dataset = ComplexMatDataset(
        real_img_path=path_real_img,
        imag_img_path=path_imag_img,
        real_label_path=path_real_lbl,
        imag_label_path=path_imag_lbl,
        img_size=(config['original_img_size'], config['original_img_size']),
        transform=None,
        normalize_method='z-score'
    )

    # 4. 加载归一化参数
    stats_path = os.path.join(args.exp_dir, 'norm_stats.json')
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}")
        full_dataset.load_stats(stats_path)
    else:
        print("⚠️ Warning: norm_stats.json not found! Testing might be inaccurate.")

    # 5. 构建 DataLoader
    if use_subset_split:
        indices = list(range(len(full_dataset)))
        np.random.seed(config['dataseed'])
        np.random.shuffle(indices)

        test_split = int(np.floor(0.1 * len(full_dataset)))
        test_indices = indices[:test_split]

        final_dataset = TransformSubset(full_dataset, test_indices, val_transform)
        print(f"Dataset Split: 10% Subset (Config Mode). Size: {len(final_dataset)}")
    else:
        all_indices = list(range(len(full_dataset)))
        final_dataset = TransformSubset(full_dataset, all_indices, val_transform)
        print(f"Dataset Split: FULL Dataset (New Data Mode). Size: {len(final_dataset)}")

    test_loader = DataLoader(final_dataset, batch_size=1, shuffle=False)

    # 6. 加载模型
    print(f"Loading Model: {config['arch']}...")
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config.get('no_kan', False)
    ).to(device)

    model_path = os.path.join(args.exp_dir, args.checkpoint)
    if not os.path.exists(model_path):
        fallback_path = os.path.join(args.exp_dir, 'model.pth')
        if os.path.exists(fallback_path):
            print(f"⚠️ {args.checkpoint} not found, using model.pth")
            model_path = fallback_path
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.exp_dir}")

    print(f"Loading weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 7. 测试循环
    meters = {
        'mse_real': AverageMeter(), 'mse_imag': AverageMeter(),
        'ssim_real': AverageMeter(), 'ssim_imag': AverageMeter()
    }

    sub_folder = "results_new_data_abs" if args.new_data_dir else "results_config_data_abs"
    save_full_path = os.path.join(args.exp_dir, sub_folder)
    os.makedirs(save_full_path, exist_ok=True)

    vis_count = min(10, len(final_dataset))
    random.seed(config['dataseed'])
    vis_indices = set(random.sample(range(len(final_dataset)), vis_count))
    saved_samples = []

    print("Running Inference...")
    with torch.no_grad():
        for i, (input_tensor, target, _) in enumerate(tqdm(test_loader)):
            input_tensor = input_tensor.to(device)
            target = target.to(device)

            outputs = model(input_tensor)
            if isinstance(outputs, list):
                pred = outputs[-1]
            else:
                pred = outputs

            pred_np = pred.cpu().numpy().squeeze(0)
            target_np = target.cpu().numpy().squeeze(0)

            mse_r, mse_i, ssim_r, ssim_i = calculate_metrics_detailed(pred_np, target_np)

            meters['mse_real'].update(mse_r)
            meters['mse_imag'].update(mse_i)
            meters['ssim_real'].update(ssim_r)
            meters['ssim_imag'].update(ssim_i)

            if i in vis_indices:
                saved_samples.append({
                    'id': i,
                    'pred': pred_np,
                    'target': target_np,
                    'ssim_r': ssim_r,
                    'ssim_i': ssim_i
                })

    # 8. 输出与保存
    print(f"\n{'=' * 20} Final Results {'=' * 20}")
    print(f"Real Part -> MSE: {meters['mse_real'].avg:.6f} | SSIM: {meters['ssim_real'].avg:.4f}")
    print(f"Imag Part -> MSE: {meters['mse_imag'].avg:.6f} | SSIM: {meters['ssim_imag'].avg:.4f}")
    print(f"Overall   -> Avg SSIM: {(meters['ssim_real'].avg + meters['ssim_imag'].avg) / 2:.4f}")
    print("=" * 55)

    txt_path = os.path.join(save_full_path, 'metrics.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Evaluation Report\n")
        f.write(f"=================\n")
        f.write(f"Data Mode: {'NEW DATA' if args.new_data_dir else 'CONFIG DEFAULT'}\n")
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Count: {len(final_dataset)}\n\n")
        f.write(f"Real SSIM: {meters['ssim_real'].avg:.6f}\n")
        f.write(f"Imag SSIM: {meters['ssim_imag'].avg:.6f}\n")
        f.write(f"Avg SSIM:  {(meters['ssim_real'].avg + meters['ssim_imag'].avg) / 2:.6f}\n")

    print("Plotting samples...")
    plot_samples_abs_bold(saved_samples, save_full_path)


def plot_samples_abs_bold(samples, save_dir):
    """
    绘制对比图：
    1. 虚部 (Imag) 强制取绝对值显示。
    2. Colorbar 字体加大、加粗。
    """
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 4.0 * num_samples))  # 稍微加宽画布

    cols = ['GT Real', 'Pred Real', 'GT |Imag|', 'Pred |Imag|']
    if num_samples == 1: axes = [axes]

    for idx, sample in enumerate(samples):
        # 1. 获取数据
        gt_r = sample['target'][0]
        pd_r = sample['pred'][0]

        # --- 虚部取绝对值 ---
        gt_i = np.abs(sample['target'][1])
        pd_i = np.abs(sample['pred'][1])
        # -------------------

        # 2. 计算动态量程
        vmin_r = min(gt_r.min(), pd_r.min())
        vmax_r = max(gt_r.max(), pd_r.max())

        # 虚部绝对值最小是0
        vmin_i = 0
        vmax_i = max(gt_i.max(), pd_i.max())

        row_axes = axes[idx]

        # 设置列标题 (加大字体)
        if idx == 0:
            for ax, col in zip(row_axes, cols):
                ax.set_title(col, fontsize=16, fontweight='bold', pad=15)

        # 定义画图辅助函数
        def plot_subplot(ax, img, vmin, vmax, cmap='jet'):
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            # --- Colorbar 加大加粗 ---
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=14)
            for l in cbar.ax.yaxis.get_ticklabels():
                l.set_weight('bold')
            # ------------------------
            ax.set_xticks([])
            ax.set_yticks([])
            return ax

        # GT Real
        ax = plot_subplot(row_axes[0], gt_r, vmin_r, vmax_r)
        ax.set_ylabel(f"ID: {sample['id']}", fontsize=14, fontweight='bold', labelpad=10)

        # Pred Real
        ax = plot_subplot(row_axes[1], pd_r, vmin_r, vmax_r)
        ax.text(0.5, -0.1, f"SSIM: {sample['ssim_r']:.2f}",
                transform=ax.transAxes, ha='center', fontsize=14, color='red', fontweight='bold')

        # GT Imag (Abs)
        plot_subplot(row_axes[2], gt_i, vmin_i, vmax_i)

        # Pred Imag (Abs)
        ax = plot_subplot(row_axes[3], pd_i, vmin_i, vmax_i)
        ax.text(0.5, -0.1, f"SSIM: {sample['ssim_i']:.2f}",
                transform=ax.transAxes, ha='center', fontsize=14, color='blue', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'comparison_vis_abs_bold.png')
    plt.savefig(save_path, dpi=300)
    print(f"✓ Visualization saved to {save_path}")


if __name__ == '__main__':
    main()