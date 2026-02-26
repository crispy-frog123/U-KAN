import argparse
import os
import random
import torch
import scipy.io as sio
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from dataset_e import ComplexMatDataset
import archs
from skimage.metrics import structural_similarity as ssim_func
import json

import warnings

warnings.filterwarnings('ignore')

# ================= 配置区 =================
DATA_PATHS = {
    'real_img': 'data/chi0_all_real_mnist.mat',
    'imag_img': 'data/chi0_all_imag_mnist.mat',
    'real_label': 'data/chi_all_real_mnist.mat',
    'imag_label': 'data/chi_all_imag_mnist.mat'
}

INDICES_PATH = 'data/remaining_test_indices.mat'
EXP_DIR = 'outputs/2000_[128,160,256]_MSE+0.1SSIM_z-score'


# =========================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default=EXP_DIR, help='Experiment directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    return parser.parse_args()


def denormalize_input(img_tensor, stats):
    """手动反归一化 Input"""
    if stats is None:
        return img_tensor

    real = img_tensor[0]
    imag = img_tensor[1]

    if 'input_real' in stats:
        mean_r = stats['input_real']['mean']
        std_r = stats['input_real']['std']
        real = real * std_r + mean_r

    if 'input_imag' in stats:
        mean_i = stats['input_imag']['mean']
        std_i = stats['input_imag']['std']
        imag = imag * std_i + mean_i

    return np.stack([real, imag], axis=0)


def calculate_detailed_metrics(pred, target):
    """同时计算 SSIM 和 MSE"""
    range_real = target[0].max() - target[0].min() + 1e-6
    range_imag = target[1].max() - target[1].min() + 1e-6

    ssim_real = ssim_func(target[0], pred[0], data_range=range_real)
    ssim_imag = ssim_func(target[1], pred[1], data_range=range_imag)

    mse_real = np.mean((target[0] - pred[0]) ** 2)
    mse_imag = np.mean((target[1] - pred[1]) ** 2)

    return ssim_real, ssim_imag, mse_real, mse_imag


def create_mosaic(images_list, rows=4, cols=4, padding=2):
    """拼接马赛克大图"""
    count = len(images_list)
    if count == 0: return None

    h, w = images_list[0].shape

    mosaic_h = rows * h + (rows - 1) * padding
    mosaic_w = cols * w + (cols - 1) * padding
    mosaic = np.zeros((mosaic_h, mosaic_w))

    for idx, img in enumerate(images_list):
        if idx >= rows * cols: break

        r = idx // cols
        c = idx % cols

        start_y = r * (h + padding)
        start_x = c * (w + padding)

        mosaic[start_y: start_y + h, start_x: start_x + w] = img

    return mosaic


def save_metrics_to_txt(samples, save_dir, filename="metric.txt"):
    """保存指标"""
    path = os.path.join(save_dir, filename)
    print(f"Saving metrics to {path}...")

    with open(path, 'w') as f:
        f.write("========================================================\n")
        f.write("       Metrics for the 16 Visualized Samples (Fig.3)    \n")
        f.write("========================================================\n")
        f.write(f"{'ID':<10} | {'SSIM(R)':<10} | {'SSIM(I)':<10} | {'MSE(R)':<10} | {'MSE(I)':<10}\n")
        f.write("-" * 65 + "\n")

        avg_ssim_r = []
        avg_ssim_i = []
        avg_mse_r = []
        avg_mse_i = []

        for s in samples:
            f.write(
                f"{s['id']:<10} | {s['ssim_r']:.4f}     | {s['ssim_i']:.4f}     | {s['mse_r']:.4f}     | {s['mse_i']:.4f}\n")
            avg_ssim_r.append(s['ssim_r'])
            avg_ssim_i.append(s['ssim_i'])
            avg_mse_r.append(s['mse_r'])
            avg_mse_i.append(s['mse_i'])

        f.write("-" * 65 + "\n")
        f.write(
            f"{'AVERAGE':<10} | {np.mean(avg_ssim_r):.4f}     | {np.mean(avg_ssim_i):.4f}     | {np.mean(avg_mse_r):.4f}     | {np.mean(avg_mse_i):.4f}\n")
        f.write("========================================================\n")


def test_remaining_fig3_style():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Running in [Fig.3 Triple Column (Input-GT-Pred)] mode...")

    # 1. 加载数据
    if not os.path.exists(INDICES_PATH): raise FileNotFoundError(f"Missing indices: {INDICES_PATH}")
    mat_content = sio.loadmat(INDICES_PATH)
    test_indices = (mat_content['test_indices'][0] - 1).astype(np.int64)

    full_dataset = ComplexMatDataset(
        real_img_path=DATA_PATHS['real_img'],
        imag_img_path=DATA_PATHS['imag_img'],
        real_label_path=DATA_PATHS['real_label'],
        imag_label_path=DATA_PATHS['imag_label'],
        img_size=(64, 64),
        normalize_method='z-score'
    )

    # 加载 stats 用于反归一化 Input
    norm_stats = None
    stats_path = os.path.join(args.exp_dir, 'norm_stats.json')
    if os.path.exists(stats_path):
        full_dataset.load_stats(stats_path)
        with open(stats_path, 'r') as f:
            norm_stats = json.load(f)
        print("Loaded normalization stats for Input visualization.")

    test_subset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. 加载模型
    model = archs.__dict__['UKAN'](num_classes=2, input_channels=2).to(device)
    ckpt_path = os.path.join(args.exp_dir, 'model.pth')
    if not os.path.exists(ckpt_path): ckpt_path = os.path.join(args.exp_dir, 'best_model.pth')

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    # 3. 随机抽取 16 个样本
    random.seed(42)
    vis_global_indices = set(random.sample(range(len(test_subset)), 16))
    saved_samples = []

    print("Start Inference...")
    current_idx = 0
    with torch.no_grad():
        for i, (input_tensor, target, _) in enumerate(test_loader):
            batch_size = input_tensor.size(0)
            input_tensor = input_tensor.to(device)
            target = target.to(device)
            output = model(input_tensor)

            pred_np = output.cpu().numpy()
            target_np = target.cpu().numpy()
            input_np = input_tensor.cpu().numpy()  # 获取 Input

            for j in range(batch_size):
                global_id = current_idx + j
                if global_id in vis_global_indices:
                    # 计算指标 (Pred vs Target)
                    s_r, s_i, m_r, m_i = calculate_detailed_metrics(pred_np[j], target_np[j])

                    # 反归一化 Input 用于展示
                    input_phys = denormalize_input(input_np[j], norm_stats)

                    saved_samples.append({
                        'id': global_id,
                        'input': input_phys,  # 保存 Input
                        'target': target_np[j],
                        'pred': pred_np[j],
                        'ssim_r': s_r, 'ssim_i': s_i,
                        'mse_r': m_r, 'mse_i': m_i
                    })
            current_idx += batch_size
            if len(saved_samples) >= 16: break

    if len(saved_samples) == 16:
        save_metrics_to_txt(saved_samples, args.exp_dir)
        plot_paper_fig3_style_triple(saved_samples, args.exp_dir)
    else:
        print(f"Error: Need exactly 16 samples, but got {len(saved_samples)}")


def plot_paper_fig3_style_triple(samples, save_dir):
    print("Generating Triple-Column (Input | GT | Pred) Rotated Plots...")

    # 准备数据 (全部逆时针旋转 90度)
    input_real_list = [np.rot90(s['input'][0], k=1) for s in samples]
    gt_real_list = [np.rot90(s['target'][0], k=1) for s in samples]
    pred_real_list = [np.rot90(s['pred'][0], k=1) for s in samples]

    # 虚部先取绝对值，再旋转
    input_imag_list = [np.rot90(np.abs(s['input'][1]), k=1) for s in samples]
    gt_imag_list = [np.rot90(np.abs(s['target'][1]), k=1) for s in samples]
    pred_imag_list = [np.rot90(np.abs(s['pred'][1]), k=1) for s in samples]

    padding = 2
    # 拼接 Mosaic
    mosaic_in_r = create_mosaic(input_real_list, 4, 4, padding)
    mosaic_gt_r = create_mosaic(gt_real_list, 4, 4, padding)
    mosaic_pd_r = create_mosaic(pred_real_list, 4, 4, padding)

    mosaic_in_i = create_mosaic(input_imag_list, 4, 4, padding)
    mosaic_gt_i = create_mosaic(gt_imag_list, 4, 4, padding)
    mosaic_pd_i = create_mosaic(pred_imag_list, 4, 4, padding)

    # ================= 1. 实部图 (1行3列: Input | GT | Pred) =================
    # 画布加宽：24宽 (3x8)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # GT 和 Pred 必须统一量程以对比
    vmin_gp = min(mosaic_gt_r.min(), mosaic_pd_r.min())
    vmax_gp = max(mosaic_gt_r.max(), mosaic_pd_r.max())

    # Input 量程自适应 (因为 BP 图像数值可能和 Permittivity 不在一个量级)
    # 也可以手动指定 Input 也是 vmin_gp/vmax_gp，但这可能导致 Input 看起来全黑或全白
    # 建议：Input 用自适应，GT/Pred 用统一。

    # Left: Input
    im1 = axes[0].imshow(mosaic_in_r, cmap='jet')  # 自适应
    axes[0].set_title("Input (Real)", fontsize=20, fontweight='bold', pad=15)
    axes[0].axis('off')
    # Input 独享一个小的 Colorbar? 或者省略
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Middle: GT
    im2 = axes[1].imshow(mosaic_gt_r, cmap='jet', vmin=vmin_gp, vmax=vmax_gp)
    axes[1].set_title("Ground Truths (Real)", fontsize=20, fontweight='bold', pad=15)
    axes[1].axis('off')

    # Right: Pred
    im3 = axes[2].imshow(mosaic_pd_r, cmap='jet', vmin=vmin_gp, vmax=vmax_gp)
    axes[2].set_title("Pred (Real)", fontsize=20, fontweight='bold', pad=15)
    axes[2].axis('off')

    # GT 和 Pred 共用一个大的 Colorbar
    cbar = fig.colorbar(im3, ax=[axes[1], axes[2]], fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
    for l in cbar.ax.yaxis.get_ticklabels(): l.set_weight('bold')

    save_path_r = os.path.join(save_dir, 'fig3_triple_Real.png')
    plt.savefig(save_path_r, dpi=300, bbox_inches='tight')
    plt.close()

    # ================= 2. 虚部图 (|Imag|) =================
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    vmax_gp_i = max(mosaic_gt_i.max(), mosaic_pd_i.max())

    # Left: Input
    im1 = axes[0].imshow(mosaic_in_i, cmap='jet')  # 自适应, Abs 最小为0
    axes[0].set_title("Input (|Imag|)", fontsize=20, fontweight='bold', pad=15)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Middle: GT
    im2 = axes[1].imshow(mosaic_gt_i, cmap='jet', vmin=0, vmax=vmax_gp_i)
    axes[1].set_title("Ground Truths (|Imag|)", fontsize=20, fontweight='bold', pad=15)
    axes[1].axis('off')

    # Right: Pred
    im3 = axes[2].imshow(mosaic_pd_i, cmap='jet', vmin=0, vmax=vmax_gp_i)
    axes[2].set_title("Pred (|Imag|)", fontsize=20, fontweight='bold', pad=15)
    axes[2].axis('off')

    cbar = fig.colorbar(im3, ax=[axes[1], axes[2]], fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
    for l in cbar.ax.yaxis.get_ticklabels(): l.set_weight('bold')

    save_path_i = os.path.join(save_dir, 'fig3_triple_Imag.png')
    plt.savefig(save_path_i, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Results saved:\n  - {save_path_r}\n  - {save_path_i}\n  - metric.txt")


if __name__ == '__main__':
    test_remaining_fig3_style()