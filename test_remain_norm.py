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
import yaml

import warnings

warnings.filterwarnings('ignore')

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
    parser.add_argument('--checkpoint', type=str, default='model_best_mse.pth', help='Checkpoint filename')
    parser.add_argument('--indices_path', type=str, default=INDICES_PATH, help='Remaining-test indices .mat path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    return parser.parse_args()


def denormalize_input(img_tensor, stats):
    """Apply manual de-normalization to the input sample."""
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
    """Compute SSIM and MSE metrics for real and imaginary channels."""
    range_real = target[0].max() - target[0].min() + 1e-6
    range_imag = target[1].max() - target[1].min() + 1e-6

    ssim_real = ssim_func(target[0], pred[0], data_range=range_real)
    ssim_imag = ssim_func(target[1], pred[1], data_range=range_imag)

    mse_real = np.mean((target[0] - pred[0]) ** 2)
    mse_imag = np.mean((target[1] - pred[1]) ** 2)

    return ssim_real, ssim_imag, mse_real, mse_imag


def create_mosaic(images_list, rows=4, cols=4, padding=2):
    """Assemble multiple images into a tiled mosaic."""
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
    """Save per-sample metrics to a text report."""
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


def save_overall_metrics_to_txt(metrics, save_dir, filename="overall_metrics.txt"):
    path = os.path.join(save_dir, filename)
    print(f"Saving overall metrics to {path}...")
    with open(path, 'w') as f:
        f.write("Overall Metrics on Remaining Test Set\n")
        f.write("====================================\n")
        f.write(f"Count: {metrics['count']}\n")
        f.write(f"Real  -> MSE: {metrics['mse_real']:.6f} | SSIM: {metrics['ssim_real']:.4f}\n")
        f.write(f"Imag  -> MSE: {metrics['mse_imag']:.6f} | SSIM: {metrics['ssim_imag']:.4f}\n")
        f.write(f"Avg   -> MSE: {metrics['mse_avg']:.6f} | SSIM: {metrics['ssim_avg']:.4f}\n")


def test_remaining_fig3_style():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Running in [Fig.3 Triple Column (Input-GT-Pred)] mode...")

    if not os.path.exists(args.indices_path):
        raise FileNotFoundError(f"Missing indices: {args.indices_path}")
    mat_content = sio.loadmat(args.indices_path)
    test_indices = (mat_content['test_indices'][0] - 1).astype(np.int64)

    cfg_path = os.path.join(args.exp_dir, 'config.yml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    full_dataset = ComplexMatDataset(
        real_img_path=DATA_PATHS['real_img'],
        imag_img_path=DATA_PATHS['imag_img'],
        real_label_path=DATA_PATHS['real_label'],
        imag_label_path=DATA_PATHS['imag_label'],
        img_size=(config.get('original_img_size', 64), config.get('original_img_size', 64)),
        normalize_method='z-score'
    )

    norm_stats = None
    stats_path = os.path.join(args.exp_dir, 'norm_stats.json')
    if os.path.exists(stats_path):
        full_dataset.load_stats(stats_path)
        with open(stats_path, 'r') as f:
            norm_stats = json.load(f)
        print("Loaded normalization stats for Input visualization.")

    test_subset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    save_dir = os.path.join(args.exp_dir, 'remain_16k')
    os.makedirs(save_dir, exist_ok=True)

    model = archs.__dict__[config['arch']](
        num_classes=config.get('num_classes', 2),
        input_channels=config.get('input_channels', 2),
        deep_supervision=config.get('deep_supervision', False),
        embed_dims=config.get('input_list', [128, 160, 256]),
        no_kan=config.get('no_kan', False),
        use_edge_residual_refine=config.get('use_edge_residual_refine', False),
        use_edge_multiscale=config.get('use_edge_multiscale', False),
        use_edge_sparse_focus=config.get('use_edge_sparse_focus', False),
        edge_focus_tau=config.get('edge_focus_tau', 0.30),
        edge_focus_gamma=config.get('edge_focus_gamma', 10.0),
        use_edge_center_boost=config.get('use_edge_center_boost', False),
        edge_center_boost=config.get('edge_center_boost', 0.6),
        edge_center_sigma=config.get('edge_center_sigma', 0.45),
        use_edge_hotspot_boost=config.get('use_edge_hotspot_boost', False),
        edge_hotspot_gain=config.get('edge_hotspot_gain', 0.8),
        edge_hotspot_mu_x=config.get('edge_hotspot_mu_x', -0.08),
        edge_hotspot_mu_y=config.get('edge_hotspot_mu_y', 0.12),
        edge_hotspot_sigma_x=config.get('edge_hotspot_sigma_x', 0.45),
        edge_hotspot_sigma_y=config.get('edge_hotspot_sigma_y', 0.20),
        edge_refine_scale=config.get('edge_refine_scale', 0.2),
        edge_refine_mid=config.get('edge_refine_mid', 48),
        use_detail_skip_refine=config.get('use_detail_skip_refine', False),
        detail_refine_scale=config.get('detail_refine_scale', 0.1),
        detail_refine_mid=config.get('detail_refine_mid', 64),
        use_fourier_refine=config.get('use_fourier_refine', False),
        fourier_use_fft=config.get('fourier_use_fft', True),
        fourier_refine_scale=config.get('fourier_refine_scale', 0.1),
        fourier_refine_mid=config.get('fourier_refine_mid', 32)
    ).to(device)

    ckpt_path = os.path.join(args.exp_dir, args.checkpoint)
    if not os.path.exists(ckpt_path):
        for cand in ['model.pth', 'model_best_mse.pth', 'model_best_ssim.pth', 'best_model.pth']:
            cand_path = os.path.join(args.exp_dir, cand)
            if os.path.exists(cand_path):
                ckpt_path = cand_path
                break
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found in {args.exp_dir}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        print(f"[WARN] Strict load failed, fallback to strict=False. Reason: {e}")
        model.load_state_dict(state, strict=False)
    model.eval()

    random.seed(42)
    vis_global_indices = set(random.sample(range(len(test_subset)), 16))
    saved_samples = []
    sum_ssim_r = 0.0
    sum_ssim_i = 0.0
    sum_mse_r = 0.0
    sum_mse_i = 0.0
    total_count = 0

    print("Start Inference...")
    current_idx = 0
    with torch.no_grad():
        for i, (input_tensor, target, _) in enumerate(test_loader):
            batch_size = input_tensor.size(0)
            input_tensor = input_tensor.to(device)
            target = target.to(device)
            output = model(input_tensor)
            if isinstance(output, list):
                output = output[-1]

            pred_np = output.cpu().numpy()
            target_np = target.cpu().numpy()
            input_np = input_tensor.cpu().numpy()  # Standardized technical note.

            for j in range(batch_size):
                global_id = current_idx + j
                s_r, s_i, m_r, m_i = calculate_detailed_metrics(pred_np[j], target_np[j])
                sum_ssim_r += s_r
                sum_ssim_i += s_i
                sum_mse_r += m_r
                sum_mse_i += m_i
                total_count += 1

                if global_id in vis_global_indices:
                    input_phys = denormalize_input(input_np[j], norm_stats)

                    saved_samples.append({
                        'id': global_id,
                        'input': input_phys,  # Standardized technical note.
                        'target': target_np[j],
                        'pred': pred_np[j],
                        'ssim_r': s_r, 'ssim_i': s_i,
                        'mse_r': m_r, 'mse_i': m_i
                    })
            current_idx += batch_size

    if total_count == 0:
        raise RuntimeError("No samples were evaluated.")

    overall = {
        'count': total_count,
        'ssim_real': sum_ssim_r / total_count,
        'ssim_imag': sum_ssim_i / total_count,
        'mse_real': sum_mse_r / total_count,
        'mse_imag': sum_mse_i / total_count
    }
    overall['ssim_avg'] = 0.5 * (overall['ssim_real'] + overall['ssim_imag'])
    overall['mse_avg'] = 0.5 * (overall['mse_real'] + overall['mse_imag'])

    print("\n==================== Overall Metrics ====================")
    print(f"Count: {overall['count']}")
    print(f"Real  -> MSE: {overall['mse_real']:.6f} | SSIM: {overall['ssim_real']:.4f}")
    print(f"Imag  -> MSE: {overall['mse_imag']:.6f} | SSIM: {overall['ssim_imag']:.4f}")
    print(f"Avg   -> MSE: {overall['mse_avg']:.6f} | SSIM: {overall['ssim_avg']:.4f}")
    print("========================================================")
    save_overall_metrics_to_txt(overall, save_dir)

    if len(saved_samples) == 16:
        save_metrics_to_txt(saved_samples, save_dir)
        plot_paper_fig3_style_triple(saved_samples, save_dir)
    else:
        print(f"Error: Need exactly 16 samples, but got {len(saved_samples)}")


def plot_paper_fig3_style_triple(samples, save_dir):
    print("Generating Triple-Column (Input | GT | Pred) Rotated Plots...")

    input_real_list = [np.rot90(s['input'][0], k=1) for s in samples]
    gt_real_list = [np.rot90(s['target'][0], k=1) for s in samples]
    pred_real_list = [np.rot90(s['pred'][0], k=1) for s in samples]

    input_imag_list = [np.rot90(np.abs(s['input'][1]), k=1) for s in samples]
    gt_imag_list = [np.rot90(np.abs(s['target'][1]), k=1) for s in samples]
    pred_imag_list = [np.rot90(np.abs(s['pred'][1]), k=1) for s in samples]

    padding = 2
    mosaic_in_r = create_mosaic(input_real_list, 4, 4, padding)
    mosaic_gt_r = create_mosaic(gt_real_list, 4, 4, padding)
    mosaic_pd_r = create_mosaic(pred_real_list, 4, 4, padding)

    mosaic_in_i = create_mosaic(input_imag_list, 4, 4, padding)
    mosaic_gt_i = create_mosaic(gt_imag_list, 4, 4, padding)
    mosaic_pd_i = create_mosaic(pred_imag_list, 4, 4, padding)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    vmin_gp = min(mosaic_gt_r.min(), mosaic_pd_r.min())
    vmax_gp = max(mosaic_gt_r.max(), mosaic_pd_r.max())


    # Left: Input
    im1 = axes[0].imshow(mosaic_in_r, cmap='jet')  # Standardized technical note.
    axes[0].set_title("Input (Real)", fontsize=20, fontweight='bold', pad=15)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Middle: GT
    im2 = axes[1].imshow(mosaic_gt_r, cmap='jet', vmin=vmin_gp, vmax=vmax_gp)
    axes[1].set_title("Ground Truths (Real)", fontsize=20, fontweight='bold', pad=15)
    axes[1].axis('off')

    # Right: Pred
    im3 = axes[2].imshow(mosaic_pd_r, cmap='jet', vmin=vmin_gp, vmax=vmax_gp)
    axes[2].set_title("Pred (Real)", fontsize=20, fontweight='bold', pad=15)
    axes[2].axis('off')

    cbar = fig.colorbar(im3, ax=[axes[1], axes[2]], fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
    for l in cbar.ax.yaxis.get_ticklabels(): l.set_weight('bold')

    save_path_r = os.path.join(save_dir, 'fig3_triple_Real.png')
    plt.savefig(save_path_r, dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    vmax_gp_i = max(mosaic_gt_i.max(), mosaic_pd_i.max())

    # Left: Input
    im1 = axes[0].imshow(mosaic_in_i, cmap='jet')  # Standardized technical note.
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

    print(f"[INFO] Results saved:\n  - {save_path_r}\n  - {save_path_i}\n  - metric.txt\n  - overall_metrics.txt")


if __name__ == '__main__':
    test_remaining_fig3_style()
