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

from skimage.metrics import structural_similarity as ssim_func

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

    parser.add_argument('--new_data_dir', type=str, default=None,
                        help='[Optional] Path to a folder containing NEW dataset .mat files.')

    return parser.parse_args()


def calculate_metrics_detailed(pred, target):
    """Compute MSE and SSIM separately for real and imaginary channels."""
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

    print("Configuring Dataset Paths...")

    val_transform = Compose([Resize(config['input_h'], config['input_w'])])

    def build_path(base, fname):
        if os.path.isabs(fname): return fname
        return os.path.normpath(os.path.join(base, fname))

    if args.new_data_dir is not None:
        print(f"[INFO] MODE: Using NEW Dataset from: {args.new_data_dir}")
        print("   (Assuming filenames are: chi0_all_real_64.mat, etc.)")

        path_real_img = os.path.join(args.new_data_dir, 'chi0_all_real_64.mat')
        path_imag_img = os.path.join(args.new_data_dir, 'chi0_all_imag_64.mat')
        path_real_lbl = os.path.join(args.new_data_dir, 'chi_all_real_64.mat')
        path_imag_lbl = os.path.join(args.new_data_dir, 'chi_all_imag_64.mat')

        use_subset_split = False
    else:
        print("[INFO] MODE: Using CONFIG Dataset (Validation Split)")

        path_real_img = build_path(config['data_dir'], config['real_img_file'])
        path_imag_img = build_path(config['data_dir'], config['imag_img_file'])
        path_real_lbl = build_path(config['data_dir'], config['real_label_file'])
        path_imag_lbl = build_path(config['data_dir'], config['imag_label_file'])

        use_subset_split = True

        # =========================================================

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

    stats_path = os.path.join(args.exp_dir, 'norm_stats.json')
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}")
        full_dataset.load_stats(stats_path)
    else:
        print("[WARN] norm_stats.json not found! Testing might be inaccurate.")

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

    print(f"Loading Model: {config['arch']}...")
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
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
        fourier_refine_mid=config.get('fourier_refine_mid', 32),
        use_dual_ri_refine=config.get('use_dual_ri_refine', False),
        ri_refine_mid=config.get('ri_refine_mid', 48),
        ri_refine_scale=config.get('ri_refine_scale', 0.08)
    ).to(device)

    model_path = os.path.join(args.exp_dir, args.checkpoint)
    if not os.path.exists(model_path):
        fallback_path = os.path.join(args.exp_dir, 'model.pth')
        if os.path.exists(fallback_path):
            print(f"[WARN] {args.checkpoint} not found, using model.pth")
            model_path = fallback_path
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.exp_dir}")

    print(f"Loading weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        print(f"[WARN] Strict load failed, fallback to strict=False. Reason: {e}")
        model.load_state_dict(state, strict=False)
    model.eval()

    meters = {
        'mse_real': AverageMeter(), 'mse_imag': AverageMeter(),
        'ssim_real': AverageMeter(), 'ssim_imag': AverageMeter()
    }

    # Pixel-level error accumulators for diagnostic analysis.
    sum_abs_err = None
    sum_sq_err = None
    edge_abs_err_sum = 0.0
    nonedge_abs_err_sum = 0.0
    edge_count = 0
    nonedge_count = 0

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

            abs_err = np.abs(pred_np - target_np)
            sq_err = (pred_np - target_np) ** 2
            if sum_abs_err is None:
                sum_abs_err = np.zeros_like(abs_err, dtype=np.float64)
                sum_sq_err = np.zeros_like(sq_err, dtype=np.float64)
            sum_abs_err += abs_err
            sum_sq_err += sq_err

            # Edge/non-edge error split based on target gradient magnitude.
            grad = 0.5 * (
                np.hypot(*np.gradient(target_np[0])) +
                np.hypot(*np.gradient(target_np[1]))
            )
            thr = np.percentile(grad, 80.0)
            edge_mask = grad > thr
            if edge_mask.sum() == 0:
                # Degenerate case when gradients are mostly zero.
                flat = grad.reshape(-1)
                k = max(1, int(0.2 * flat.size))
                topk_idx = np.argpartition(flat, -k)[-k:]
                edge_mask = np.zeros_like(flat, dtype=bool)
                edge_mask[topk_idx] = True
                edge_mask = edge_mask.reshape(grad.shape)
            nonedge_mask = ~edge_mask
            per_pixel_abs = abs_err.mean(axis=0)
            edge_abs_err_sum += float(per_pixel_abs[edge_mask].sum())
            nonedge_abs_err_sum += float(per_pixel_abs[nonedge_mask].sum())
            edge_count += int(edge_mask.sum())
            nonedge_count += int(nonedge_mask.sum())

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

    # Save pixel-wise error diagnostics.
    sample_count = max(1, len(final_dataset))
    mean_abs_err = (sum_abs_err / sample_count).astype(np.float32)
    mean_sq_err = (sum_sq_err / sample_count).astype(np.float32)
    np.save(os.path.join(save_full_path, 'pixel_mae_map.npy'), mean_abs_err)
    np.save(os.path.join(save_full_path, 'pixel_mse_map.npy'), mean_sq_err)

    # Max-error coordinates per channel.
    r_max_idx = np.unravel_index(np.argmax(mean_abs_err[0]), mean_abs_err[0].shape)
    i_max_idx = np.unravel_index(np.argmax(mean_abs_err[1]), mean_abs_err[1].shape)
    combined_mae = mean_abs_err.mean(axis=0)
    topk = 20
    flat_idx = np.argpartition(combined_mae.ravel(), -topk)[-topk:]
    flat_idx = flat_idx[np.argsort(combined_mae.ravel()[flat_idx])[::-1]]
    topk_coords = [np.unravel_index(int(idx), combined_mae.shape) for idx in flat_idx]
    topk_y = float(np.mean([p[0] for p in topk_coords]))
    topk_x = float(np.mean([p[1] for p in topk_coords]))

    edge_mae = edge_abs_err_sum / max(1, edge_count)
    nonedge_mae = nonedge_abs_err_sum / max(1, nonedge_count)

    report_path = os.path.join(save_full_path, 'pixel_error_report.txt')
    with open(report_path, 'w') as f:
        f.write("Pixel Error Report\n")
        f.write("==================\n")
        f.write(f"Count: {sample_count}\n")
        f.write(f"Real max MAE: {mean_abs_err[0][r_max_idx]:.6f} at (y={r_max_idx[0]}, x={r_max_idx[1]})\n")
        f.write(f"Imag max MAE: {mean_abs_err[1][i_max_idx]:.6f} at (y={i_max_idx[0]}, x={i_max_idx[1]})\n")
        f.write(f"Edge MAE (top20% grad): {edge_mae:.6f}\n")
        f.write(f"Non-edge MAE: {nonedge_mae:.6f}\n")
        if nonedge_mae > 0:
            f.write(f"Edge/Non-edge ratio: {edge_mae / nonedge_mae:.4f}\n")
        f.write(f"Top-20 mean location: (y={topk_y:.2f}, x={topk_x:.2f})\n")
        f.write("\nTop-20 combined MAE pixels (y, x, mae):\n")
        for y, x in topk_coords:
            f.write(f"({y:02d}, {x:02d}) -> {combined_mae[y, x]:.6f}\n")

    # Heatmap visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axes[0].imshow(mean_abs_err[0], cmap='hot')
    axes[0].set_title('Mean Abs Error (Real)')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(mean_abs_err[1], cmap='hot')
    axes[1].set_title('Mean Abs Error (Imag)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = axes[2].imshow(combined_mae, cmap='hot')
    axes[2].set_title('Mean Abs Error (Combined)')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    err_vis_path = os.path.join(save_full_path, 'pixel_mae_heatmap.png')
    plt.savefig(err_vis_path, dpi=300)
    plt.close()
    print(f"[INFO] Pixel error report saved to {report_path}")
    print(f"[INFO] Pixel error heatmap saved to {err_vis_path}")

    print("Plotting samples...")
    plot_samples_abs_bold(saved_samples, save_full_path)


def plot_samples_abs_bold(samples, save_dir):
    """
    Technical description.
    Technical description.
    Technical description.
    """
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 4.0 * num_samples))  # Standardized technical note.

    cols = ['GT Real', 'Pred Real', 'GT |Imag|', 'Pred |Imag|']
    if num_samples == 1: axes = [axes]

    for idx, sample in enumerate(samples):
        gt_r = sample['target'][0]
        pd_r = sample['pred'][0]

        gt_i = np.abs(sample['target'][1])
        pd_i = np.abs(sample['pred'][1])
        # -------------------

        vmin_r = min(gt_r.min(), pd_r.min())
        vmax_r = max(gt_r.max(), pd_r.max())

        vmin_i = 0
        vmax_i = max(gt_i.max(), pd_i.max())

        row_axes = axes[idx]

        if idx == 0:
            for ax, col in zip(row_axes, cols):
                ax.set_title(col, fontsize=16, fontweight='bold', pad=15)

        def plot_subplot(ax, img, vmin, vmax, cmap='jet'):
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
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
    print(f"[INFO] Visualization saved to {save_path}")


if __name__ == '__main__':
    main()
