import argparse
import os
import random
import numpy as np
import torch
import yaml
import scipy.io as sio
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations.core.composition import Compose
from albumentations import Resize
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

import archs
from dataset_e import ComplexMatDataset, TransformSubset
import warnings

warnings.filterwarnings('ignore')


LARGE_DATA_PATHS = {
    'real_img': 'data/chi0_all_real_mnist.mat',
    'imag_img': 'data/chi0_all_imag_mnist.mat',
    'real_label': 'data/chi_all_real_mnist.mat',
    'imag_label': 'data/chi_all_imag_mnist.mat'
}
INDICES_PATH = 'data/remaining_test_indices.mat'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True, help='Experiment directory')
    parser.add_argument('--checkpoint', type=str, default='model.pth', help='Model checkpoint')
    parser.add_argument('--save_dir', type=str, default='paper_results/dim1_fixed_ticks', help='Save directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Inference batch size')
    return parser.parse_args()


def calculate_batch_metrics(pred, target):
    ssim_scores = []
    mse_scores = []
    rrmse_scores = []
    psnr_scores = []
    B = pred.shape[0]

    for i in range(B):
        p = pred[i]
        t = target[i]

        range_r = t[0].max() - t[0].min() + 1e-6
        range_i = t[1].max() - t[1].min() + 1e-6

        s_r = ssim_func(t[0], p[0], data_range=range_r)
        s_i = ssim_func(t[1], p[1], data_range=range_i)
        ssim_scores.append((s_r + s_i) / 2.0)

        p_r = psnr_func(t[0], p[0], data_range=range_r)
        p_i = psnr_func(t[1], p[1], data_range=range_i)
        psnr_scores.append((p_r + p_i) / 2.0)

        mse_val = np.mean((p - t) ** 2)
        mse_scores.append(mse_val)

        eps_t_real = t[0] + 1.0
        eps_t_imag = t[1]
        eps_t_mag = np.sqrt(eps_t_real ** 2 + eps_t_imag ** 2)

        diff_mag = np.sqrt((p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2)
        error_ratio = diff_mag / (eps_t_mag + 1e-8)
        rrmse_val = np.sqrt(np.mean(error_ratio ** 2))
        rrmse_scores.append(rrmse_val)

    return ssim_scores, mse_scores, rrmse_scores, psnr_scores


def run_inference(model, loader, device, name='Dataset'):
    model.eval()
    all_ssims = []
    all_mses = []
    all_rrmses = []
    all_psnrs = []

    print(f'Running inference on {name} ({len(loader.dataset)} samples)...')
    with torch.no_grad():
        for input_tensor, target, _ in tqdm(loader, desc=name):
            input_tensor = input_tensor.to(device)
            target = target.to(device)

            if hasattr(model, 'deep_supervision') and model.deep_supervision:
                output = model(input_tensor)
                if isinstance(output, list):
                    output = output[-1]
            else:
                output = model(input_tensor)

            pred_np = output.cpu().numpy()
            target_np = target.cpu().numpy()

            batch_ssim, batch_mse, batch_rrmse, batch_psnr = calculate_batch_metrics(pred_np, target_np)
            all_ssims.extend(batch_ssim)
            all_mses.extend(batch_mse)
            all_rrmses.extend(batch_rrmse)
            all_psnrs.extend(batch_psnr)

    return all_ssims, all_mses, all_rrmses, all_psnrs


def plot_paper_style_hist_count(data_list, label_list, color_list, metric_name, save_path):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5

    fig = plt.figure(figsize=(8, 6))
    sns.set_style('ticks')
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

    all_data = np.concatenate(data_list)
    min_val, max_val = np.min(all_data), np.max(all_data)
    bins = np.linspace(min_val, max_val, 80)

    for data, label, color in zip(data_list, label_list, color_list):
        mean_val = np.mean(data)
        count_val = len(data)
        label_text = f'{label} ($N={count_val}, \\mu={mean_val:.4f}$)'

        sns.histplot(
            data,
            bins=bins,
            color=color,
            label=label_text,
            kde=True,
            stat='count',
            element='bars',
            fill=True,
            alpha=0.35,
            edgecolor=color,
            linewidth=0.5,
            ax=ax
        )

        ax.axvline(mean_val, color=color, linestyle='--', linewidth=2.0, alpha=0.9)

    ax.set_xlabel('')
    ax.set_ylabel('Frequency (Count)', fontsize=16, fontweight='bold', fontname='Times New Roman')

    loc = 'upper right' if metric_name in ['MSE', 'RRMSE'] else 'upper left'
    ax.legend(fontsize=11, loc=loc, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)

    if metric_name == 'MSE':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_xlim(0, max(1.0, max_val * 1.1))
    elif metric_name == 'SSIM':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_xlim(0, 1.0)
    elif metric_name == 'RRMSE':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_xlim(0, max(1.0, max_val * 1.1))
    elif metric_name == 'PSNR':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
        ax.set_xlim(max(0, min_val - 2.0), max_val + 2.0)

    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
    ax.set_axisbelow(True)

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'{metric_name} Histogram saved to: {save_path}')


def _plot_paper_style_hist_on_ax(ax, data_list, label_list, color_list, metric_name):
    all_data = np.concatenate(data_list)
    min_val, max_val = np.min(all_data), np.max(all_data)
    bins = np.linspace(min_val, max_val, 80)

    for data, label, color in zip(data_list, label_list, color_list):
        mean_val = np.mean(data)
        count_val = len(data)
        label_text = f'{label} ($N={count_val}, \\mu={mean_val:.4f}$)'

        sns.histplot(
            data,
            bins=bins,
            color=color,
            label=label_text,
            kde=True,
            stat='count',
            element='bars',
            fill=True,
            alpha=0.35,
            edgecolor=color,
            linewidth=0.5,
            ax=ax
        )

        ax.axvline(mean_val, color=color, linestyle='--', linewidth=2.0, alpha=0.9)

    ax.set_xlabel('')
    ax.set_ylabel('Frequency (Count)', fontsize=16, fontweight='bold', fontname='Times New Roman')

    loc = 'upper right' if metric_name in ['MSE', 'RRMSE'] else 'upper left'
    ax.legend(fontsize=11, loc=loc, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)

    if metric_name == 'MSE':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_xlim(0, max(1.0, max_val * 1.1))
    elif metric_name == 'SSIM':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_xlim(0, 1.0)
    elif metric_name == 'RRMSE':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_xlim(0, max(1.0, max_val * 1.1))
    elif metric_name == 'PSNR':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
        ax.set_xlim(max(0, min_val - 2.0), max_val + 2.0)

    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
    ax.set_axisbelow(True)


def plot_combined_ssim_mse(data_list_ssim, data_list_mse, label_list, color_list, save_path):
    # Keep style identical to single-plot figures.
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5

    fig = plt.figure(figsize=(8, 6))
    sns.set_style('ticks')
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

    merged_data = []
    merged_labels = []
    merged_colors = []
    # SSIM uses solid mean line
    for data, label, color in zip(data_list_ssim, label_list, color_list):
        merged_data.append(data)
        merged_labels.append(f'{label} SSIM')
        merged_colors.append(color)
    # MSE uses the same base color but with lighter alpha; label differentiates metric
    for data, label, color in zip(data_list_mse, label_list, color_list):
        merged_data.append(data)
        merged_labels.append(f'{label} MSE')
        merged_colors.append(color)

    all_data = np.concatenate(merged_data)
    min_val, max_val = np.min(all_data), np.max(all_data)
    bins = np.linspace(min_val, max_val, 80)

    for i, (data, label, color) in enumerate(zip(merged_data, merged_labels, merged_colors)):
        mean_val = np.mean(data)
        count_val = len(data)
        label_text = f'{label} ($N={count_val}, \\mu={mean_val:.4f}$)'

        # Keep same histogram style; use lower alpha for MSE overlays
        alpha = 0.35 if i < len(data_list_ssim) else 0.20
        sns.histplot(
            data,
            bins=bins,
            color=color,
            label=label_text,
            kde=True,
            stat='count',
            element='bars',
            fill=True,
            alpha=alpha,
            edgecolor=color,
            linewidth=0.5,
            ax=ax
        )

        # SSIM mean: solid, MSE mean: dashed
        line_style = '-' if i < len(data_list_ssim) else '--'
        ax.axvline(mean_val, color=color, linestyle=line_style, linewidth=2.0, alpha=0.9)

    ax.set_xlabel('')
    ax.set_ylabel('MSE', fontsize=16, fontweight='bold', fontname='Times New Roman', color='#d62728')
    ax.tick_params(axis='y', labelcolor='#d62728')
    ax.legend(fontsize=10, loc='upper right', frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlim(0, max(1.0, max_val * 1.1))
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
    ax.set_axisbelow(True)

    # Right-side title for MSE on the same count scale.
    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_ylabel('SSIM', fontsize=16, fontweight='bold', fontname='Times New Roman', color='#1f77b4')
    ax_right.tick_params(axis='y', labelcolor='#1f77b4')

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'SSIM+MSE combined histogram (same axes) saved to: {save_path}')


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_path = os.path.join(args.exp_dir, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    stats_path = os.path.join(args.exp_dir, 'norm_stats.json')
    val_transform = Compose([Resize(config['input_h'], config['input_w'])])

    print('\n--- Loading Datasets ---')

    def build_path(base_dir, file_path):
        if os.path.isabs(file_path):
            return file_path
        return os.path.normpath(os.path.join(base_dir, file_path))

    dataset_A = ComplexMatDataset(
        real_img_path=build_path(config['data_dir'], config['real_img_file']),
        imag_img_path=build_path(config['data_dir'], config['imag_img_file']),
        real_label_path=build_path(config['data_dir'], config['real_label_file']),
        imag_label_path=build_path(config['data_dir'], config['imag_label_file']),
        img_size=(config['original_img_size'], config['original_img_size']),
        normalize_method='z-score'
    )
    if os.path.exists(stats_path):
        dataset_A.load_stats(stats_path)

    indices_A = list(range(len(dataset_A)))
    np.random.seed(config['dataseed'])
    np.random.shuffle(indices_A)

    idx_test_A = indices_A[:200]
    idx_train_A = indices_A[400:]

    train_set = TransformSubset(dataset_A, idx_train_A, val_transform)
    test_set = TransformSubset(dataset_A, idx_test_A, val_transform)

    dataset_B = ComplexMatDataset(
        real_img_path=LARGE_DATA_PATHS['real_img'],
        imag_img_path=LARGE_DATA_PATHS['imag_img'],
        real_label_path=LARGE_DATA_PATHS['real_label'],
        imag_label_path=LARGE_DATA_PATHS['imag_label'],
        img_size=(64, 64),
        normalize_method='z-score'
    )
    if os.path.exists(stats_path):
        dataset_B.load_stats(stats_path)

    if not os.path.exists(INDICES_PATH):
        raise FileNotFoundError(f'Missing: {INDICES_PATH}')
    mat_content = sio.loadmat(INDICES_PATH)
    idx_remain_B = (mat_content['test_indices'][0] - 1).astype(np.int64)
    remain_set = TransformSubset(dataset_B, idx_remain_B, val_transform)

    bs = args.batch_size
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=4)
    remain_loader = DataLoader(remain_set, batch_size=bs * 2, shuffle=False, num_workers=4)

    print(f'\nLoading Model: {config["arch"]}...')
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config.get('no_kan', False),
        use_edge_residual_refine=config.get('use_edge_residual_refine', False),
        use_edge_multiscale=config.get('use_edge_multiscale', False),
        use_edge_sparse_focus=config.get('use_edge_sparse_focus', False),
        edge_focus_tau=config.get('edge_focus_tau', 0.25),
        edge_focus_gamma=config.get('edge_focus_gamma', 2.0),
        use_edge_center_boost=config.get('use_edge_center_boost', False),
        edge_center_boost=config.get('edge_center_boost', 0.1),
        edge_center_sigma=config.get('edge_center_sigma', 0.1),
        edge_refine_scale=config.get('edge_refine_scale', 1.0),
        edge_refine_mid=config.get('edge_refine_mid', 64),
        use_detail_skip_refine=config.get('use_detail_skip_refine', False),
        detail_refine_scale=config.get('detail_refine_scale', 1.0),
        detail_refine_mid=config.get('detail_refine_mid', 64),
        use_fourier_refine=config.get('use_fourier_refine', False),
        fourier_use_fft=config.get('fourier_use_fft', False),
        fourier_refine_scale=config.get('fourier_refine_scale', 1.0),
        fourier_refine_mid=config.get('fourier_refine_mid', 64)
    ).to(device)

    ckpt = torch.load(os.path.join(args.exp_dir, args.checkpoint), map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)

    train_ssim, train_mse, train_rrmse, train_psnr = run_inference(model, train_loader, device, 'Train Set')
    test_ssim, test_mse, test_rrmse, test_psnr = run_inference(model, test_loader, device, 'Test Set')
    remain_ssim, remain_mse, remain_rrmse, remain_psnr = run_inference(model, remain_loader, device, 'Unseen Set')

    print('\nPlotting separated histograms...')

    plot_combined_ssim_mse(
        [train_ssim, test_ssim],
        [train_mse, test_mse],
        ['Train Set', 'Test Set'],
        ['#1f77b4', '#ff7f0e'],
        os.path.join(args.save_dir, 'SSIM_MSE_small_data.png')
    )
    plot_paper_style_hist_count([train_rrmse, test_rrmse], ['Train Set', 'Test Set'], ['#1f77b4', '#ff7f0e'], 'RRMSE',
                                os.path.join(args.save_dir, 'RRMSE_small_data.png'))
    plot_paper_style_hist_count([train_psnr, test_psnr], ['Train Set', 'Test Set'], ['#1f77b4', '#ff7f0e'], 'PSNR',
                                os.path.join(args.save_dir, 'PSNR_small_data.png'))

    plot_combined_ssim_mse(
        [remain_ssim],
        [remain_mse],
        ['Unseen Set'],
        ['#2ca02c'],
        os.path.join(args.save_dir, 'SSIM_MSE_large_data.png')
    )
    plot_paper_style_hist_count([remain_rrmse], ['Unseen Set'], ['#2ca02c'], 'RRMSE',
                                os.path.join(args.save_dir, 'RRMSE_large_data.png'))
    plot_paper_style_hist_count([remain_psnr], ['Unseen Set'], ['#2ca02c'], 'PSNR',
                                os.path.join(args.save_dir, 'PSNR_large_data.png'))

    print('\nCalculating metrics fluctuation...')

    baseline_ssim, unseen_ssim = np.mean(test_ssim), np.mean(remain_ssim)
    ssim_diff = unseen_ssim - baseline_ssim
    ssim_fluctuation_percent = (ssim_diff / baseline_ssim) * 100

    baseline_mse, unseen_mse = np.mean(test_mse), np.mean(remain_mse)
    mse_diff = unseen_mse - baseline_mse
    mse_fluctuation_percent = (mse_diff / baseline_mse) * 100

    baseline_rrmse, unseen_rrmse = np.mean(test_rrmse), np.mean(remain_rrmse)
    rrmse_diff = unseen_rrmse - baseline_rrmse
    rrmse_fluctuation_percent = (rrmse_diff / baseline_rrmse) * 100

    baseline_psnr, unseen_psnr = np.mean(test_psnr), np.mean(remain_psnr)
    psnr_diff = unseen_psnr - baseline_psnr
    psnr_fluctuation_percent = (psnr_diff / baseline_psnr) * 100

    txt_save_path = os.path.join(args.save_dir, 'metrics_fluctuation.txt')
    with open(txt_save_path, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('Model Robustness Analysis: Small vs Large Dataset Fluctuation\n')
        f.write('=' * 60 + '\n\n')

        f.write('[Baseline] Small Data (Test Set, N~200):\n')
        f.write(f'  Mean SSIM:  {baseline_ssim:.6f}\n')
        f.write(f'  Mean PSNR:  {baseline_psnr:.6f} dB\n')
        f.write(f'  Mean MSE:   {baseline_mse:.6f}\n')
        f.write(f'  Mean RRMSE: {baseline_rrmse:.6f}\n\n')

        f.write('[Target] Large Data (Unseen Set, N~1.8w):\n')
        f.write(f'  Mean SSIM:  {unseen_ssim:.6f}\n')
        f.write(f'  Mean PSNR:  {unseen_psnr:.6f} dB\n')
        f.write(f'  Mean MSE:   {unseen_mse:.6f}\n')
        f.write(f'  Mean RRMSE: {unseen_rrmse:.6f}\n\n')

        f.write('-' * 60 + '\n')
        f.write('Fluctuation Results (Target - Baseline):\n')
        f.write('-' * 60 + '\n')
        f.write(f'SSIM Diff: {ssim_diff:+.6f}\n')
        f.write(f'SSIM Fluctuation Rate: {ssim_fluctuation_percent:+.4f}%\n\n')

        f.write(f'PSNR Diff: {psnr_diff:+.6f} dB\n')
        f.write(f'PSNR Fluctuation Rate: {psnr_fluctuation_percent:+.4f}%\n\n')

        f.write(f'MSE Diff:  {mse_diff:+.6f}\n')
        f.write(f'MSE Fluctuation Rate:  {mse_fluctuation_percent:+.4f}%\n\n')

        f.write(f'RRMSE Diff:  {rrmse_diff:+.6f}\n')
        f.write(f'RRMSE Fluctuation Rate:  {rrmse_fluctuation_percent:+.4f}%\n')
        f.write('=' * 60 + '\n')

    print(f'Fluctuation analysis saved to: {txt_save_path}')
    print(f'Done! Check: {args.save_dir}')


if __name__ == '__main__':
    main()
