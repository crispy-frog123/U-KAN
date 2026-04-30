"""Evaluation script for configurable splits and dataset groups.

This script loads a trained experiment directory, reconstructs/reads splits,
runs inference, and reports MSE/SSIM/RRMSE metrics with optional visual output.
"""

import argparse
import json
import os
import random
import numpy as np
import torch
import yaml
import scipy.io as sio
import h5py
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
    """Define and parse CLI arguments for evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Experiment directory (e.g., outputs/DualStream_Weight...)')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='Model filename to load (default: best_model.pth)')
    parser.add_argument('--save_dir', type=str, default='test_results_flexible',
                        help='Directory to save visualization results')

    parser.add_argument('--new_data_dir', type=str, default=None,
                        help='[Optional] Path to a folder containing NEW dataset .mat files.')
    parser.add_argument(
        '--test_group',
        type=str,
        default='config_test10',
        choices=['config_test10', 'config_full', 'measured_new','combined'],
        help='Selectable test data group when --new_data_dir is not used.'
    )
    parser.add_argument('--measured_real_img', type=str, default='inputs\\input\\chi0_all_real_new_64.mat',
                         help='Measured data_input_real')
    parser.add_argument('--measured_imag_img', type=str, default='',
                        help='Measured data_input_imag (optional in --real_only mode)')
    parser.add_argument('--measured_real_lbl', type=str, default='inputs\\label\\chi_all_real_new_64.mat',
                         help='Measured data_label_real')
    parser.add_argument('--measured_imag_lbl', type=str, default='',
                        help='Measured data_label_imag (optional in --real_only mode)')
    parser.add_argument('--combined_real_img', type=str, default='inputs\\input\\chi0_all_real_combine.mat',
                         help='Combine data_input_real')
    parser.add_argument('--combined_imag_img', type=str, default='inputs\\input\\chi0_all_imag_combine.mat',
                        help='Combine data_input_imag (optional in --real_only mode)')
    parser.add_argument('--combined_real_lbl', type=str, default='inputs\\label\\chi_all_real_combine.mat',
                         help='Combine data_label_real')
    parser.add_argument('--combined_imag_lbl', type=str, default='inputs\\label\\chi_all_imag_combine.mat',
                        help='Combined data_label_imag (optional in --real_only mode)')
    parser.add_argument('--real_only', action='store_true',
                        help='Evaluate only real channel metrics/visualization. Imag files can be omitted.')
    parser.add_argument(
        '--real_norm_mode',
        type=str,
        default='train',
        choices=['train', 'current', 'none'],
        help='Real-input normalization mode: train=use norm_stats.json; current=use current eval real data stats; none=identity.'
    )
    parser.add_argument(
        '--norm_stats_file',
        type=str,
        default='',
        help='Optional normalization stats JSON path. If relative, it is resolved under --exp_dir.'
    )
    parser.add_argument(
        '--eval_split',
        type=str,
        default='all',
        choices=['all', 'train', 'val', 'test'],
        help='Evaluate a specific split subset from finetune run. all=existing behavior; train/val/test=only that subset.'
    )

    return parser.parse_args()


def load_mat_array(path):
    """Load a MAT array using scipy, with h5py fallback for v7.3 files."""
    try:
        data = sio.loadmat(path)
        keys = [k for k in data.keys() if not k.startswith('__')]
        return np.asarray(data[keys[0]])
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
            return np.asarray(f[keys[0]])


def make_zero_mat_like(real_mat_path, save_path, key_name):
    """Create a zero-valued MAT file that matches a reference tensor shape."""
    ref = load_mat_array(real_mat_path)
    zeros = np.zeros_like(ref, dtype=np.float32)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sio.savemat(save_path, {key_name: zeros})
    return save_path


def _slugify_name(text):
    """Convert a free-form name into a filesystem-safe token."""
    text = str(text).strip()
    if not text:
        return "empty"
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "empty"


def _unique_dir(path):
    """Return a non-colliding output directory path."""
    if not os.path.exists(path):
        return path
    idx = 2
    while True:
        cand = f"{path}__run{idx}"
        if not os.path.exists(cand):
            return cand
        idx += 1


def build_split_indices_from_config(total_n, config):
    """Rebuild train/val/test indices from split ratios and random seed."""
    if total_n < 3:
        raise ValueError(f"Need at least 3 samples for train/val/test split, got {total_n}")

    seed = int(config.get('seed', config.get('dataseed', 2981)))
    val_split = float(config.get('val_split', 0.1))
    test_split = float(config.get('test_split', 0.1))
    if val_split <= 0 or test_split <= 0 or (val_split + test_split) >= 1.0:
        raise ValueError(
            f"Invalid split settings in config for reconstruction: val_split={val_split}, test_split={test_split}"
        )

    indices = list(range(total_n))
    np.random.seed(seed)
    np.random.shuffle(indices)

    val_n = max(1, int(round(val_split * total_n)))
    test_n = max(1, int(round(test_split * total_n)))
    if val_n + test_n >= total_n:
        raise ValueError(f"Invalid reconstructed split sizes: n={total_n}, val_n={val_n}, test_n={test_n}")

    return {
        'seed': seed,
        'train_indices': indices[test_n + val_n:],
        'val_indices': indices[test_n:test_n + val_n],
        'test_indices': indices[:test_n],
    }


def get_indices_for_eval_split(args, config, total_n):
    """Resolve evaluation indices from saved split file or reconstructed split."""
    split_key = f"{args.eval_split}_indices"
    split_path = os.path.join(args.exp_dir, 'split_indices.json')

    if os.path.exists(split_path):
        with open(split_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if split_key not in payload:
            raise KeyError(f"{split_key} not found in {split_path}")
        selected = [int(i) for i in payload[split_key]]
        if any((i < 0 or i >= total_n) for i in selected):
            raise ValueError(
                f"Found out-of-range index in {split_path} for dataset size {total_n}. "
                f"Please verify dataset and exp_dir match."
            )
        print(f"[INFO] Using saved split from {split_path}: {args.eval_split} ({len(selected)} samples)")
        return selected

    print(f"[WARN] {split_path} not found. Reconstructing split from config.yml.")
    rebuilt = build_split_indices_from_config(total_n, config)
    selected = [int(i) for i in rebuilt[split_key]]
    print(
        f"[INFO] Reconstructed split from config: seed={rebuilt['seed']}, "
        f"{args.eval_split}={len(selected)} samples"
    )
    return selected


def calculate_metrics_detailed(pred, target):
    """Compute MSE/SSIM and ECCV-style pixel-wise RRMSE."""
    if pred.shape[0] < 2 or target.shape[0] < 2:
        mse_real = np.mean((pred[0] - target[0]) ** 2)
        range_real = target[0].max() - target[0].min()
        if range_real == 0:
            range_real = 1e-6
        ssim_real = ssim_func(target[0], pred[0], data_range=range_real)
        return mse_real, np.nan, ssim_real, np.nan, np.nan

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

    # ECCV-style pixel-wise RRMSE on complex epsilon magnitude.
    eps_t_real = target[0] + 1.0
    eps_t_imag = target[1]
    eps_t_mag = np.sqrt(eps_t_real ** 2 + eps_t_imag ** 2)

    diff_mag = np.sqrt((pred[0] - target[0]) ** 2 + (pred[1] - target[1]) ** 2)
    error_ratio = diff_mag / (eps_t_mag + 1e-8)
    rrmse = np.sqrt(np.mean(error_ratio ** 2))

    return mse_real, mse_imag, ssim_real, ssim_imag, rrmse


def main():
    """Run evaluation and save metrics/visualization artifacts."""
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

    selected_mode = None

    if args.new_data_dir is not None:
        print(f"[INFO] MODE: Using NEW Dataset from: {args.new_data_dir}")
        print("   (Assuming filenames are: chi0_all_real_64.mat, etc.)")

        path_real_img = os.path.join(args.new_data_dir, 'chi0_all_real_64.mat')
        path_imag_img = os.path.join(args.new_data_dir, 'chi0_all_imag_64.mat')
        path_real_lbl = os.path.join(args.new_data_dir, 'chi_all_real_64.mat')
        path_imag_lbl = os.path.join(args.new_data_dir, 'chi_all_imag_64.mat')
        if args.real_only:
            tmp_dir = os.path.join(args.exp_dir, "_tmp_real_only")
            if not os.path.exists(path_imag_img):
                path_imag_img = make_zero_mat_like(
                    path_real_img, os.path.join(tmp_dir, "auto_zero_imag_input_new_data.mat"), "chi0_all_imag"
                )
                print(f"[INFO] --real_only: auto-generated zero imag input: {path_imag_img}")
            if not os.path.exists(path_imag_lbl):
                path_imag_lbl = make_zero_mat_like(
                    path_real_lbl, os.path.join(tmp_dir, "auto_zero_imag_label_new_data.mat"), "chi_all_imag"
                )
                print(f"[INFO] --real_only: auto-generated zero imag label: {path_imag_lbl}")

        use_subset_split = False
        selected_mode = 'new_data_full'
    else:
        print(f"[INFO] MODE: Using selectable group: {args.test_group}")

        if args.test_group == 'measured_new':
            path_real_img = args.measured_real_img.strip()
            path_imag_img = args.measured_imag_img.strip()
            path_real_lbl = args.measured_real_lbl.strip()
            path_imag_lbl = args.measured_imag_lbl.strip()
            required_paths = {
                '--measured_real_img': path_real_img,
                '--measured_real_lbl': path_real_lbl,
            }
            if not args.real_only:
                required_paths['--measured_imag_img'] = path_imag_img
                required_paths['--measured_imag_lbl'] = path_imag_lbl
            missing_args = [k for k, v in required_paths.items() if not v]
            if missing_args:
                raise ValueError(
                    "When --test_group measured_new is selected, please provide: " + ', '.join(missing_args)
                )

            # In real-only mode, allow missing imag files and auto-create zero placeholders.
            if args.real_only:
                tmp_dir = os.path.join(args.exp_dir, "_tmp_real_only")
                if not path_imag_img:
                    path_imag_img = make_zero_mat_like(
                        path_real_img, os.path.join(tmp_dir, "auto_zero_imag_input.mat"), "chi0_all_imag"
                    )
                    print(f"[INFO] --real_only: auto-generated zero imag input: {path_imag_img}")
                if not path_imag_lbl:
                    path_imag_lbl = make_zero_mat_like(
                        path_real_lbl, os.path.join(tmp_dir, "auto_zero_imag_label.mat"), "chi_all_imag"
                    )
                    print(f"[INFO] --real_only: auto-generated zero imag label: {path_imag_lbl}")

            use_subset_split = False
            selected_mode = 'measured_new'
        elif args.test_group == 'combined':
            path_real_img = args.combined_real_img.strip()
            path_imag_img = args.combined_imag_img.strip()
            path_real_lbl = args.combined_real_lbl.strip()
            path_imag_lbl = args.combined_imag_lbl.strip()

            required_paths = {
                '--combined_real_img': path_real_img,
                '--combined_real_lbl': path_real_lbl,
            }
            if not args.real_only:
                required_paths['--combined_imag_img'] = path_imag_img
                required_paths['--combined_imag_lbl'] = path_imag_lbl
            missing_args = [k for k, v in required_paths.items() if not v]
            if missing_args:
                raise ValueError(
                    "When --test_group combined is selected, please provide: " + ', '.join(missing_args)
                )

            # In real-only mode, allow missing/nonexistent imag files and auto-create zero placeholders.
            if args.real_only:
                tmp_dir = os.path.join(args.exp_dir, "_tmp_real_only")
                if (not path_imag_img) or (not os.path.exists(path_imag_img)):
                    path_imag_img = make_zero_mat_like(
                        path_real_img, os.path.join(tmp_dir, "auto_zero_imag_input_combined.mat"), "chi0_all_imag"
                    )
                    print(f"[INFO] --real_only: auto-generated zero imag input (combined): {path_imag_img}")
                if (not path_imag_lbl) or (not os.path.exists(path_imag_lbl)):
                    path_imag_lbl = make_zero_mat_like(
                        path_real_lbl, os.path.join(tmp_dir, "auto_zero_imag_label_combined.mat"), "chi_all_imag"
                    )
                    print(f"[INFO] --real_only: auto-generated zero imag label (combined): {path_imag_lbl}")

            use_subset_split = False
            selected_mode = 'combined'
        else:
            path_real_img = build_path(config['data_dir'], config['real_img_file'])
            path_imag_img = build_path(config['data_dir'], config['imag_img_file'])
            path_real_lbl = build_path(config['data_dir'], config['real_label_file'])
            path_imag_lbl = build_path(config['data_dir'], config['imag_label_file'])
            use_subset_split = True
            selected_mode = args.test_group

        # =========================================================

    for p in [path_real_img, path_imag_img, path_real_lbl, path_imag_lbl]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Dataset file not found: {p}")

    full_dataset = ComplexMatDataset(
        real_img_path=path_real_img,
        imag_img_path=path_imag_img,
        real_label_path=path_real_lbl,
        imag_label_path=path_imag_lbl,
        img_size=(config['original_img_size'], config['original_img_size']),
        transform=None,
        normalize_method='z-score'
    )

    if args.norm_stats_file.strip():
        stats_path = args.norm_stats_file.strip()
        if not os.path.isabs(stats_path):
            stats_path = os.path.join(args.exp_dir, stats_path)
    else:
        stats_path = os.path.join(args.exp_dir, 'norm_stats.json')
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}")
        full_dataset.load_stats(stats_path)
    else:
        print(f"[WARN] Stats file not found: {stats_path}. Testing might be inaccurate.")

    # Real-input normalization override for distribution-shift diagnosis.
    if args.real_norm_mode == 'current':
        cur_mean = float(np.mean(full_dataset.real_img))
        cur_std = float(np.std(full_dataset.real_img)) + 1e-8
        full_dataset.inp_real_mean = cur_mean
        full_dataset.inp_real_std = cur_std
        print(f"[INFO] real_norm_mode=current -> inp_real_mean={cur_mean:.6f}, inp_real_std={cur_std:.6f}")
    elif args.real_norm_mode == 'none':
        full_dataset.inp_real_mean = 0.0
        full_dataset.inp_real_std = 1.0
        print("[INFO] real_norm_mode=none -> using identity normalization for real input.")
    else:
        print(f"[INFO] real_norm_mode=train -> using real stats from: {stats_path}")

    if args.eval_split != 'all':
        chosen_indices = get_indices_for_eval_split(args, config, len(full_dataset))
        final_dataset = TransformSubset(full_dataset, chosen_indices, val_transform)
        print(f"Dataset Split: {args.eval_split.upper()} subset (from finetune split). Size: {len(final_dataset)}")
    elif use_subset_split:
        indices = list(range(len(full_dataset)))
        np.random.seed(config['dataseed'])
        np.random.shuffle(indices)

        if selected_mode == 'config_full':
            final_dataset = TransformSubset(full_dataset, indices, val_transform)
            print(f"Dataset Split: FULL Dataset (config_full). Size: {len(final_dataset)}")
        else:
            test_split = int(np.floor(0.1 * len(full_dataset)))
            test_indices = indices[:test_split]
            final_dataset = TransformSubset(full_dataset, test_indices, val_transform)
            print(f"Dataset Split: 10% Subset (config_test10). Size: {len(final_dataset)}")
    else:
        all_indices = list(range(len(full_dataset)))
        final_dataset = TransformSubset(full_dataset, all_indices, val_transform)
        print(f"Dataset Split: FULL Dataset (New Data Mode). Size: {len(final_dataset)}")

    test_loader = DataLoader(final_dataset, batch_size=1, shuffle=False)

    print(f"Loading Model: {config['arch']}...")
    base_model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        kan_inner_layers=config.get('kan_inner_layers', 3),
        backbone_stages=config.get('backbone_stages', 3),
        stage4_depth=config.get('stage4_depth', 1),
        use_edge_residual_refine=config.get('use_edge_residual_refine', True),
        use_edge_multiscale=config.get('use_edge_multiscale', True),
        use_edge_sparse_focus=config.get('use_edge_sparse_focus', True),
        edge_focus_tau=config.get('edge_focus_tau', 0.25),
        edge_focus_gamma=config.get('edge_focus_gamma', 12.0),
        edge_refine_scale=config.get('edge_refine_scale', 0.25),
        edge_refine_mid=config.get('edge_refine_mid', 48),
        use_fourier_refine=config.get('use_fourier_refine', True),
        fourier_use_fft=config.get('fourier_use_fft', True),
        fourier_refine_scale=config.get('fourier_refine_scale', 0.12),
        fourier_refine_mid=config.get('fourier_refine_mid', 48),
        use_dual_ri_refine=config.get('use_dual_ri_refine', True),
        ri_refine_mid=config.get('ri_refine_mid', 48),
        ri_refine_scale=config.get('ri_refine_scale', 0.06)
    ).to(device)

    model = base_model
    model_input_channels = int(config.get('input_channels', 2))
    model_num_classes = int(config.get('num_classes', 2))
    if model_input_channels == 1:
        print("[INFO] Detected single-channel model (input_channels=1). Inference will use real channel only.")
    if model_num_classes == 1 and not args.real_only:
        print("[WARN] Detected single-channel output (num_classes=1). Forcing real-only metrics.")
        args.real_only = True

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

    if args.real_only:
        meters = {
            'mse_real': AverageMeter(),
            'ssim_real': AverageMeter(),
        }
    else:
        meters = {
            'mse_real': AverageMeter(), 'mse_imag': AverageMeter(),
            'ssim_real': AverageMeter(), 'ssim_imag': AverageMeter(),
            'rrmse': AverageMeter()
        }

    # Pixel-level error accumulators for diagnostic analysis.
    sum_abs_err = None
    sum_sq_err = None
    edge_abs_err_sum = 0.0
    nonedge_abs_err_sum = 0.0
    edge_count = 0
    nonedge_count = 0

    if selected_mode == 'new_data_full':
        sub_folder = "results_new_data_abs"
    elif selected_mode == 'measured_new':
        sub_folder = "results_measured_new_abs"
    elif selected_mode == 'combined':
        sub_folder = "results_combined_abs"
    elif selected_mode == 'config_full':
        sub_folder = "results_config_full_abs"
    else:
        sub_folder = "results_config_test10_abs"

    save_dir_arg = args.save_dir.strip()
    if save_dir_arg and save_dir_arg != 'test_results_flexible':
        if os.path.isabs(save_dir_arg):
            save_full_path = _unique_dir(save_dir_arg)
        else:
            save_full_path = _unique_dir(os.path.join(args.exp_dir, save_dir_arg))
    else:
        ckpt_tag = _slugify_name(os.path.splitext(os.path.basename(args.checkpoint))[0])
        split_tag = _slugify_name(args.eval_split)
        norm_tag = _slugify_name(args.real_norm_mode)
        save_leaf = f"{sub_folder}__ckpt_{ckpt_tag}__split_{split_tag}__rnorm_{norm_tag}"
        save_full_path = _unique_dir(os.path.join(args.exp_dir, save_leaf))

    os.makedirs(save_full_path, exist_ok=False)
    print(f"[INFO] Saving test outputs to: {save_full_path}")

    vis_count = min(10, len(final_dataset))
    random.seed(config['dataseed'])
    vis_indices = set(random.sample(range(len(final_dataset)), vis_count))
    saved_samples = []

    print("Running Inference...")
    with torch.no_grad():
        for i, (input_tensor, target, _) in enumerate(tqdm(test_loader)):
            input_tensor = input_tensor.to(device)
            target = target.to(device)
            if model_input_channels == 1 and input_tensor.shape[1] > 1:
                model_input = input_tensor[:, 0:1]
            else:
                model_input = input_tensor

            outputs = model(model_input)
            if isinstance(outputs, list):
                pred = outputs[-1]
            else:
                pred = outputs

            if pred.shape[1] == 1 and target.shape[1] > 1:
                target_eval = target[:, 0:1]
            else:
                target_eval = target[:, :pred.shape[1]]

            pred_np = pred.cpu().numpy().squeeze(0)
            target_np = target_eval.cpu().numpy().squeeze(0)

            abs_err = np.abs(pred_np - target_np)
            sq_err = (pred_np - target_np) ** 2
            if sum_abs_err is None:
                sum_abs_err = np.zeros_like(abs_err, dtype=np.float64)
                sum_sq_err = np.zeros_like(sq_err, dtype=np.float64)
            sum_abs_err += abs_err
            sum_sq_err += sq_err

            # Edge/non-edge error split based on target gradient magnitude.
            if args.real_only:
                grad = np.hypot(*np.gradient(target_np[0]))
            else:
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
            if args.real_only:
                per_pixel_abs = abs_err[0]
            else:
                per_pixel_abs = abs_err.mean(axis=0)
            edge_abs_err_sum += float(per_pixel_abs[edge_mask].sum())
            nonedge_abs_err_sum += float(per_pixel_abs[nonedge_mask].sum())
            edge_count += int(edge_mask.sum())
            nonedge_count += int(nonedge_mask.sum())

            mse_r, mse_i, ssim_r, ssim_i, rrmse = calculate_metrics_detailed(pred_np, target_np)

            meters['mse_real'].update(mse_r)
            meters['ssim_real'].update(ssim_r)
            if not args.real_only:
                meters['mse_imag'].update(mse_i)
                meters['ssim_imag'].update(ssim_i)
                meters['rrmse'].update(rrmse)

            if i in vis_indices:
                saved_samples.append({
                    'id': i,
                    'pred': pred_np,
                    'target': target_np,
                    'ssim_r': ssim_r,
                    'ssim_i': ssim_i
                })

    print(f"\n{'=' * 20} Final Results {'=' * 20}")
    if args.real_only:
        mean_mse = meters['mse_real'].avg
        avg_ssim = meters['ssim_real'].avg
        print(f"Real Part -> MSE: {meters['mse_real'].avg:.6f} | SSIM: {meters['ssim_real'].avg:.4f}")
        print(f"Overall   -> Mean MSE: {mean_mse:.6f} | Avg SSIM: {avg_ssim:.4f}")
    else:
        mean_mse = (meters['mse_real'].avg + meters['mse_imag'].avg) / 2
        avg_ssim = (meters['ssim_real'].avg + meters['ssim_imag'].avg) / 2
        print(f"Real Part -> MSE: {meters['mse_real'].avg:.6f} | SSIM: {meters['ssim_real'].avg:.4f}")
        print(f"Imag Part -> MSE: {meters['mse_imag'].avg:.6f} | SSIM: {meters['ssim_imag'].avg:.4f}")
        print(f"Overall   -> Mean MSE: {mean_mse:.6f} | Avg SSIM: {avg_ssim:.4f} | RRMSE: {meters['rrmse'].avg:.6f}")
    print("=" * 55)

    txt_path = os.path.join(save_full_path, 'metrics.txt')
    with open(txt_path, 'w') as f:
        f.write("Evaluation Report\n")
        f.write("=================\n")
        f.write(f"Data Mode: {selected_mode}\n")
        f.write(f"Real Only: {args.real_only}\n")
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Count: {len(final_dataset)}\n\n")
        f.write(f"Mean MSE:  {mean_mse:.6f}\n")
        f.write(f"Real SSIM: {meters['ssim_real'].avg:.6f}\n")
        if args.real_only:
            f.write(f"Avg SSIM:  {meters['ssim_real'].avg:.6f}\n")
        else:
            f.write(f"Imag SSIM: {meters['ssim_imag'].avg:.6f}\n")
            f.write(f"Avg SSIM:  {avg_ssim:.6f}\n")
            f.write(f"RRMSE:     {meters['rrmse'].avg:.6f}\n")

    # Save pixel-wise error diagnostics.
    sample_count = max(1, len(final_dataset))
    mean_abs_err = (sum_abs_err / sample_count).astype(np.float32)
    mean_sq_err = (sum_sq_err / sample_count).astype(np.float32)
    np.save(os.path.join(save_full_path, 'pixel_mae_map.npy'), mean_abs_err)
    np.save(os.path.join(save_full_path, 'pixel_mse_map.npy'), mean_sq_err)

    # Max-error coordinates per channel.
    r_max_idx = np.unravel_index(np.argmax(mean_abs_err[0]), mean_abs_err[0].shape)
    if args.real_only:
        combined_mae = mean_abs_err[0]
    else:
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
        if not args.real_only:
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
    if args.real_only:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
        im = ax.imshow(mean_abs_err[0], cmap='hot')
        ax.set_title('Mean Abs Error (Real)')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
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
    plot_samples_abs_bold(saved_samples, save_full_path, real_only=args.real_only)


def plot_samples_abs_bold(samples, save_dir, real_only=False):
    """
    Technical description.
    Technical description.
    Technical description.
    """
    num_samples = len(samples)
    n_cols = 2 if real_only else 4
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(5.5 * n_cols, 4.0 * num_samples))

    cols = ['GT Real', 'Pred Real'] if real_only else ['GT Real', 'Pred Real', 'GT |Imag|', 'Pred |Imag|']
    if num_samples == 1:
        axes = [axes]

    for idx, sample in enumerate(samples):
        gt_r = sample['target'][0]
        pd_r = sample['pred'][0]

        if not real_only:
            gt_i = np.abs(sample['target'][1])
            pd_i = np.abs(sample['pred'][1])

        vmin_r = min(gt_r.min(), pd_r.min())
        vmax_r = max(gt_r.max(), pd_r.max())

        if not real_only:
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

        if not real_only:
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
