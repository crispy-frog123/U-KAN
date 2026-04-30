"""Diagnostic evaluation script for U-KAN electromagnetic reconstruction.

This script keeps test-time loading behavior aligned with `test_norm.py`,
while adding richer diagnostics without modifying `test_norm.py`:

1) Local structure maps: SSIM map + GMS map
2) Boundary-band error profile
3) Shape/position diagnostics
4) Frequency-domain radial error spectrum
5) Complex magnitude/phase decomposition diagnostics
"""

import argparse
import csv
import json
import os
import random
import warnings

import h5py
import matplotlib
import numpy as np
import scipy.io as sio
import torch
import yaml
from albumentations import Resize
from albumentations.core.composition import Compose
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as ssim_func
from torch.utils.data import DataLoader
from tqdm import tqdm

import archs
from dataset_e import ComplexMatDataset, TransformSubset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory (e.g., outputs/xxx)")
    parser.add_argument("--checkpoint", type=str, default="model_best_ssim.pth",
                        help="Model filename to load")
    parser.add_argument("--save_dir", type=str, default="diagnostic_report",
                        help="Output folder for diagnostic artifacts")

    # Keep dataset route behavior consistent with test_norm.py
    parser.add_argument("--new_data_dir", type=str, default=None)
    parser.add_argument("--test_group", type=str, default="config_test10",
                        choices=["config_test10", "config_full", "measured_new", "combined"])
    parser.add_argument("--measured_real_img", type=str, default="inputs\\input\\chi0_all_real_new_64.mat")
    parser.add_argument("--measured_imag_img", type=str, default="")
    parser.add_argument("--measured_real_lbl", type=str, default="inputs\\label\\chi_all_real_new_64.mat")
    parser.add_argument("--measured_imag_lbl", type=str, default="")
    parser.add_argument("--combined_real_img", type=str, default="inputs\\input\\chi0_all_real_combine.mat")
    parser.add_argument("--combined_imag_img", type=str, default="inputs\\input\\chi0_all_imag_combine.mat")
    parser.add_argument("--combined_real_lbl", type=str, default="inputs\\label\\chi_all_real_combine.mat")
    parser.add_argument("--combined_imag_lbl", type=str, default="inputs\\label\\chi_all_imag_combine.mat")
    parser.add_argument("--real_only", action="store_true")
    parser.add_argument("--real_norm_mode", type=str, default="train", choices=["train", "current", "none"])
    parser.add_argument("--norm_stats_file", type=str, default="")
    parser.add_argument("--eval_split", type=str, default="all", choices=["all", "train", "val", "test"])

    # Diagnostic options
    parser.add_argument("--edge_percentile", type=float, default=80.0,
                        help="Percentile used to define boundary mask from GT gradient")
    parser.add_argument("--boundary_bands", type=str, default="0,1,3,5,8,1000",
                        help="Comma-separated distance band edges in pixels")
    parser.add_argument("--shape_thresh_method", type=str, default="otsu", choices=["otsu", "percentile"])
    parser.add_argument("--shape_percentile", type=float, default=75.0,
                        help="Used when shape_thresh_method=percentile")
    parser.add_argument("--gms_c", type=float, default=0.0026,
                        help="Stability constant in GMS map computation")
    parser.add_argument("--topk_hard", type=int, default=20,
                        help="Number of hard samples to export")
    parser.add_argument("--max_cases", type=int, default=0,
                        help="Debug option: cap evaluated samples (0 means all)")
    return parser.parse_args()


def load_mat_array(path):
    try:
        data = sio.loadmat(path)
        keys = [k for k in data.keys() if not k.startswith("__")]
        return np.asarray(data[keys[0]])
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            return np.asarray(f[keys[0]])


def make_zero_mat_like(real_mat_path, save_path, key_name):
    ref = load_mat_array(real_mat_path)
    zeros = np.zeros_like(ref, dtype=np.float32)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sio.savemat(save_path, {key_name: zeros})
    return save_path


def _slugify_name(text):
    text = str(text).strip()
    if not text:
        return "empty"
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "empty"


def _unique_dir(path):
    if not os.path.exists(path):
        return path
    idx = 2
    while True:
        cand = f"{path}__run{idx}"
        if not os.path.exists(cand):
            return cand
        idx += 1


def build_split_indices_from_config(total_n, config):
    if total_n < 3:
        raise ValueError(f"Need at least 3 samples for train/val/test split, got {total_n}")

    seed = int(config.get("seed", config.get("dataseed", 2981)))
    val_split = float(config.get("val_split", 0.1))
    test_split = float(config.get("test_split", 0.1))
    if val_split <= 0 or test_split <= 0 or (val_split + test_split) >= 1.0:
        raise ValueError(
            f"Invalid split settings: val_split={val_split}, test_split={test_split}"
        )

    indices = list(range(total_n))
    np.random.seed(seed)
    np.random.shuffle(indices)

    val_n = max(1, int(round(val_split * total_n)))
    test_n = max(1, int(round(test_split * total_n)))
    if val_n + test_n >= total_n:
        raise ValueError(f"Invalid reconstructed split sizes: n={total_n}, val_n={val_n}, test_n={test_n}")

    return {
        "seed": seed,
        "train_indices": indices[test_n + val_n:],
        "val_indices": indices[test_n:test_n + val_n],
        "test_indices": indices[:test_n],
    }


def get_indices_for_eval_split(args, config, total_n):
    split_key = f"{args.eval_split}_indices"
    split_path = os.path.join(args.exp_dir, "split_indices.json")

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if split_key not in payload:
            raise KeyError(f"{split_key} not found in {split_path}")
        selected = [int(i) for i in payload[split_key]]
        if any((i < 0 or i >= total_n) for i in selected):
            raise ValueError(f"Out-of-range index found in {split_path} for dataset size {total_n}")
        print(f"[INFO] Using saved split from {split_path}: {args.eval_split} ({len(selected)} samples)")
        return selected

    print(f"[WARN] {split_path} not found. Reconstructing split from config.yml.")
    rebuilt = build_split_indices_from_config(total_n, config)
    selected = [int(i) for i in rebuilt[split_key]]
    print(f"[INFO] Reconstructed split: seed={rebuilt['seed']}, {args.eval_split}={len(selected)} samples")
    return selected


def safe_data_range(a):
    dr = float(np.max(a) - np.min(a))
    return dr if dr > 1e-6 else 1e-6


def gradient_mag(img2d):
    gx, gy = np.gradient(img2d)
    return np.sqrt(gx * gx + gy * gy + 1e-12)


def complex_mag(arr_chw):
    if arr_chw.shape[0] >= 2:
        return np.sqrt(arr_chw[0] ** 2 + arr_chw[1] ** 2)
    return np.abs(arr_chw[0])


def complex_phase(arr_chw):
    if arr_chw.shape[0] >= 2:
        return np.arctan2(arr_chw[1], arr_chw[0])
    return np.zeros_like(arr_chw[0], dtype=np.float32)


def wrapped_phase_diff(pred_phase, gt_phase):
    return np.angle(np.exp(1j * (pred_phase - gt_phase)))


def ssim_score_and_map(gt2d, pred2d):
    dr = safe_data_range(gt2d)
    score, smap = ssim_func(gt2d, pred2d, data_range=dr, full=True)
    return float(score), smap.astype(np.float32)


def gms_map(gt2d, pred2d, c=0.0026):
    lo = min(float(gt2d.min()), float(pred2d.min()))
    hi = max(float(gt2d.max()), float(pred2d.max()))
    scale = max(hi - lo, 1e-6)
    gt_n = (gt2d - lo) / scale
    pred_n = (pred2d - lo) / scale
    g1 = gradient_mag(gt_n)
    g2 = gradient_mag(pred_n)
    return ((2.0 * g1 * g2 + c) / (g1 * g1 + g2 * g2 + c)).astype(np.float32)


def parse_bands(text):
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    vals = sorted(vals)
    if len(vals) < 2:
        raise ValueError("boundary_bands must contain at least two numbers")
    return vals


def boundary_mask_from_target(target_np, percentile):
    mag = complex_mag(target_np)
    grad = gradient_mag(mag)
    thr = float(np.percentile(grad, percentile))
    mask = grad > thr
    if mask.sum() == 0:
        flat = grad.reshape(-1)
        k = max(1, int(0.2 * flat.size))
        topk_idx = np.argpartition(flat, -k)[-k:]
        mask = np.zeros_like(flat, dtype=bool)
        mask[topk_idx] = True
        mask = mask.reshape(grad.shape)
    return mask


def largest_component(mask):
    if mask.sum() == 0:
        return mask
    lbl, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = int(np.argmax(sizes))
    return lbl == keep


def mask_from_mag(mag, method, percentile):
    if method == "otsu":
        try:
            thr = float(threshold_otsu(mag))
        except Exception:
            thr = float(np.percentile(mag, percentile))
    else:
        thr = float(np.percentile(mag, percentile))
    m = mag > thr
    return largest_component(m), thr


def centroid(mask):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return float(np.mean(ys)), float(np.mean(xs))


def mask_boundary(mask):
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    er = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.logical_xor(mask, er)


def pratt_fom(pred_edge, gt_edge, alpha=1.0 / 9.0):
    n_pred = int(pred_edge.sum())
    n_gt = int(gt_edge.sum())
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    dist = distance_transform_edt(~gt_edge)
    dvals = dist[pred_edge]
    return float(np.sum(1.0 / (1.0 + alpha * dvals * dvals)) / max(n_pred, n_gt))


def dice_iou(pred_mask, gt_mask):
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    a = int(pred_mask.sum())
    b = int(gt_mask.sum())
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    dice = (2.0 * inter) / max(a + b, 1)
    iou = inter / max(union, 1)
    return float(dice), float(iou), a, b


def radial_profile(img2d):
    h, w = img2d.shape
    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_int = np.floor(rr).astype(np.int32)
    max_r = min(h, w) // 2
    mask = r_int < max_r
    tbin = np.bincount(r_int[mask].ravel(), weights=img2d[mask].ravel(), minlength=max_r)
    nr = np.bincount(r_int[mask].ravel(), minlength=max_r)
    nr = np.maximum(nr, 1)
    return (tbin / nr).astype(np.float64)


def radial_power_spectrum(img2d):
    f = np.fft.fftshift(np.fft.fft2(img2d.astype(np.float32)))
    p = np.abs(f) ** 2
    return radial_profile(p)


def calculate_metrics_detailed(pred, target):
    if pred.shape[0] < 2 or target.shape[0] < 2:
        mse_real = np.mean((pred[0] - target[0]) ** 2)
        range_real = target[0].max() - target[0].min()
        if range_real == 0:
            range_real = 1e-6
        ssim_real = ssim_func(target[0], pred[0], data_range=range_real)
        return mse_real, np.nan, ssim_real, np.nan, np.nan

    mse_real = np.mean((pred[0] - target[0]) ** 2)
    mse_imag = np.mean((pred[1] - target[1]) ** 2)
    range_real = target[0].max() - target[0].min()
    if range_real == 0:
        range_real = 1e-6
    ssim_real = ssim_func(target[0], pred[0], data_range=range_real)

    range_imag = target[1].max() - target[1].min()
    if range_imag == 0:
        range_imag = 1e-6
    ssim_imag = ssim_func(target[1], pred[1], data_range=range_imag)

    eps_t_real = target[0] + 1.0
    eps_t_imag = target[1]
    eps_t_mag = np.sqrt(eps_t_real ** 2 + eps_t_imag ** 2)
    diff_mag = np.sqrt((pred[0] - target[0]) ** 2 + (pred[1] - target[1]) ** 2)
    error_ratio = diff_mag / (eps_t_mag + 1e-8)
    rrmse = np.sqrt(np.mean(error_ratio ** 2))
    return mse_real, mse_imag, ssim_real, ssim_imag, rrmse


def save_map_image(map2d, path, title, cmap="viridis", vmin=None, vmax=None):
    plt.figure(figsize=(5.5, 4.5))
    im = plt.imshow(map2d, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    bands = parse_bands(args.boundary_bands)

    config_path = os.path.join(args.exp_dir, "config.yml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f" Diagnostic Eval: {config.get('name', 'Unknown')}")
    print(f" Model Arch: {config['arch']}")
    print(f"{'=' * 60}\n")

    val_transform = Compose([Resize(config["input_h"], config["input_w"])])

    def build_path(base, fname):
        if os.path.isabs(fname):
            return fname
        return os.path.normpath(os.path.join(base, fname))

    selected_mode = None
    if args.new_data_dir is not None:
        path_real_img = os.path.join(args.new_data_dir, "chi0_all_real_64.mat")
        path_imag_img = os.path.join(args.new_data_dir, "chi0_all_imag_64.mat")
        path_real_lbl = os.path.join(args.new_data_dir, "chi_all_real_64.mat")
        path_imag_lbl = os.path.join(args.new_data_dir, "chi_all_imag_64.mat")
        if args.real_only:
            tmp_dir = os.path.join(args.exp_dir, "_tmp_real_only")
            if not os.path.exists(path_imag_img):
                path_imag_img = make_zero_mat_like(
                    path_real_img, os.path.join(tmp_dir, "auto_zero_imag_input_new_data.mat"), "chi0_all_imag"
                )
            if not os.path.exists(path_imag_lbl):
                path_imag_lbl = make_zero_mat_like(
                    path_real_lbl, os.path.join(tmp_dir, "auto_zero_imag_label_new_data.mat"), "chi_all_imag"
                )
        use_subset_split = False
        selected_mode = "new_data_full"
    else:
        if args.test_group == "measured_new":
            path_real_img = args.measured_real_img.strip()
            path_imag_img = args.measured_imag_img.strip()
            path_real_lbl = args.measured_real_lbl.strip()
            path_imag_lbl = args.measured_imag_lbl.strip()
            required_paths = {
                "--measured_real_img": path_real_img,
                "--measured_real_lbl": path_real_lbl,
            }
            if not args.real_only:
                required_paths["--measured_imag_img"] = path_imag_img
                required_paths["--measured_imag_lbl"] = path_imag_lbl
            missing = [k for k, v in required_paths.items() if not v]
            if missing:
                raise ValueError("Missing measured_new args: " + ", ".join(missing))
            if args.real_only:
                tmp_dir = os.path.join(args.exp_dir, "_tmp_real_only")
                if not path_imag_img:
                    path_imag_img = make_zero_mat_like(
                        path_real_img, os.path.join(tmp_dir, "auto_zero_imag_input.mat"), "chi0_all_imag"
                    )
                if not path_imag_lbl:
                    path_imag_lbl = make_zero_mat_like(
                        path_real_lbl, os.path.join(tmp_dir, "auto_zero_imag_label.mat"), "chi_all_imag"
                    )
            use_subset_split = False
            selected_mode = "measured_new"
        elif args.test_group == "combined":
            path_real_img = args.combined_real_img.strip()
            path_imag_img = args.combined_imag_img.strip()
            path_real_lbl = args.combined_real_lbl.strip()
            path_imag_lbl = args.combined_imag_lbl.strip()
            required_paths = {
                "--combined_real_img": path_real_img,
                "--combined_real_lbl": path_real_lbl,
            }
            if not args.real_only:
                required_paths["--combined_imag_img"] = path_imag_img
                required_paths["--combined_imag_lbl"] = path_imag_lbl
            missing = [k for k, v in required_paths.items() if not v]
            if missing:
                raise ValueError("Missing combined args: " + ", ".join(missing))
            if args.real_only:
                tmp_dir = os.path.join(args.exp_dir, "_tmp_real_only")
                if (not path_imag_img) or (not os.path.exists(path_imag_img)):
                    path_imag_img = make_zero_mat_like(
                        path_real_img, os.path.join(tmp_dir, "auto_zero_imag_input_combined.mat"), "chi0_all_imag"
                    )
                if (not path_imag_lbl) or (not os.path.exists(path_imag_lbl)):
                    path_imag_lbl = make_zero_mat_like(
                        path_real_lbl, os.path.join(tmp_dir, "auto_zero_imag_label_combined.mat"), "chi_all_imag"
                    )
            use_subset_split = False
            selected_mode = "combined"
        else:
            path_real_img = build_path(config["data_dir"], config["real_img_file"])
            path_imag_img = build_path(config["data_dir"], config["imag_img_file"])
            path_real_lbl = build_path(config["data_dir"], config["real_label_file"])
            path_imag_lbl = build_path(config["data_dir"], config["imag_label_file"])
            use_subset_split = True
            selected_mode = args.test_group

    for p in [path_real_img, path_imag_img, path_real_lbl, path_imag_lbl]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Dataset file not found: {p}")

    full_dataset = ComplexMatDataset(
        real_img_path=path_real_img,
        imag_img_path=path_imag_img,
        real_label_path=path_real_lbl,
        imag_label_path=path_imag_lbl,
        img_size=(config["original_img_size"], config["original_img_size"]),
        transform=None,
        normalize_method="z-score",
    )

    if args.norm_stats_file.strip():
        stats_path = args.norm_stats_file.strip()
        if not os.path.isabs(stats_path):
            stats_path = os.path.join(args.exp_dir, stats_path)
    else:
        stats_path = os.path.join(args.exp_dir, "norm_stats.json")
    if os.path.exists(stats_path):
        full_dataset.load_stats(stats_path)
    else:
        print(f"[WARN] Stats file not found: {stats_path}")

    if args.real_norm_mode == "current":
        cur_mean = float(np.mean(full_dataset.real_img))
        cur_std = float(np.std(full_dataset.real_img)) + 1e-8
        full_dataset.inp_real_mean = cur_mean
        full_dataset.inp_real_std = cur_std
        print(f"[INFO] real_norm_mode=current -> inp_real_mean={cur_mean:.6f}, inp_real_std={cur_std:.6f}")
    elif args.real_norm_mode == "none":
        full_dataset.inp_real_mean = 0.0
        full_dataset.inp_real_std = 1.0
        print("[INFO] real_norm_mode=none -> identity normalization for real input")
    else:
        print(f"[INFO] real_norm_mode=train -> using stats from: {stats_path}")

    if args.eval_split != "all":
        chosen_indices = get_indices_for_eval_split(args, config, len(full_dataset))
        final_dataset = TransformSubset(full_dataset, chosen_indices, val_transform)
        print(f"Dataset Split: {args.eval_split.upper()} ({len(final_dataset)} samples)")
    elif use_subset_split:
        indices = list(range(len(full_dataset)))
        np.random.seed(config["dataseed"])
        np.random.shuffle(indices)
        if selected_mode == "config_full":
            final_dataset = TransformSubset(full_dataset, indices, val_transform)
            print(f"Dataset Split: FULL ({len(final_dataset)} samples)")
        else:
            test_split = int(np.floor(0.1 * len(full_dataset)))
            test_indices = indices[:test_split]
            final_dataset = TransformSubset(full_dataset, test_indices, val_transform)
            print(f"Dataset Split: TEST10 ({len(final_dataset)} samples)")
    else:
        all_indices = list(range(len(full_dataset)))
        final_dataset = TransformSubset(full_dataset, all_indices, val_transform)
        print(f"Dataset Split: FULL NEW ({len(final_dataset)} samples)")

    test_loader = DataLoader(final_dataset, batch_size=1, shuffle=False)

    print(f"Loading model: {config['arch']}")
    model = archs.__dict__[config["arch"]](
        config["num_classes"],
        config["input_channels"],
        config["deep_supervision"],
        embed_dims=config["input_list"],
        kan_inner_layers=config.get("kan_inner_layers", 3),
        backbone_stages=config.get("backbone_stages", 3),
        stage4_depth=config.get("stage4_depth", 1),
        use_edge_residual_refine=config.get("use_edge_residual_refine", True),
        use_edge_multiscale=config.get("use_edge_multiscale", True),
        use_edge_sparse_focus=config.get("use_edge_sparse_focus", True),
        edge_focus_tau=config.get("edge_focus_tau", 0.25),
        edge_focus_gamma=config.get("edge_focus_gamma", 12.0),
        use_edge_center_boost=config.get("use_edge_center_boost", False),
        edge_center_boost=config.get("edge_center_boost", 0.6),
        edge_center_sigma=config.get("edge_center_sigma", 0.45),
        edge_refine_scale=config.get("edge_refine_scale", 0.25),
        edge_refine_mid=config.get("edge_refine_mid", 48),
        use_detail_skip_refine=config.get("use_detail_skip_refine", False),
        detail_refine_scale=config.get("detail_refine_scale", 0.1),
        detail_refine_mid=config.get("detail_refine_mid", 64),
        use_fourier_refine=config.get("use_fourier_refine", True),
        fourier_use_fft=config.get("fourier_use_fft", True),
        fourier_refine_scale=config.get("fourier_refine_scale", 0.12),
        fourier_refine_mid=config.get("fourier_refine_mid", 48),
        use_dual_ri_refine=config.get("use_dual_ri_refine", True),
        ri_refine_mid=config.get("ri_refine_mid", 48),
        ri_refine_scale=config.get("ri_refine_scale", 0.06),
    ).to(device)

    model_input_channels = int(config.get("input_channels", 2))
    model_num_classes = int(config.get("num_classes", 2))
    if model_num_classes == 1 and not args.real_only:
        print("[WARN] Single-channel output model detected, forcing --real_only behavior.")
        args.real_only = True

    model_path = os.path.join(args.exp_dir, args.checkpoint)
    if not os.path.exists(model_path):
        fallback = os.path.join(args.exp_dir, "model.pth")
        if os.path.exists(fallback):
            model_path = fallback
            print(f"[WARN] {args.checkpoint} not found, using model.pth")
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.exp_dir}")

    print(f"Loading weights from: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        print(f"[WARN] strict load failed, fallback strict=False: {e}")
        model.load_state_dict(state, strict=False)
    model.eval()

    save_dir_arg = args.save_dir.strip()
    if save_dir_arg:
        if os.path.isabs(save_dir_arg):
            save_full_path = _unique_dir(save_dir_arg)
        else:
            save_full_path = _unique_dir(os.path.join(args.exp_dir, save_dir_arg))
    else:
        ckpt_tag = _slugify_name(os.path.splitext(os.path.basename(args.checkpoint))[0])
        split_tag = _slugify_name(args.eval_split)
        save_leaf = f"diagnostic__ckpt_{ckpt_tag}__split_{split_tag}"
        save_full_path = _unique_dir(os.path.join(args.exp_dir, save_leaf))
    os.makedirs(save_full_path, exist_ok=False)
    print(f"[INFO] Saving diagnostics to: {save_full_path}")

    # Accumulators
    count = 0
    sum_mse_r = 0.0
    sum_mse_i = 0.0
    sum_ssim_r = 0.0
    sum_ssim_i = 0.0
    sum_rrmse = 0.0
    valid_i = 0
    valid_rrmse = 0

    sum_abs_err = None
    sum_ssim_map = None
    sum_gms_map = None
    sum_mag_abs_err = None
    sum_phase_abs_err = None

    band_sum = np.zeros(len(bands) - 1, dtype=np.float64)
    band_count = np.zeros(len(bands) - 1, dtype=np.int64)

    freq_rel_sum = None
    freq_rel_count = None

    shape_rows = []
    sample_rows = []

    print("Running diagnostic inference...")
    with torch.no_grad():
        for i, (input_tensor, target, _) in enumerate(tqdm(test_loader)):
            if args.max_cases > 0 and count >= args.max_cases:
                break
            input_tensor = input_tensor.to(device)
            target = target.to(device)
            if model_input_channels == 1 and input_tensor.shape[1] > 1:
                model_input = input_tensor[:, 0:1]
            else:
                model_input = input_tensor

            outputs = model(model_input)
            pred = outputs[-1] if isinstance(outputs, list) else outputs
            if pred.shape[1] == 1 and target.shape[1] > 1:
                target_eval = target[:, 0:1]
            else:
                target_eval = target[:, :pred.shape[1]]

            pred_np = pred.cpu().numpy().squeeze(0).astype(np.float32)
            target_np = target_eval.cpu().numpy().squeeze(0).astype(np.float32)

            abs_err = np.abs(pred_np - target_np)
            if sum_abs_err is None:
                sum_abs_err = np.zeros_like(abs_err, dtype=np.float64)
                sum_ssim_map = np.zeros_like(abs_err, dtype=np.float64)
                sum_gms_map = np.zeros_like(abs_err, dtype=np.float64)
                sum_mag_abs_err = np.zeros_like(abs_err[0], dtype=np.float64)
                sum_phase_abs_err = np.zeros_like(abs_err[0], dtype=np.float64)
            sum_abs_err += abs_err

            mse_r, mse_i, ssim_r, ssim_i, rrmse = calculate_metrics_detailed(pred_np, target_np)
            sum_mse_r += float(mse_r)
            sum_ssim_r += float(ssim_r)
            if not np.isnan(mse_i):
                sum_mse_i += float(mse_i)
                sum_ssim_i += float(ssim_i)
                valid_i += 1
            if not np.isnan(rrmse):
                sum_rrmse += float(rrmse)
                valid_rrmse += 1

            ch = pred_np.shape[0]
            ssim_ch = []
            for c in range(ch):
                s_score, s_map = ssim_score_and_map(target_np[c], pred_np[c])
                g_map = gms_map(target_np[c], pred_np[c], c=args.gms_c)
                sum_ssim_map[c] += s_map
                sum_gms_map[c] += g_map
                ssim_ch.append(s_score)

            combined_abs = abs_err[0] if ch == 1 else abs_err.mean(axis=0)

            bmask = boundary_mask_from_target(target_np, args.edge_percentile)
            dist_to_b = distance_transform_edt(~bmask)
            for bi in range(len(bands) - 1):
                lo, hi = bands[bi], bands[bi + 1]
                m = (dist_to_b >= lo) & (dist_to_b < hi)
                cnt = int(m.sum())
                if cnt > 0:
                    band_sum[bi] += float(combined_abs[m].sum())
                    band_count[bi] += cnt

            gt_mag = complex_mag(target_np)
            pd_mag = complex_mag(pred_np)
            mag_abs = np.abs(pd_mag - gt_mag)
            sum_mag_abs_err += mag_abs

            if ch >= 2:
                gt_phase = complex_phase(target_np)
                pd_phase = complex_phase(pred_np)
                p_abs = np.abs(wrapped_phase_diff(pd_phase, gt_phase))
            else:
                p_abs = np.zeros_like(gt_mag, dtype=np.float32)
            sum_phase_abs_err += p_abs

            gt_mask, gt_thr = mask_from_mag(gt_mag, args.shape_thresh_method, args.shape_percentile)
            pd_mask = largest_component(pd_mag > gt_thr)
            dice, iou, area_p, area_g = dice_iou(pd_mask, gt_mask)
            area_rel = abs(area_p - area_g) / max(area_g, 1)

            c_gt = centroid(gt_mask)
            c_pd = centroid(pd_mask)
            if c_gt is None and c_pd is None:
                cen_err = 0.0
            elif c_gt is None or c_pd is None:
                cen_err = float(np.sqrt(gt_mag.shape[0] ** 2 + gt_mag.shape[1] ** 2))
            else:
                cen_err = float(np.sqrt((c_gt[0] - c_pd[0]) ** 2 + (c_gt[1] - c_pd[1]) ** 2))
            diag = float(np.sqrt(gt_mag.shape[0] ** 2 + gt_mag.shape[1] ** 2))
            cen_err_norm = cen_err / max(diag, 1e-6)

            gt_edge = mask_boundary(gt_mask)
            pd_edge = mask_boundary(pd_mask)
            pfom = pratt_fom(pd_edge, gt_edge)

            rp_gt = radial_power_spectrum(gt_mag)
            rp_pd = radial_power_spectrum(pd_mag)
            rel = np.abs(rp_pd - rp_gt) / (rp_gt + 1e-8)
            if freq_rel_sum is None:
                freq_rel_sum = np.zeros_like(rel, dtype=np.float64)
                freq_rel_count = np.zeros_like(rel, dtype=np.int64)
            n = min(len(freq_rel_sum), len(rel))
            freq_rel_sum[:n] += rel[:n]
            freq_rel_count[:n] += 1

            phase_mae = float(np.mean(p_abs))
            phase_rmse = float(np.sqrt(np.mean(p_abs ** 2)))
            mag_mae = float(np.mean(mag_abs))
            mag_mse = float(np.mean((pd_mag - gt_mag) ** 2))

            shape_rows.append({
                "id": i,
                "dice": dice,
                "iou": iou,
                "area_pred": area_p,
                "area_gt": area_g,
                "area_rel_error": area_rel,
                "centroid_error_px": cen_err,
                "centroid_error_norm": cen_err_norm,
                "pratt_fom": pfom,
            })

            sample_rows.append({
                "id": i,
                "mse_real": float(mse_r),
                "mse_imag": float(mse_i) if not np.isnan(mse_i) else np.nan,
                "ssim_real": float(ssim_r),
                "ssim_imag": float(ssim_i) if not np.isnan(ssim_i) else np.nan,
                "rrmse": float(rrmse) if not np.isnan(rrmse) else np.nan,
                "combined_mae": float(np.mean(combined_abs)),
                "mag_mae": mag_mae,
                "mag_mse": mag_mse,
                "phase_mae_rad": phase_mae,
                "phase_rmse_rad": phase_rmse,
                "dice": dice,
                "iou": iou,
                "centroid_error_px": cen_err,
                "pratt_fom": pfom,
                "mean_ssim_map": float(np.mean([v for v in ssim_ch if not np.isnan(v)])),
            })

            count += 1

    if count == 0:
        raise RuntimeError("No samples were evaluated. Please check dataset and --max_cases.")

    mean_abs_err = (sum_abs_err / count).astype(np.float32)
    mean_ssim_map = (sum_ssim_map / count).astype(np.float32)
    mean_gms_map = (sum_gms_map / count).astype(np.float32)
    mean_mag_abs = (sum_mag_abs_err / count).astype(np.float32)
    mean_phase_abs = (sum_phase_abs_err / count).astype(np.float32)

    os.makedirs(os.path.join(save_full_path, "maps"), exist_ok=True)
    np.save(os.path.join(save_full_path, "maps", "mean_abs_err.npy"), mean_abs_err)
    np.save(os.path.join(save_full_path, "maps", "mean_ssim_map.npy"), mean_ssim_map)
    np.save(os.path.join(save_full_path, "maps", "mean_gms_map.npy"), mean_gms_map)
    np.save(os.path.join(save_full_path, "maps", "mean_mag_abs_err.npy"), mean_mag_abs)
    np.save(os.path.join(save_full_path, "maps", "mean_phase_abs_err.npy"), mean_phase_abs)

    for c in range(mean_ssim_map.shape[0]):
        save_map_image(
            mean_ssim_map[c],
            os.path.join(save_full_path, "maps", f"mean_ssim_map_ch{c}.png"),
            f"Mean SSIM Map (ch{c})",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        save_map_image(
            mean_gms_map[c],
            os.path.join(save_full_path, "maps", f"mean_gms_map_ch{c}.png"),
            f"Mean GMS Map (ch{c})",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        save_map_image(
            mean_abs_err[c],
            os.path.join(save_full_path, "maps", f"mean_abs_err_ch{c}.png"),
            f"Mean Abs Error (ch{c})",
            cmap="hot",
        )
    save_map_image(mean_mag_abs, os.path.join(save_full_path, "maps", "mean_mag_abs_err.png"),
                   "Mean |chi| Abs Error", cmap="hot")
    save_map_image(mean_phase_abs, os.path.join(save_full_path, "maps", "mean_phase_abs_err.png"),
                   "Mean Phase Abs Error (rad)", cmap="viridis")

    band_mae = np.divide(band_sum, np.maximum(band_count, 1), where=np.ones_like(band_sum, dtype=bool))
    with open(os.path.join(save_full_path, "boundary_band_mae.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["band_start_px", "band_end_px", "pixel_count", "mean_abs_error"])
        for bi in range(len(bands) - 1):
            w.writerow([bands[bi], bands[bi + 1], int(band_count[bi]), float(band_mae[bi])])
    centers = 0.5 * (np.array(bands[:-1]) + np.array(bands[1:]))
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(centers, band_mae, marker="o")
    plt.xlabel("Distance to GT Boundary (px)")
    plt.ylabel("Mean Abs Error")
    plt.title("Boundary-Band Error Profile")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_full_path, "boundary_band_mae.png"), dpi=300)
    plt.close()

    freq_rel = np.divide(freq_rel_sum, np.maximum(freq_rel_count, 1), where=np.ones_like(freq_rel_sum, dtype=bool))
    with open(os.path.join(save_full_path, "frequency_radial_relative_error.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["radius_bin", "relative_error"])
        for r, v in enumerate(freq_rel.tolist()):
            w.writerow([r, float(v)])
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(np.arange(len(freq_rel)), freq_rel)
    plt.xlabel("Radial Frequency Bin")
    plt.ylabel("Relative Spectral Error")
    plt.title("Radial Frequency Error Spectrum")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_full_path, "frequency_radial_relative_error.png"), dpi=300)
    plt.close()

    shape_csv = os.path.join(save_full_path, "shape_position_metrics_per_sample.csv")
    with open(shape_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(shape_rows[0].keys()))
        w.writeheader()
        w.writerows(shape_rows)

    sample_csv = os.path.join(save_full_path, "diagnostic_metrics_per_sample.csv")
    with open(sample_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        w.writeheader()
        w.writerows(sample_rows)

    hard_sorted = sorted(sample_rows, key=lambda x: x["combined_mae"], reverse=True)
    topk = hard_sorted[:max(1, args.topk_hard)]
    with open(os.path.join(save_full_path, "top_hard_samples.json"), "w", encoding="utf-8") as f:
        json.dump(topk, f, indent=2, ensure_ascii=False)

    def mean_of(key, rows):
        vals = [float(r[key]) for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "count": count,
        "data_mode": selected_mode,
        "checkpoint": args.checkpoint,
        "eval_split": args.eval_split,
        "real_norm_mode": args.real_norm_mode,
        "basic_metrics": {
            "mean_mse_real": sum_mse_r / count,
            "mean_ssim_real": sum_ssim_r / count,
            "mean_mse_imag": (sum_mse_i / valid_i) if valid_i > 0 else None,
            "mean_ssim_imag": (sum_ssim_i / valid_i) if valid_i > 0 else None,
            "mean_rrmse": (sum_rrmse / valid_rrmse) if valid_rrmse > 0 else None,
        },
        "boundary_band_mae": [
            {
                "start_px": float(bands[i]),
                "end_px": float(bands[i + 1]),
                "count": int(band_count[i]),
                "mae": float(band_mae[i]),
            }
            for i in range(len(bands) - 1)
        ],
        "shape_position_summary": {
            "dice_mean": mean_of("dice", shape_rows),
            "iou_mean": mean_of("iou", shape_rows),
            "centroid_error_px_mean": mean_of("centroid_error_px", shape_rows),
            "area_rel_error_mean": mean_of("area_rel_error", shape_rows),
            "pratt_fom_mean": mean_of("pratt_fom", shape_rows),
        },
        "complex_decomp_summary": {
            "mag_mae_mean": mean_of("mag_mae", sample_rows),
            "mag_mse_mean": mean_of("mag_mse", sample_rows),
            "phase_mae_rad_mean": mean_of("phase_mae_rad", sample_rows),
            "phase_rmse_rad_mean": mean_of("phase_rmse_rad", sample_rows),
        },
        "frequency_spectrum_summary": {
            "low_bin_mean_rel_err": float(np.mean(freq_rel[:max(1, len(freq_rel) // 8)])),
            "mid_bin_mean_rel_err": float(np.mean(freq_rel[max(1, len(freq_rel) // 8):max(2, len(freq_rel) // 3)])),
            "high_bin_mean_rel_err": float(np.mean(freq_rel[max(2, len(freq_rel) // 3):])),
        },
    }
    with open(os.path.join(save_full_path, "diagnostic_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(save_full_path, "diagnostic_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Diagnostic Summary\n")
        f.write("==================\n")
        f.write(f"Count: {summary['count']}\n")
        f.write(f"Mode: {summary['data_mode']}\n")
        f.write(f"Checkpoint: {summary['checkpoint']}\n")
        f.write(f"Split: {summary['eval_split']}\n")
        f.write(f"Norm: {summary['real_norm_mode']}\n\n")
        f.write("[Basic]\n")
        f.write(f"Mean MSE Real: {summary['basic_metrics']['mean_mse_real']:.6f}\n")
        f.write(f"Mean SSIM Real: {summary['basic_metrics']['mean_ssim_real']:.6f}\n")
        if summary["basic_metrics"]["mean_mse_imag"] is not None:
            f.write(f"Mean MSE Imag: {summary['basic_metrics']['mean_mse_imag']:.6f}\n")
            f.write(f"Mean SSIM Imag: {summary['basic_metrics']['mean_ssim_imag']:.6f}\n")
        if summary["basic_metrics"]["mean_rrmse"] is not None:
            f.write(f"Mean RRMSE: {summary['basic_metrics']['mean_rrmse']:.6f}\n")
        f.write("\n[Shape/Position]\n")
        f.write(f"Dice Mean: {summary['shape_position_summary']['dice_mean']:.6f}\n")
        f.write(f"IoU Mean: {summary['shape_position_summary']['iou_mean']:.6f}\n")
        f.write(f"Centroid Error Mean (px): {summary['shape_position_summary']['centroid_error_px_mean']:.6f}\n")
        f.write(f"Area Relative Error Mean: {summary['shape_position_summary']['area_rel_error_mean']:.6f}\n")
        f.write(f"Pratt FOM Mean: {summary['shape_position_summary']['pratt_fom_mean']:.6f}\n")
        f.write("\n[Complex Decomposition]\n")
        f.write(f"|chi| MAE Mean: {summary['complex_decomp_summary']['mag_mae_mean']:.6f}\n")
        f.write(f"|chi| MSE Mean: {summary['complex_decomp_summary']['mag_mse_mean']:.6f}\n")
        f.write(f"Phase MAE Mean (rad): {summary['complex_decomp_summary']['phase_mae_rad_mean']:.6f}\n")
        f.write(f"Phase RMSE Mean (rad): {summary['complex_decomp_summary']['phase_rmse_rad_mean']:.6f}\n")
        f.write("\n[Frequency Radial Relative Error]\n")
        f.write(f"Low bins mean: {summary['frequency_spectrum_summary']['low_bin_mean_rel_err']:.6f}\n")
        f.write(f"Mid bins mean: {summary['frequency_spectrum_summary']['mid_bin_mean_rel_err']:.6f}\n")
        f.write(f"High bins mean: {summary['frequency_spectrum_summary']['high_bin_mean_rel_err']:.6f}\n")

    print("\n[Done] Diagnostic files generated:")
    print(f"- {txt_path}")
    print(f"- {os.path.join(save_full_path, 'diagnostic_summary.json')}")
    print(f"- {sample_csv}")
    print(f"- {shape_csv}")
    print(f"- {os.path.join(save_full_path, 'boundary_band_mae.csv')}")
    print(f"- {os.path.join(save_full_path, 'frequency_radial_relative_error.csv')}")
    print(f"- {os.path.join(save_full_path, 'maps')}")


if __name__ == "__main__":
    main()
