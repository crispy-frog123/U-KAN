"""Evaluate a baseline U-KAN checkpoint on the saved 10% test split."""

import argparse
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import yaml
from albumentations import Compose, Resize
from skimage.metrics import structural_similarity as ssim_func
from torch.utils.data import DataLoader
from tqdm import tqdm

import archs
from config import baseline_config
from dataset import ComplexMatDataset, TransformSubset
from utils import AverageMeter


def parse_args(argv=None):
    """Parse evaluation options with a runnable baseline default."""
    defaults = baseline_config()
    parser = argparse.ArgumentParser(
        description="Evaluate U-KAN on the baseline config_test10 split."
    )
    parser.add_argument(
        "--exp_dir",
        default=os.path.join(defaults["output_dir"], "baseline"),
        help="experiment directory containing config.yml, norm_stats.json, and checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        default="model_best_ssim.pth",
        help="checkpoint filename or absolute path",
    )
    parser.add_argument(
        "--save_dir",
        default="",
        help="optional output directory; defaults to a results folder under exp_dir",
    )
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--new_data_dir", default=None)
    parser.add_argument(
        "--test_group",
        default="config_test10",
        choices=["config_test10", "config_full", "measured_new", "combined"],
    )
    parser.add_argument("--measured_real_img", default=r"inputs\input\chi0_all_real_new_64.mat")
    parser.add_argument("--measured_imag_img", default="")
    parser.add_argument("--measured_real_lbl", default=r"inputs\label\chi_all_real_new_64.mat")
    parser.add_argument("--measured_imag_lbl", default="")
    parser.add_argument("--combined_real_img", default=r"inputs\input\chi0_all_real_combine.mat")
    parser.add_argument("--combined_imag_img", default=r"inputs\input\chi0_all_imag_combine.mat")
    parser.add_argument("--combined_real_lbl", default=r"inputs\label\chi_all_real_combine.mat")
    parser.add_argument("--combined_imag_lbl", default=r"inputs\label\chi_all_imag_combine.mat")
    parser.add_argument("--real_only", action="store_true")
    parser.add_argument(
        "--real_norm_mode",
        default="train",
        choices=["train", "current", "none"],
    )
    parser.add_argument("--norm_stats_file", default="")
    parser.add_argument(
        "--eval_split",
        default="all",
        choices=["all", "train", "val", "test"],
    )
    parser.add_argument("--save_pre_tail", default=0, type=int, choices=[0, 1])
    return parser.parse_args(argv)


def unique_dir(path):
    """Return a non-existing directory path by appending a numeric suffix."""
    path = Path(path)
    if not path.exists():
        return path
    for idx in range(2, 10_000):
        candidate = path.with_name(f"{path.name}__run{idx}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique output directory under {path.parent}")


def load_config(exp_dir):
    """Load run config, falling back to embedded baseline defaults for clean repos."""
    config_path = Path(exp_dir) / "config.yml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return baseline_config()


def build_path(base_dir, file_path):
    """Resolve dataset file paths from config.yml."""
    if os.path.isabs(file_path):
        return file_path
    return os.path.normpath(os.path.join(base_dir, file_path))


def make_zero_mat_like(real_mat_path, save_path, key_name):
    """Create a zero-valued MAT file for real-only compatibility modes."""
    data = sio.loadmat(real_mat_path)
    keys = [key for key in data if not key.startswith("__")]
    zeros = np.zeros_like(data[keys[0]], dtype=np.float32)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(save_path, {key_name: zeros})
    return str(save_path)


def build_test_indices(total_n, config):
    """Reconstruct the baseline test indices: shuffled first 10% samples."""
    indices = list(range(total_n))
    rng = np.random.RandomState(int(config["dataseed"]))
    rng.shuffle(indices)
    test_count = int(np.floor(0.1 * total_n))
    return indices[:test_count]


def split_indices(total_n, config):
    """Rebuild baseline train/val/test splits for eval_split compatibility."""
    indices = list(range(total_n))
    rng = np.random.RandomState(int(config["dataseed"]))
    rng.shuffle(indices)
    test_count = int(np.floor(0.1 * total_n))
    val_count = int(np.floor(0.1 * total_n))
    return {
        "train": indices[test_count + val_count :],
        "val": indices[test_count : test_count + val_count],
        "test": indices[:test_count],
        "all": indices,
    }


def resolve_dataset_paths(config, args, exp_dir):
    """Resolve baseline and compatibility evaluation data paths."""
    if args.new_data_dir:
        base = Path(args.new_data_dir)
        return (
            str(base / "chi0_all_real_64.mat"),
            str(base / "chi0_all_imag_64.mat"),
            str(base / "chi_all_real_64.mat"),
            str(base / "chi_all_imag_64.mat"),
            "new_data_full",
        )
    if args.test_group == "measured_new":
        return (
            args.measured_real_img,
            args.measured_imag_img,
            args.measured_real_lbl,
            args.measured_imag_lbl,
            "measured_new",
        )
    if args.test_group == "combined":
        return (
            args.combined_real_img,
            args.combined_imag_img,
            args.combined_real_lbl,
            args.combined_imag_lbl,
            "combined",
        )
    return (
        build_path(config["data_dir"], config["real_img_file"]),
        build_path(config["data_dir"], config["imag_img_file"]),
        build_path(config["data_dir"], config["real_label_file"]),
        build_path(config["data_dir"], config["imag_label_file"]),
        args.test_group,
    )


def make_dataset(config, args, exp_dir):
    """Create a normalized evaluation dataset while defaulting to config_test10."""
    real_img_path, imag_img_path, real_label_path, imag_label_path, data_mode = (
        resolve_dataset_paths(config, args, exp_dir)
    )

    if args.real_only:
        tmp_dir = Path(exp_dir) / "_tmp_real_only"
        if not imag_img_path or not os.path.exists(imag_img_path):
            imag_img_path = make_zero_mat_like(
                real_img_path,
                tmp_dir / "auto_zero_imag_input.mat",
                "chi0_all_imag",
            )
        if not imag_label_path or not os.path.exists(imag_label_path):
            imag_label_path = make_zero_mat_like(
                real_label_path,
                tmp_dir / "auto_zero_imag_label.mat",
                "chi_all_imag",
            )

    for path in (real_img_path, imag_img_path, real_label_path, imag_label_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

    dataset = ComplexMatDataset(
        real_img_path=real_img_path,
        imag_img_path=imag_img_path,
        real_label_path=real_label_path,
        imag_label_path=imag_label_path,
        img_size=(config["original_img_size"], config["original_img_size"]),
        normalize_method="z-score",
    )

    stats_path = Path(args.norm_stats_file) if args.norm_stats_file else Path(exp_dir) / "norm_stats.json"
    if args.norm_stats_file and not stats_path.is_absolute():
        stats_path = Path(exp_dir) / stats_path
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {stats_path}. Run train.py first or provide exp_dir."
        )
    dataset.load_stats(stats_path)

    if args.real_norm_mode == "current":
        dataset.inp_real_mean = float(np.mean(dataset.real_img))
        dataset.inp_real_std = float(np.std(dataset.real_img)) + 1e-8
    elif args.real_norm_mode == "none":
        dataset.inp_real_mean = 0.0
        dataset.inp_real_std = 1.0

    eval_transform = Compose([Resize(config["input_h"], config["input_w"])])
    if args.eval_split != "all":
        indices = split_indices(len(dataset), config)[args.eval_split]
    elif data_mode == "config_test10":
        indices = build_test_indices(len(dataset), config)
    else:
        indices = list(range(len(dataset)))
    return TransformSubset(dataset, indices, eval_transform), data_mode


def build_model(config, device):
    """Instantiate the same model topology used by training."""
    return archs.__dict__[config["arch"]](
        config["num_classes"],
        config["input_channels"],
        config["deep_supervision"],
        embed_dims=config["input_list"],
        no_kan=config.get("no_kan", False),
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


def load_checkpoint(model, checkpoint_path, device):
    """Load plain state_dict checkpoints and DataParallel-prefixed checkpoints."""
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state)


def safe_data_range(array):
    """Return a non-zero dynamic range for SSIM."""
    data_range = float(array.max() - array.min())
    return data_range if data_range > 0 else 1e-6


def calculate_metrics(pred, target):
    """Compute real/imag MSE, SSIM, and complex-magnitude RRMSE."""
    mse_real = float(np.mean((pred[0] - target[0]) ** 2))
    mse_imag = float(np.mean((pred[1] - target[1]) ** 2))
    ssim_real = float(
        ssim_func(target[0], pred[0], data_range=safe_data_range(target[0]))
    )
    ssim_imag = float(
        ssim_func(target[1], pred[1], data_range=safe_data_range(target[1]))
    )

    eps_target_real = target[0] + 1.0
    eps_target_mag = np.sqrt(eps_target_real**2 + target[1] ** 2)
    diff_mag = np.sqrt((pred[0] - target[0]) ** 2 + (pred[1] - target[1]) ** 2)
    rrmse = float(np.sqrt(np.mean((diff_mag / (eps_target_mag + 1e-8)) ** 2)))
    return mse_real, mse_imag, ssim_real, ssim_imag, rrmse


def save_metrics(save_dir, results, count, checkpoint_name, data_mode, real_only):
    """Write the same core metrics reported by the baseline output folder."""
    with open(Path(save_dir) / "metrics.txt", "w", encoding="utf-8") as handle:
        handle.write("Evaluation Report\n")
        handle.write("=================\n")
        handle.write(f"Data Mode: {data_mode}\n")
        handle.write(f"Real Only: {real_only}\n")
        handle.write(f"Model: {checkpoint_name}\n")
        handle.write(f"Count: {count}\n\n")
        handle.write(f"Mean MSE:  {results['mean_mse']:.6f}\n")
        handle.write(f"Real SSIM: {results['ssim_real']:.6f}\n")
        handle.write(f"Avg SSIM:  {results['avg_ssim']:.6f}\n")
        if not real_only:
            handle.write(f"Imag SSIM: {results['ssim_imag']:.6f}\n")
            handle.write(f"RRMSE:     {results['rrmse']:.6f}\n")


def plot_error_heatmap(save_dir, mean_abs_err, real_only=False):
    """Save per-pixel MAE heatmaps for quick spatial error inspection."""
    if real_only:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
        im = ax.imshow(mean_abs_err[0], cmap="hot")
        ax.set_title("Mean Abs Error (Real)")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(Path(save_dir) / "pixel_mae_heatmap.png", dpi=300)
        plt.close(fig)
        return

    combined = mean_abs_err.mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, image, title in zip(
        axes,
        (mean_abs_err[0], mean_abs_err[1], combined),
        ("Mean Abs Error (Real)", "Mean Abs Error (Imag)", "Mean Abs Error (Combined)"),
    ):
        im = ax.imshow(image, cmap="hot")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "pixel_mae_heatmap.png", dpi=300)
    plt.close(fig)


def plot_samples(samples, save_dir, real_only=False):
    """Save up to ten real/imag target-vs-prediction examples."""
    if not samples:
        return

    n_cols = 2 if real_only else 4
    fig, axes = plt.subplots(len(samples), n_cols, figsize=(5.5 * n_cols, 4 * len(samples)))
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    titles = ("GT Real", "Pred Real") if real_only else (
        "GT Real",
        "Pred Real",
        "GT |Imag|",
        "Pred |Imag|",
    )
    for row, sample in enumerate(samples):
        target = sample["target"]
        pred = sample["pred"]
        real_range = (
            min(target[0].min(), pred[0].min()),
            max(target[0].max(), pred[0].max()),
        )
        if real_only:
            images = (target[0], pred[0])
            ranges = (real_range, real_range)
        else:
            imag_range = (0.0, max(np.abs(target[1]).max(), np.abs(pred[1]).max()))
            images = (target[0], pred[0], np.abs(target[1]), np.abs(pred[1]))
            ranges = (real_range, real_range, imag_range, imag_range)

        for col, (image, value_range) in enumerate(zip(images, ranges)):
            ax = axes[row, col]
            im = ax.imshow(image, cmap="jet", vmin=value_range[0], vmax=value_range[1])
            if row == 0:
                ax.set_title(titles[col], fontsize=14, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[row, 0].set_ylabel(f"ID {sample['id']}", fontsize=12, fontweight="bold")
        axes[row, 1].text(
            0.5,
            -0.1,
            f"SSIM: {sample['ssim_real']:.2f}",
            transform=axes[row, 1].transAxes,
            ha="center",
            fontsize=12,
            color="red",
        )
        if not real_only:
            axes[row, 3].text(
                0.5,
                -0.1,
                f"SSIM: {sample['ssim_imag']:.2f}",
                transform=axes[row, 3].transAxes,
                ha="center",
                fontsize=12,
                color="blue",
            )

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "comparison_vis_abs_bold.png", dpi=300)
    plt.close(fig)


def evaluate(args):
    """Run checkpoint inference and save metrics plus compact diagnostics."""
    exp_dir = Path(args.exp_dir)
    config = load_config(exp_dir)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = exp_dir / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.save_pre_tail:
        print("[WARN] --save_pre_tail is kept for CLI compatibility but is not used by the clean evaluator.")

    dataset, data_mode = make_dataset(config, args, exp_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(config, device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    save_dir = Path(args.save_dir) if args.save_dir else unique_dir(
        exp_dir / f"results_config_test10__ckpt_{checkpoint_path.stem}"
    )
    save_dir.mkdir(parents=True, exist_ok=False)

    meters = {
        "mse_real": AverageMeter(),
        "mse_imag": AverageMeter(),
        "ssim_real": AverageMeter(),
        "ssim_imag": AverageMeter(),
        "rrmse": AverageMeter(),
    }
    sum_abs_err = None
    sum_sq_err = None
    sample_count = 0

    rng = random.Random(int(config["dataseed"]))
    sample_positions = set(rng.sample(range(len(dataset)), min(10, len(dataset))))
    saved_samples = []

    with torch.no_grad():
        for input_tensor, target, sample_ids in tqdm(loader, desc="Evaluation"):
            input_tensor = input_tensor.to(device)
            target = target.to(device)
            output = model(input_tensor)
            pred = output[-1] if isinstance(output, list) else output

            pred_np_batch = pred.detach().cpu().numpy()
            target_np_batch = target.detach().cpu().numpy()
            for item_idx in range(pred_np_batch.shape[0]):
                pred_np = pred_np_batch[item_idx]
                target_np = target_np_batch[item_idx]
                mse_r, mse_i, ssim_r, ssim_i, rrmse = calculate_metrics(pred_np, target_np)
                meters["mse_real"].update(mse_r)
                meters["ssim_real"].update(ssim_r)
                if not args.real_only:
                    meters["mse_imag"].update(mse_i)
                    meters["ssim_imag"].update(ssim_i)
                    meters["rrmse"].update(rrmse)

                abs_err = np.abs(pred_np - target_np)
                sq_err = (pred_np - target_np) ** 2
                sum_abs_err = abs_err if sum_abs_err is None else sum_abs_err + abs_err
                sum_sq_err = sq_err if sum_sq_err is None else sum_sq_err + sq_err

                if sample_count in sample_positions:
                    saved_samples.append(
                        {
                            "id": int(sample_ids[item_idx]),
                            "pred": pred_np,
                            "target": target_np,
                            "ssim_real": ssim_r,
                            "ssim_imag": ssim_i,
                        }
                    )
                sample_count += 1

    mean_abs_err = (sum_abs_err / max(1, sample_count)).astype(np.float32)
    mean_sq_err = (sum_sq_err / max(1, sample_count)).astype(np.float32)
    np.save(save_dir / "pixel_mae_map.npy", mean_abs_err)
    np.save(save_dir / "pixel_mse_map.npy", mean_sq_err)

    if args.real_only:
        results = {
            "mean_mse": meters["mse_real"].avg,
            "ssim_real": meters["ssim_real"].avg,
            "ssim_imag": float("nan"),
            "avg_ssim": meters["ssim_real"].avg,
            "rrmse": float("nan"),
        }
    else:
        results = {
            "mean_mse": (meters["mse_real"].avg + meters["mse_imag"].avg) / 2,
            "ssim_real": meters["ssim_real"].avg,
            "ssim_imag": meters["ssim_imag"].avg,
            "avg_ssim": (meters["ssim_real"].avg + meters["ssim_imag"].avg) / 2,
            "rrmse": meters["rrmse"].avg,
        }
    save_metrics(save_dir, results, sample_count, checkpoint_path.name, data_mode, args.real_only)
    plot_error_heatmap(save_dir, mean_abs_err, real_only=args.real_only)
    plot_samples(saved_samples, save_dir, real_only=args.real_only)

    print("\nFinal Results")
    print(f"  Mean MSE:  {results['mean_mse']:.6f}")
    print(f"  Real SSIM: {results['ssim_real']:.6f}")
    print(f"  Avg SSIM:  {results['avg_ssim']:.6f}")
    if not args.real_only:
        print(f"  Imag SSIM: {results['ssim_imag']:.6f}")
        print(f"  RRMSE:     {results['rrmse']:.6f}")
    print(f"  Saved to:  {save_dir}")


def main(argv=None):
    """CLI entrypoint."""
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
