"""Export random samples and baseline model predictions.

The script saves per-sample input/prediction visualizations and a small MAT
dataset with the same file layout, variable names, and ``4096 x N`` orientation
as the training data under ``inputs/``.
"""

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
import yaml

import archs
from config import baseline_config
from dataset import ComplexMatDataset


def parse_args(argv=None):
    """Parse export options."""
    defaults = baseline_config()
    parser = argparse.ArgumentParser(description="Export random dataset samples.")
    parser.add_argument("--output_dir", default="sample_exports/random10_model_outputs")
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--seed", default=2981, type=int)
    parser.add_argument("--dpi", default=300, type=int)
    parser.add_argument("--exp_dir", default="outputs/baseline")
    parser.add_argument("--checkpoint", default="model_best_ssim.pth")
    parser.add_argument("--data_dir", default=defaults["data_dir"])
    parser.add_argument("--real_img_file", default=defaults["real_img_file"])
    parser.add_argument("--imag_img_file", default=defaults["imag_img_file"])
    parser.add_argument("--real_label_file", default=defaults["real_label_file"])
    parser.add_argument("--imag_label_file", default=defaults["imag_label_file"])
    parser.add_argument("--original_img_size", default=defaults["original_img_size"], type=int)
    return parser.parse_args(argv)


def resolve_path(base_dir, file_path):
    """Resolve a dataset path while allowing absolute file overrides."""
    path = Path(file_path)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def load_config(exp_dir):
    """Load experiment config with baseline defaults as a clean-repo fallback."""
    config_path = Path(exp_dir) / "config.yml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return baseline_config()


def build_model(config, device):
    """Instantiate the baseline model topology."""
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
    """Load a plain state_dict checkpoint."""
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state)


def save_pair_figure(left, right, titles, save_path, dpi):
    """Save real/imag components side by side with independent colorbars."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), constrained_layout=True)
    for ax, image, title in zip(axes, (left, right), titles):
        im = ax.imshow(image, cmap="jet")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def save_hdf5_mat(path, key, samples):
    """Save samples as a v7.3-style HDF5 MAT dataset shaped 4096 x N."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = samples.reshape(samples.shape[0], -1).T
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=flat)


def save_raw_dataset(output_dir, selected):
    """Write sampled raw inputs/labels using the same layout as ``inputs/``."""
    input_dir = output_dir / "inputs" / "input"
    label_dir = output_dir / "inputs" / "label"
    save_hdf5_mat(input_dir / "chi0_all_real_mnist.mat", "chi0_all_real", selected["input_real"])
    save_hdf5_mat(input_dir / "chi0_all_imag_mnist.mat", "chi0_all_imag", selected["input_imag"])
    save_hdf5_mat(label_dir / "chi_all_real_mnist.mat", "chi_all_real", selected["target_real"])
    save_hdf5_mat(label_dir / "chi_all_imag_mnist.mat", "chi_all_imag", selected["target_imag"])


def main(argv=None):
    """Export random samples, model predictions, and a raw mini dataset."""
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_image_dir = output_dir / "input_imag"
    pred_image_dir = output_dir / "pred_imag"

    # Remove the older per-sample image folders so the export stays flat.
    for old_dir in output_dir.glob("sample_*"):
        if old_dir.is_dir():
            shutil.rmtree(old_dir)
    input_image_dir.mkdir(parents=True, exist_ok=True)
    pred_image_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = Path(args.exp_dir)

    config = load_config(exp_dir)

    dataset = ComplexMatDataset(
        real_img_path=resolve_path(args.data_dir, args.real_img_file),
        imag_img_path=resolve_path(args.data_dir, args.imag_img_file),
        real_label_path=resolve_path(args.data_dir, args.real_label_file),
        imag_label_path=resolve_path(args.data_dir, args.imag_label_file),
        img_size=(args.original_img_size, args.original_img_size),
        normalize_method="z-score",
    )
    stats_path = exp_dir / "norm_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
    dataset.load_stats(stats_path)

    if args.num_samples > len(dataset):
        raise ValueError(f"num_samples={args.num_samples} exceeds dataset size {len(dataset)}")

    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(dataset), size=args.num_samples, replace=False))

    selected_input_real = dataset.real_img[indices]
    selected_input_imag = dataset.imag_img[indices]
    selected_target_real = dataset.real_label[indices]
    selected_target_imag = dataset.imag_label[indices]

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = exp_dir / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    normalized_inputs = []
    for data_idx in indices:
        input_tensor, _, _ = dataset[int(data_idx)]
        normalized_inputs.append(input_tensor)
    batch = torch.stack(normalized_inputs, dim=0).to(device)

    with torch.no_grad():
        prediction = model(batch)
        prediction = prediction[-1] if isinstance(prediction, list) else prediction
    selected_output = prediction.detach().cpu().numpy().astype(np.float32)
    selected_output_real = selected_output[:, 0]
    selected_output_imag = selected_output[:, 1]

    for sample_idx, data_idx in enumerate(indices):
        save_pair_figure(
            selected_input_real[sample_idx],
            selected_input_imag[sample_idx],
            ("Input Real", "Input Imag"),
            input_image_dir / f"sample_{int(data_idx):04d}.png",
            args.dpi,
        )
        save_pair_figure(
            selected_output_real[sample_idx],
            selected_output_imag[sample_idx],
            ("Output Real", "Output Imag"),
            pred_image_dir / f"sample_{int(data_idx):04d}.png",
            args.dpi,
        )

    save_raw_dataset(
        output_dir,
        {
            "input_real": selected_input_real,
            "input_imag": selected_input_imag,
            "target_real": selected_target_real,
            "target_imag": selected_target_imag,
        },
    )
    indices_path = output_dir / "selected_indices.txt"
    indices_path.write_text("\n".join(str(int(i)) for i in indices) + "\n", encoding="utf-8")

    print(f"Saved {args.num_samples} samples to {output_dir}")
    print(f"Selected indices: {indices.tolist()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Raw MAT dataset: {output_dir / 'inputs'}")


if __name__ == "__main__":
    main()
