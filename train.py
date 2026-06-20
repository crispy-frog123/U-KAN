"""Train the baseline U-KAN model for inverse-scattering reconstruction."""

import argparse
import json
import os
import random
import shutil
import warnings
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from albumentations import Compose, HorizontalFlip, RandomRotate90, Resize, VerticalFlip
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import archs
from config import CODE_BACKUP_FILES, baseline_config
from dataset import ComplexMatDataset, TransformSubset
from utils import AverageMeter, SSIMLoss, str2bool

warnings.filterwarnings("ignore", category=FutureWarning)


DEEP_SUPERVISION_WEIGHTS = (0.4, 0.4, 1.0)


def list_type(value):
    """Parse comma-separated embedding widths, e.g. ``128,160,256``."""
    if isinstance(value, list):
        return value
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    """Parse CLI arguments whose defaults exactly match the baseline config."""
    defaults = baseline_config()
    parser = argparse.ArgumentParser(
        description="Train U-KAN with the repository baseline configuration."
    )

    parser.add_argument("--name", default=defaults["name"], help="experiment name")
    parser.add_argument("--epochs", default=defaults["epochs"], type=int)
    parser.add_argument("-b", "--batch_size", default=defaults["batch_size"], type=int)
    parser.add_argument("--dataseed", default=defaults["dataseed"], type=int)

    parser.add_argument("--arch", "-a", default=defaults["arch"], choices=archs.__all__)
    parser.add_argument("--deep_supervision", default=defaults["deep_supervision"], type=str2bool)
    parser.add_argument("--input_channels", default=defaults["input_channels"], type=int)
    parser.add_argument("--num_classes", default=defaults["num_classes"], type=int)
    parser.add_argument("--input_w", default=defaults["input_w"], type=int)
    parser.add_argument("--input_h", default=defaults["input_h"], type=int)
    parser.add_argument("--input_list", type=list_type, default=defaults["input_list"])
    parser.add_argument(
        "--loss",
        default=defaults["loss"],
        choices=["MSELoss", "MSE_SSIM", "L1Loss", "SmoothL1Loss"],
    )

    parser.add_argument("--data_dir", default=defaults["data_dir"])
    parser.add_argument("--real_img_file", default=defaults["real_img_file"])
    parser.add_argument("--imag_img_file", default=defaults["imag_img_file"])
    parser.add_argument("--real_label_file", default=defaults["real_label_file"])
    parser.add_argument("--imag_label_file", default=defaults["imag_label_file"])
    parser.add_argument("--original_img_size", default=defaults["original_img_size"], type=int)
    parser.add_argument("--output_dir", default=defaults["output_dir"])

    parser.add_argument("--optimizer", default=defaults["optimizer"], choices=["Adam", "SGD"])
    parser.add_argument("--lr", "--learning_rate", default=defaults["lr"], type=float)
    parser.add_argument("--momentum", default=defaults["momentum"], type=float)
    parser.add_argument("--weight_decay", default=defaults["weight_decay"], type=float)
    parser.add_argument("--nesterov", default=defaults["nesterov"], type=str2bool)
    parser.add_argument("--kan_lr", default=defaults["kan_lr"], type=float)
    parser.add_argument("--kan_weight_decay", default=defaults["kan_weight_decay"], type=float)

    parser.add_argument(
        "--scheduler",
        default=defaults["scheduler"],
        choices=["CosineAnnealingLR", "ReduceLROnPlateau", "MultiStepLR", "ConstantLR"],
    )
    parser.add_argument("--min_lr", default=defaults["min_lr"], type=float)
    parser.add_argument("--factor", default=defaults["factor"], type=float)
    parser.add_argument("--patience", default=defaults["patience"], type=int)
    parser.add_argument("--milestones", default=defaults["milestones"])
    parser.add_argument("--gamma", default=defaults["gamma"], type=float)
    parser.add_argument("--early_stopping", default=defaults["early_stopping"], type=int)
    parser.add_argument(
        "--early_stop_metric",
        default=defaults["early_stop_metric"],
        choices=["mse", "ssim"],
    )
    parser.add_argument("--num_workers", default=defaults["num_workers"], type=int)
    parser.add_argument("--no_kan", default=defaults["no_kan"], action="store_true")

    parser.add_argument(
        "--use_edge_residual_refine",
        default=defaults["use_edge_residual_refine"],
        type=str2bool,
    )
    parser.add_argument(
        "--use_edge_multiscale", default=defaults["use_edge_multiscale"], type=str2bool
    )
    parser.add_argument(
        "--use_edge_sparse_focus",
        default=defaults["use_edge_sparse_focus"],
        type=str2bool,
    )
    parser.add_argument("--edge_focus_tau", default=defaults["edge_focus_tau"], type=float)
    parser.add_argument("--edge_focus_gamma", default=defaults["edge_focus_gamma"], type=float)
    parser.add_argument(
        "--use_edge_center_boost",
        default=defaults["use_edge_center_boost"],
        type=str2bool,
    )
    parser.add_argument("--edge_center_boost", default=defaults["edge_center_boost"], type=float)
    parser.add_argument("--edge_center_sigma", default=defaults["edge_center_sigma"], type=float)
    parser.add_argument("--edge_refine_scale", default=defaults["edge_refine_scale"], type=float)
    parser.add_argument("--edge_refine_mid", default=defaults["edge_refine_mid"], type=int)
    parser.add_argument(
        "--use_detail_skip_refine",
        default=defaults["use_detail_skip_refine"],
        type=str2bool,
    )
    parser.add_argument(
        "--detail_refine_scale", default=defaults["detail_refine_scale"], type=float
    )
    parser.add_argument("--detail_refine_mid", default=defaults["detail_refine_mid"], type=int)
    parser.add_argument(
        "--use_fourier_refine", default=defaults["use_fourier_refine"], type=str2bool
    )
    parser.add_argument("--fourier_use_fft", default=defaults["fourier_use_fft"], type=str2bool)
    parser.add_argument(
        "--fourier_refine_scale", default=defaults["fourier_refine_scale"], type=float
    )
    parser.add_argument("--fourier_refine_mid", default=defaults["fourier_refine_mid"], type=int)
    parser.add_argument(
        "--use_dual_ri_refine", default=defaults["use_dual_ri_refine"], type=str2bool
    )
    parser.add_argument("--ri_refine_mid", default=defaults["ri_refine_mid"], type=int)
    parser.add_argument("--ri_refine_scale", default=defaults["ri_refine_scale"], type=float)

    return parser.parse_args(argv)


class MSESSIMLoss(nn.Module):
    """Baseline objective: MSE plus a half-weighted SSIM loss term."""

    def __init__(self, channel=2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=11, channel=channel)

    def forward(self, pred, target):
        return self.mse(pred, target) + 0.5 * self.ssim_loss(pred, target)


def seed_torch(seed=1029):
    """Seed Python, NumPy, and PyTorch for repeatable data splits and init."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def build_path(base_dir, file_path):
    """Resolve a data path while allowing absolute overrides."""
    if os.path.isabs(file_path):
        return file_path
    return os.path.normpath(os.path.join(base_dir, file_path))


def split_indices(dataset_size, seed):
    """Reproduce the baseline 8:1:1 train/val/test split."""
    indices = list(range(dataset_size))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    test_count = int(np.floor(0.1 * dataset_size))
    val_count = int(np.floor(0.1 * dataset_size))
    return {
        "train_indices": indices[test_count + val_count :],
        "val_indices": indices[test_count : test_count + val_count],
        "test_indices": indices[:test_count],
    }


def make_loaders(config, exp_dir):
    """Load MAT files, compute train-only normalization stats, and build loaders."""
    real_img_path = build_path(config["data_dir"], config["real_img_file"])
    imag_img_path = build_path(config["data_dir"], config["imag_img_file"])
    real_label_path = build_path(config["data_dir"], config["real_label_file"])
    imag_label_path = build_path(config["data_dir"], config["imag_label_file"])

    print("\nData files:")
    print(f"  Real input:  {real_img_path}")
    print(f"  Imag input:  {imag_img_path}")
    print(f"  Real target: {real_label_path}")
    print(f"  Imag target: {imag_label_path}")

    full_dataset = ComplexMatDataset(
        real_img_path=real_img_path,
        imag_img_path=imag_img_path,
        real_label_path=real_label_path,
        imag_label_path=imag_label_path,
        img_size=(config["original_img_size"], config["original_img_size"]),
        normalize_method="z-score",
    )

    splits = split_indices(len(full_dataset), config["dataseed"])
    full_dataset.calculate_normalization_stats(splits["train_indices"])
    full_dataset.save_stats(exp_dir / "norm_stats.json")
    with open(exp_dir / "split_indices.json", "w", encoding="utf-8") as handle:
        json.dump({"seed": config["dataseed"], **splits}, handle, indent=2)

    train_transform = Compose(
        [
            RandomRotate90(),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Resize(config["input_h"], config["input_w"]),
        ]
    )
    eval_transform = Compose([Resize(config["input_h"], config["input_w"])])

    train_dataset = TransformSubset(full_dataset, splits["train_indices"], train_transform)
    val_dataset = TransformSubset(full_dataset, splits["val_indices"], eval_transform)

    print("\nDataset split (8:1:1):")
    print(f"  Total:      {len(full_dataset)}")
    print(f"  Train:      {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test:       {len(splits['test_indices'])}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def build_model(config, device):
    """Instantiate the baseline UKAN variant from the shared config."""
    model = archs.__dict__[config["arch"]](
        config["num_classes"],
        config["input_channels"],
        config["deep_supervision"],
        embed_dims=config["input_list"],
        no_kan=config["no_kan"],
        use_edge_residual_refine=config["use_edge_residual_refine"],
        use_edge_multiscale=config["use_edge_multiscale"],
        use_edge_sparse_focus=config["use_edge_sparse_focus"],
        edge_focus_tau=config["edge_focus_tau"],
        edge_focus_gamma=config["edge_focus_gamma"],
        use_edge_center_boost=config["use_edge_center_boost"],
        edge_center_boost=config["edge_center_boost"],
        edge_center_sigma=config["edge_center_sigma"],
        edge_refine_scale=config["edge_refine_scale"],
        edge_refine_mid=config["edge_refine_mid"],
        use_detail_skip_refine=config["use_detail_skip_refine"],
        detail_refine_scale=config["detail_refine_scale"],
        detail_refine_mid=config["detail_refine_mid"],
        use_fourier_refine=config["use_fourier_refine"],
        fourier_use_fft=config["fourier_use_fft"],
        fourier_refine_scale=config["fourier_refine_scale"],
        fourier_refine_mid=config["fourier_refine_mid"],
        use_dual_ri_refine=config["use_dual_ri_refine"],
        ri_refine_mid=config["ri_refine_mid"],
        ri_refine_scale=config["ri_refine_scale"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel:")
    print(f"  Total params:     {total_params / 1e6:.2f}M")
    print(f"  Trainable params: {trainable_params / 1e6:.2f}M")
    return model


def build_optimizer(config, model):
    """Use the baseline two-rate optimizer: KAN params and all other params."""
    kan_params = []
    other_params = []
    for name, param in model.named_parameters():
        is_kan = "kan" in name.lower() or ("layer" in name.lower() and "fc" in name.lower())
        (kan_params if is_kan else other_params).append(param)

    param_groups = []
    if kan_params:
        param_groups.append(
            {
                "params": kan_params,
                "lr": config["kan_lr"],
                "weight_decay": config["kan_weight_decay"],
            }
        )
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": config["lr"],
                "weight_decay": config["weight_decay"],
            }
        )

    print("\nOptimizer:")
    print(f"  KAN params:   {sum(p.numel() for p in kan_params) / 1e6:.2f}M")
    print(f"  Other params: {sum(p.numel() for p in other_params) / 1e6:.2f}M")
    if config["optimizer"] == "Adam":
        return optim.Adam(param_groups)
    if config["optimizer"] == "SGD":
        return optim.SGD(
            param_groups,
            momentum=config["momentum"],
            nesterov=config["nesterov"],
        )
    raise NotImplementedError(f"Unsupported optimizer: {config['optimizer']}")


def build_scheduler(config, optimizer):
    """Build the selected scheduler; defaults match the baseline cosine schedule."""
    if config["scheduler"] == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"],
            eta_min=config["min_lr"],
        )
    if config["scheduler"] == "ReduceLROnPlateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["factor"],
            patience=config["patience"],
            min_lr=config["min_lr"],
        )
    if config["scheduler"] == "MultiStepLR":
        milestones = [int(epoch) for epoch in config["milestones"].split(",")]
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=config["gamma"])
    if config["scheduler"] == "ConstantLR":
        return None
    raise NotImplementedError(f"Unsupported scheduler: {config['scheduler']}")


def build_criterion(config, device):
    """Build the selected loss; the default is the baseline MSE+SSIM objective."""
    if config["loss"] == "MSE_SSIM":
        return MSESSIMLoss(channel=config["num_classes"]).to(device)
    if config["loss"] == "MSELoss":
        return nn.MSELoss().to(device)
    if config["loss"] == "L1Loss":
        return nn.L1Loss().to(device)
    if config["loss"] == "SmoothL1Loss":
        return nn.SmoothL1Loss().to(device)
    raise NotImplementedError(f"Unsupported loss: {config['loss']}")


def forward_with_loss(config, model, criterion, input_tensor, target):
    """Run a forward pass and apply deep-supervision loss weights if enabled."""
    if config["deep_supervision"]:
        outputs = model(input_tensor)
        loss = sum(
            weight * criterion(output, target)
            for output, weight in zip(outputs, DEEP_SUPERVISION_WEIGHTS)
        )
        return outputs[-1], loss

    output = model(input_tensor)
    return output, criterion(output, target)


def train_one_epoch(config, train_loader, model, criterion, optimizer):
    """Run one training epoch."""
    model.train()
    meters = {key: AverageMeter() for key in ("loss", "mse", "mae", "rmse")}

    pbar = tqdm(total=len(train_loader), desc="Training")
    for input_tensor, target, _ in train_loader:
        input_tensor = input_tensor.to(config["device"])
        target = target.to(config["device"])

        output, loss = forward_with_loss(config, model, criterion, input_tensor, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mse = F.mse_loss(output, target)
            mae = F.l1_loss(output, target)
            rmse = torch.sqrt(mse)

        batch_size = input_tensor.size(0)
        meters["loss"].update(loss.item(), batch_size)
        meters["mse"].update(mse.item(), batch_size)
        meters["mae"].update(mae.item(), batch_size)
        meters["rmse"].update(rmse.item(), batch_size)
        pbar.set_postfix(OrderedDict((key, f"{meter.avg:.6f}") for key, meter in meters.items()))
        pbar.update(1)

    pbar.close()
    return {key: meter.avg for key, meter in meters.items()}


def validate(config, val_loader, model, criterion):
    """Evaluate one validation epoch."""
    model.eval()
    meters = {key: AverageMeter() for key in ("loss", "mse", "mae", "rmse", "ssim")}
    ssim_metric = SSIMLoss(window_size=11, channel=config["num_classes"]).to(config["device"])

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc="Validation")
        for input_tensor, target, _ in val_loader:
            input_tensor = input_tensor.to(config["device"])
            target = target.to(config["device"])

            output, loss = forward_with_loss(config, model, criterion, input_tensor, target)
            mse = F.mse_loss(output, target)
            mae = F.l1_loss(output, target)
            rmse = torch.sqrt(mse)
            ssim = 1.0 - ssim_metric(output, target)

            batch_size = input_tensor.size(0)
            meters["loss"].update(loss.item(), batch_size)
            meters["mse"].update(mse.item(), batch_size)
            meters["mae"].update(mae.item(), batch_size)
            meters["rmse"].update(rmse.item(), batch_size)
            meters["ssim"].update(ssim.item(), batch_size)
            pbar.set_postfix(
                OrderedDict((f"val_{key}", f"{meter.avg:.6f}") for key, meter in meters.items())
            )
            pbar.update(1)
        pbar.close()

    return {key: meter.avg for key, meter in meters.items()}


def snapshot_code(exp_dir):
    """Copy the runnable source files into the experiment directory."""
    backup_dir = exp_dir / "code_backup"
    backup_dir.mkdir(exist_ok=True)
    for file_name in CODE_BACKUP_FILES:
        path = Path(file_name)
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)


def write_best_results(exp_dir, exp_name, config, best):
    """Persist the best validation metrics in a compact text report."""
    with open(exp_dir / "best_results.txt", "w", encoding="utf-8") as handle:
        handle.write("=" * 40 + "\n")
        handle.write(f"Experiment: {exp_name}\n")
        handle.write(f"Finished at Epoch: {config['epochs']}\n")
        handle.write("=" * 40 + "\n\n")
        handle.write("Best Metrics during Training:\n")
        handle.write("-" * 30 + "\n")
        handle.write(f"Best Loss: {best['loss']:.8f}\n")
        handle.write(f"Best MSE:  {best['mse']:.8f}\n")
        handle.write(f"Best MAE:  {best['mae']:.8f}\n")
        handle.write(f"Best RMSE: {best['rmse']:.8f}\n")
        handle.write(f"Best SSIM: {best['ssim']:.8f}\n")
        handle.write("-" * 30 + "\n")


def main(argv=None):
    """Run the full baseline training pipeline."""
    seed_torch()
    config = vars(parse_args(argv))

    exp_name = config["name"]
    exp_dir = Path(config["output_dir"]) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.yml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("CUDA is required for the baseline training configuration.")
    config["device"] = device

    cudnn.benchmark = True
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"\nGPU: {torch.cuda.get_device_name(0)} ({torch.version.cuda})")

    writer = SummaryWriter(str(exp_dir))
    try:
        model = build_model(config, device)
        criterion = build_criterion(config, device)
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer)
        snapshot_code(exp_dir)
        train_loader, val_loader = make_loaders(config, exp_dir)

        log = OrderedDict(
            (key, [])
            for key in (
                "epoch",
                "lr",
                "loss",
                "mse",
                "mae",
                "rmse",
                "val_loss",
                "val_mse",
                "val_mae",
                "val_rmse",
                "val_ssim",
            )
        )

        best = {
            "loss": float("inf"),
            "mae": float("inf"),
            "mse": float("inf"),
            "rmse": float("inf"),
            "ssim": -float("inf"),
            "loss_for_mse": float("inf"),
            "loss_for_ssim": float("inf"),
        }
        trigger = 0

        print("\nStart training")
        for epoch in range(config["epochs"]):
            print(f"\nEpoch [{epoch}/{config['epochs']}]")
            train_log = train_one_epoch(config, train_loader, model, criterion, optimizer)
            val_log = validate(config, val_loader, model, criterion)

            best["mae"] = min(best["mae"], val_log["mae"])
            best["rmse"] = min(best["rmse"], val_log["rmse"])
            improved_mse = val_log["mse"] < best["mse"]
            improved_ssim = val_log["ssim"] > best["ssim"]

            if improved_mse:
                best["mse"] = val_log["mse"]
                best["loss_for_mse"] = val_log["loss"]
                torch.save(model.state_dict(), exp_dir / "model_best_mse.pth")
                print(f"=> saved best-MSE model: val_mse={val_log['mse']:.6f}")

            if improved_ssim:
                best["ssim"] = val_log["ssim"]
                best["loss_for_ssim"] = val_log["loss"]
                torch.save(model.state_dict(), exp_dir / "model_best_ssim.pth")
                print(f"=> saved best-SSIM model: val_ssim={val_log['ssim']:.6f}")

            if config["early_stop_metric"] == "mse":
                improved_main = improved_mse
                best["loss"] = best["loss_for_mse"]
            else:
                improved_main = improved_ssim
                best["loss"] = best["loss_for_ssim"]

            if improved_main:
                torch.save(model.state_dict(), exp_dir / "model.pth")
                trigger = 0
            else:
                trigger += 1
                print(f"=> no improvement for {trigger} epochs on val_{config['early_stop_metric']}")

            if config["early_stopping"] >= 0 and trigger >= config["early_stopping"]:
                print("=> early stopping triggered")
                break

            if config["scheduler"] == "CosineAnnealingLR":
                scheduler.step()
            elif config["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(val_log["loss"])
            elif config["scheduler"] == "MultiStepLR":
                scheduler.step()
            print(
                "Train - "
                f"loss={train_log['loss']:.6f}, mse={train_log['mse']:.6f}, "
                f"mae={train_log['mae']:.6f}, rmse={train_log['rmse']:.6f}"
            )
            print(
                "Val   - "
                f"loss={val_log['loss']:.6f}, mse={val_log['mse']:.6f}, "
                f"mae={val_log['mae']:.6f}, rmse={val_log['rmse']:.6f}, "
                f"ssim={val_log['ssim']:.6f}"
            )

            log["epoch"].append(epoch)
            log["lr"].append(optimizer.param_groups[0]["lr"])
            log["loss"].append(train_log["loss"])
            log["mse"].append(train_log["mse"])
            log["mae"].append(train_log["mae"])
            log["rmse"].append(train_log["rmse"])
            log["val_loss"].append(val_log["loss"])
            log["val_mse"].append(val_log["mse"])
            log["val_mae"].append(val_log["mae"])
            log["val_rmse"].append(val_log["rmse"])
            log["val_ssim"].append(val_log["ssim"])
            pd.DataFrame(log).to_csv(exp_dir / "log.csv", index=False)

            writer.add_scalar("train/loss", train_log["loss"], epoch)
            writer.add_scalar("train/mse", train_log["mse"], epoch)
            writer.add_scalar("train/mae", train_log["mae"], epoch)
            writer.add_scalar("train/rmse", train_log["rmse"], epoch)
            writer.add_scalar("val/loss", val_log["loss"], epoch)
            writer.add_scalar("val/mse", val_log["mse"], epoch)
            writer.add_scalar("val/mae", val_log["mae"], epoch)
            writer.add_scalar("val/rmse", val_log["rmse"], epoch)
            writer.add_scalar("val/ssim", val_log["ssim"], epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

            torch.cuda.empty_cache()

        write_best_results(exp_dir, exp_name, config, best)
        print(f"\nTraining finished. Results saved to {exp_dir}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
