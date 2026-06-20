"""Baseline configuration shared by training and evaluation.

The values below mirror ``outputs/baseline/config.yml``. Keeping them in one
place prevents README examples, train defaults, and evaluation code from
quietly drifting apart.
"""

from copy import deepcopy


BASELINE_CONFIG = {
    "arch": "UKAN",
    "batch_size": 8,
    "data_dir": "inputs",
    "dataseed": 2981,
    "deep_supervision": True,
    "detail_refine_mid": 64,
    "detail_refine_scale": 0.1,
    "early_stop_metric": "mse",
    "early_stopping": -1,
    "edge_center_boost": 0.6,
    "edge_center_sigma": 0.45,
    "edge_focus_gamma": 12.0,
    "edge_focus_tau": 0.25,
    "edge_refine_mid": 48,
    "edge_refine_scale": 0.25,
    "epochs": 400,
    "factor": 0.1,
    "fourier_refine_mid": 48,
    "fourier_refine_scale": 0.12,
    "fourier_use_fft": True,
    "gamma": 2 / 3,
    "imag_img_file": "input/chi0_all_imag_mnist.mat",
    "imag_label_file": "label/chi_all_imag_mnist.mat",
    "input_channels": 2,
    "input_h": 64,
    "input_list": [128, 160, 256],
    "input_w": 64,
    "kan_lr": 0.001,
    "kan_weight_decay": 0.0001,
    "loss": "MSE_SSIM",
    "lr": 0.0001,
    "milestones": "1,2",
    "min_lr": 1.0e-6,
    "momentum": 0.9,
    "name": "edge2h_fft_msfocus_b_dualri_s006_edsfa_tuned",
    "nesterov": False,
    "no_kan": False,
    "num_classes": 2,
    "num_workers": 0,
    "optimizer": "Adam",
    "original_img_size": 64,
    "output_dir": "outputs",
    "patience": 5,
    "real_img_file": "input/chi0_all_real_mnist.mat",
    "real_label_file": "label/chi_all_real_mnist.mat",
    "ri_refine_mid": 48,
    "ri_refine_scale": 0.06,
    "scheduler": "CosineAnnealingLR",
    "use_detail_skip_refine": False,
    "use_dual_ri_refine": True,
    "use_edge_center_boost": False,
    "use_edge_multiscale": True,
    "use_edge_residual_refine": True,
    "use_edge_sparse_focus": True,
    "use_fourier_refine": True,
    "weight_decay": 0.0001,
}


CODE_BACKUP_FILES = (
    "config.py",
    "train.py",
    "test.py",
    "archs.py",
    "kan.py",
    "dataset.py",
    "utils.py",
)


def baseline_config():
    """Return an isolated copy of the baseline defaults."""
    return deepcopy(BASELINE_CONFIG)
