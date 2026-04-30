# U-KAN for Two-Step Electromagnetic Inverse Scattering Imaging

This repository contains the training and evaluation code for a U-KAN-based pipeline for complex-valued electromagnetic inverse scattering imaging (real/imaginary channels).

## Features

- U-KAN backbone with KAN blocks (`archs.py`, `kan.py`)
- Frequency-channel-spatial attention module (`FCS_attention.py`)
- Complex-valued dataset loader from `.mat` files (`dataset_e.py`)
- Training script with optional refinement modules (`train.py`)
- evaluation scripts（`test_norm.py`）

## Repository Structure

```text
.
|-- archs.py
|-- kan.py
|-- FCS_attention.py
|-- dataset_e.py
|-- utils.py
|-- train.py
|-- test_norm.py
|-- inputs/
|   |-- input/
|   `-- label/
```

## Enviro

- Python 3.10 (recommended)
- PyTorch with CUDA support

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The default training script expects:

- `inputs/input/chi0_all_real_mnist.mat`
- `inputs/input/chi0_all_imag_mnist.mat`
- `inputs/label/chi_all_real_mnist.mat`
- `inputs/label/chi_all_imag_mnist.mat`

If you use different files, pass custom paths via command-line arguments.

## Training

Baseline example:

```bash
python train.py --name baseline_run --epochs 400 -b 8 --arch UKAN --deep_supervision True --loss MSE_SSIM --data_dir inputs --real_img_file input/chi0_all_real_mnist.mat --imag_img_file input/chi0_all_imag_mnist.mat --real_label_file label/chi_all_real_mnist.mat --imag_label_file label/chi_all_imag_mnist.mat --input_h 64 --input_w 64 --original_img_size 64 --input_channels 2 --num_classes 2 --input_list 128,160,256 --optimizer Adam --lr 0.0001 --kan_lr 0.001 --weight_decay 0.0001 --kan_weight_decay 0.0001 --scheduler CosineAnnealingLR --min_lr 1e-6 --dataseed 2981 --num_workers 0
```

Training outputs are written to:

- `outputs/<experiment_name>/`

Typical files include:

- `config.yml`
- `log.csv`
- `model.pth`
- `model_best_mse.pth`
- `model_best_ssim.pth` (if enabled by metric logic)
- `norm_stats.json`

## Evaluation

Evaluate on configurable split/group:

```bash
python test_norm.py --exp_dir outputs/baseline_run --checkpoint model_best_mse.pth
```


## Notes

- This code assumes CUDA is available for training.
- Large datasets are not included by default.
- Open-source model weights are located in /checkpoints.

## License

This project is released under the MIT License. See `LICENSE` for details.
