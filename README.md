# PIR-UKAN for Electromagnetic Inverse Scattering

This repository implements Physics-Initialized Refinement U-KAN (PIR-UKAN), a
BP-guided two-step framework for complex-valued electromagnetic inverse
scattering. Back-propagation (BP) first produces a physics-guided complex
contrast initialization from scattered-field measurements. A U-shaped
Kolmogorov-Arnold Network (U-KAN) then performs image-domain nonlinear
compensation to suppress artifacts, recover boundaries, and correct local
details.

PIR-UKAN adapts U-KAN to inverse-scattering reconstruction with
Electromagnetically Driven Sparse Fusion Attention (EDSFA), FFT refinement,
real-imaginary residual calibration, and edge-aware residual refinement. Inputs
and targets are represented as real/imaginary channels.

## Repository Structure

```text
.
|-- config.py       # single source of baseline defaults
|-- archs.py        # U-KAN backbone, FCSA blocks, refinement heads
|-- kan.py          # spline-augmented KAN layers
|-- dataset.py      # complex-valued MAT dataset loader
|-- utils.py        # meters, SSIM loss, relative-error helper
|-- train.py        # baseline training entrypoint
|-- test.py         # baseline evaluation entrypoint
|-- requirements.txt
`-- LICENSE
```

## Environment

Python 3.10 is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

The baseline was trained with PyTorch 2.5.1 + CUDA 12.1. If your package index
does not resolve the CUDA wheels from `requirements.txt`, install PyTorch first
from the official CUDA index shown in the comments inside `requirements.txt`.

## Data Layout

The default baseline configuration expects:

```text
inputs/
|-- input/
|   |-- chi0_all_real_mnist.mat
|   `-- chi0_all_imag_mnist.mat
`-- label/
    |-- chi_all_real_mnist.mat
    `-- chi_all_imag_mnist.mat
```

## Training

The default command is intentionally enough: all defaults come from
`config.py` and mirror `outputs/baseline/config.yml`.

```bash
python train.py
```

Useful overrides are still available, for example:

```bash
python train.py --name my_baseline_run --epochs 400 --batch_size 8
```

Each run writes `config.yml`, `norm_stats.json`, `split_indices.json`,
checkpoints, metrics, TensorBoard logs, and a minimal `code_backup/` snapshot
under `outputs/<name>/`.

## Evaluation

Evaluate the packaged baseline checkpoint directory:

```bash
python test.py --exp_dir outputs/baseline --checkpoint model_best_ssim.pth
```

By default, evaluation uses the same 10% shuffled test split and train-set
normalization statistics as the baseline run.

## Baseline Reference

The saved baseline run reports:

- Best validation MSE: `0.02010381`
- Best validation SSIM: `0.90062162`
- Test mean MSE: `0.018322`
- Test average SSIM: `0.880230`
- Test RRMSE: `0.135269`

## License

This project is released under the MIT License. See `LICENSE` for details.
