import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from collections import OrderedDict
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from albumentations.core.composition import Compose
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize, HorizontalFlip, VerticalFlip

import archs
from dataset_e import ComplexMatDataset, TransformSubset
from utils import AverageMeter, str2bool, SSIMLoss
from torch.utils.tensorboard import SummaryWriter

import shutil
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

ARCH_NAMES = archs.__all__


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='random seed for dataset split')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN')
    parser.add_argument('--deep_supervision', default=True, type=str2bool)

    parser.add_argument('--input_channels', default=2, type=int,
                        help='input channels ')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')

    parser.add_argument('--input_w', default=64, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=64, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    parser.add_argument('--loss', default='MSE_SSIM',
                        help='training loss (kept for CLI compatibility)')

    parser.add_argument('--data_dir', default='inputs', help='dataset base directory')

    parser.add_argument('--real_img_file', default='input/chi0_all_real_mnist.mat',
                        help='real part image .mat file')
    parser.add_argument('--imag_img_file', default='input/chi0_all_imag_mnist.mat',
                        help='imaginary part image .mat file')
    parser.add_argument('--real_label_file', default='label/chi_all_real_mnist.mat',
                        help='real part label .mat file')
    parser.add_argument('--imag_label_file', default='label/chi_all_imag_mnist.mat',
                        help='imaginary part label .mat file')

    parser.add_argument('--img_var_name', default='chi0_all_real',
                        help='variable name in .mat file for images')
    parser.add_argument('--label_var_name', default='chi_all_real',
                        help='variable name in .mat file for labels')

    parser.add_argument('--original_img_size', default=64, type=int,
                        help='original image size (H=W)')

    parser.add_argument('--output_dir', default='outputs', help='output directory')

    parser.add_argument('--val_split', default=0.2, type=float,
                        help='validation split ratio')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        help='optimizer (Adam is the supported baseline setting)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')

    parser.add_argument('--kan_lr', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate for KAN layers')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay for KAN layers')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        help='scheduler (CosineAnnealingLR is the supported baseline setting)')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='minimum learning rate')
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (-1 disables)')
    parser.add_argument('--early_stop_metric', default='mse', choices=['mse', 'ssim'],
                        help='metric used for early stopping and model.pth alias')

    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--use_edge_residual_refine', default=True, type=str2bool,
                        help='enable edge-aware residual refinement head')
    parser.add_argument('--use_edge_multiscale', default=True, type=str2bool,
                        help='use multi-scale dilated edge context in edge refine')
    parser.add_argument('--use_edge_sparse_focus', default=True, type=str2bool,
                        help='focus edge refine on high-gradient sparse regions')
    parser.add_argument('--edge_focus_tau', default=0.25, type=float,
                        help='edge focus threshold on normalized gradient')
    parser.add_argument('--edge_focus_gamma', default=12.0, type=float,
                        help='edge focus sharpness')
    parser.add_argument('--edge_refine_scale', default=0.25, type=float,
                        help='initial scaling for edge residual refinement')
    parser.add_argument('--edge_refine_mid', default=48, type=int,
                        help='hidden channels in edge residual refinement head')
    parser.add_argument('--use_fourier_refine', default=True, type=str2bool,
                        help='enable Fourier-domain refinement on final feature map')
    parser.add_argument('--fourier_use_fft', default=True, type=str2bool,
                        help='use FFT kernel in fourier refine; disable for unstable CUDA/cuFFT stacks')
    parser.add_argument('--fourier_refine_scale', default=0.12, type=float,
                        help='initial scaling for Fourier refinement residual')
    parser.add_argument('--fourier_refine_mid', default=48, type=int,
                        help='hidden channels for Fourier refinement MLP')
    parser.add_argument('--use_dual_ri_refine', default=True, type=str2bool,
                        help='enable decoupled real/imag tail refinement on shared decoder output')
    parser.add_argument('--ri_refine_mid', default=48, type=int,
                        help='hidden channels for dual real/imag tail refinement')
    parser.add_argument('--ri_refine_scale', default=0.06, type=float,
                        help='initial residual scaling for dual real/imag tail refinement')

    config = parser.parse_args()

    return config

class MSE_SSIM_Loss(nn.Module):
    def __init__(self, channel=2):
        super(MSE_SSIM_Loss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=11, channel=channel)

    def forward(self, pred, target):
        return self.mse(pred, target) + 0.5 * self.ssim_loss(pred, target)


def train(config, train_loader, model, criterion, optimizer):
    """Run one training epoch for the regression objective."""
    device = config['device']
    avg_meters = {
        'loss': AverageMeter(),
        'mse': AverageMeter(),
        'mae': AverageMeter(),
        'rmse': AverageMeter()
    }

    model.train()

    pbar = tqdm(total=len(train_loader), desc='Training')
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        # Deep supervision uses fixed weights from the released baseline.
        if config['deep_supervision']:
            outputs = model(input)
            weights = [0.4, 0.4, 1.0]
            loss = 0
            for output_item, weight in zip(outputs, weights):
                loss += weight * criterion(output_item, target)
            output = outputs[-1]
        else:
            output = model(input)
            loss = criterion(output, target)

        with torch.no_grad():
            mse = F.mse_loss(output, target)
            mae = F.l1_loss(output, target)
            rmse = torch.sqrt(mse)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['mse'].update(mse.item(), input.size(0))
        avg_meters['mae'].update(mae.item(), input.size(0))
        avg_meters['rmse'].update(rmse.item(), input.size(0))

        postfix = OrderedDict([
            ('loss', f"{avg_meters['loss'].avg:.6f}"),
            ('mse', f"{avg_meters['mse'].avg:.6f}"),
            ('mae', f"{avg_meters['mae'].avg:.6f}"),
            ('rmse', f"{avg_meters['rmse'].avg:.6f}"),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)

    pbar.close()

    return {
        'loss': avg_meters['loss'].avg,
        'mse': avg_meters['mse'].avg,
        'mae': avg_meters['mae'].avg,
        'rmse': avg_meters['rmse'].avg
    }


def validate(config, val_loader, model, criterion):
    """Run one validation epoch for the regression objective."""
    device = config['device']
    avg_meters = {
        'loss': AverageMeter(),
        'mse': AverageMeter(),
        'mae': AverageMeter(),
        'rmse': AverageMeter(),
        'ssim': AverageMeter()
    }

    model.eval()
    ssim_metric = SSIMLoss(window_size=11, channel=config['num_classes']).to(device)

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc='Validation')
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # Forward
            if config['deep_supervision']:
                outputs = model(input)
                weights = [0.4, 0.4, 1.0]
                loss = 0
                for output_item, weight in zip(outputs, weights):
                    loss += weight * criterion(output_item, target)
                output = outputs[-1]
            else:
                output = model(input)
                loss = criterion(output, target)

            mse = F.mse_loss(output, target)
            mae = F.l1_loss(output, target)
            rmse = torch.sqrt(mse)
            ssim = 1.0 - ssim_metric(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['mse'].update(mse.item(), input.size(0))
            avg_meters['mae'].update(mae.item(), input.size(0))
            avg_meters['rmse'].update(rmse.item(), input.size(0))
            avg_meters['ssim'].update(ssim.item(), input.size(0))

            postfix = OrderedDict([
                ('val_loss', f"{avg_meters['loss'].avg:.6f}"),
                ('val_mse', f"{avg_meters['mse'].avg:.6f}"),
                ('val_mae', f"{avg_meters['mae'].avg:.6f}"),
                ('val_rmse', f"{avg_meters['rmse'].avg:.6f}"),
                ('val_ssim', f"{avg_meters['ssim'].avg:.6f}"),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)

        pbar.close()

    return {
        'loss': avg_meters['loss'].avg,
        'mse': avg_meters['mse'].avg,
        'mae': avg_meters['mae'].avg,
        'rmse': avg_meters['rmse'].avg,
        'ssim': avg_meters['ssim'].avg
    }


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    if config['name'] is None:
        if config['loss'] == 'MSE_SSIM':
            config['name'] = 'test3_%s_wDS_Combo' % config['arch']
        else:
            config['name'] = 'test3_%s_wDS_%s' % (config['arch'], config['loss'])
        exp_name = config['name']

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('\n' + '=' * 60)
    print('Configuration:')
    print('=' * 60)
    for key in config:
        print(f'  {key}: {config[key]}')
    print('=' * 60 + '\n')

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # TensorBoard writer
    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cpu':
        raise RuntimeError("[ERROR] CUDA device is required for this training script.")

    print(f"\n{'=' * 60}")
    print("GPU Info")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")
    print(f"{'=' * 60}\n")

    config['device'] = device

    if config['loss'] != 'MSE_SSIM':
        raise ValueError(f"Unsupported loss '{config['loss']}'. Use MSE_SSIM for this cleaned baseline.")
    criterion = MSE_SSIM_Loss(channel=config['num_classes']).to(device)
    print("[INFO] Loss: MSE + SSIM")

    print("\nBuilding model...")
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        use_edge_residual_refine=config['use_edge_residual_refine'],
        use_edge_multiscale=config['use_edge_multiscale'],
        use_edge_sparse_focus=config['use_edge_sparse_focus'],
        edge_focus_tau=config['edge_focus_tau'],
        edge_focus_gamma=config['edge_focus_gamma'],
        edge_refine_scale=config['edge_refine_scale'],
        edge_refine_mid=config['edge_refine_mid'],
        use_fourier_refine=config['use_fourier_refine'],
        fourier_use_fft=config['fourier_use_fft'],
        fourier_refine_scale=config['fourier_refine_scale'],
        fourier_refine_mid=config['fourier_refine_mid'],
        use_dual_ri_refine=config['use_dual_ri_refine'],
        ri_refine_mid=config['ri_refine_mid'],
        ri_refine_scale=config['ri_refine_scale']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("[INFO] Model initialized")
    print(f"  Total params: {total_params / 1e6:.2f}M")
    print(f"  Trainable params: {trainable_params / 1e6:.2f}M")
    print(f"  Device: {next(model.parameters()).device}\n")
    print(f"  edge_residual_refine: {config['use_edge_residual_refine']}")
    print(f"  edge_multiscale: {config['use_edge_multiscale']}")
    print(f"  edge_sparse_focus: {config['use_edge_sparse_focus']}")
    print(f"  fourier_refine: {config['use_fourier_refine']}")
    print(f"  dual_ri_refine: {config['use_dual_ri_refine']}")

    param_groups = []
    kan_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'kan' in name.lower() or ('layer' in name.lower() and 'fc' in name.lower()):
            kan_params.append(param)
        else:
            other_params.append(param)

    if len(kan_params) > 0:
        param_groups.append({
            'params': kan_params,
            'lr': config['kan_lr'],
            'weight_decay': config['kan_weight_decay']
        })
        print(f"[INFO] KAN params: {sum(p.numel() for p in kan_params) / 1e6:.2f}M, lr={config['kan_lr']}")

    if len(other_params) > 0:
        param_groups.append({
            'params': other_params,
            'lr': config['lr'],
            'weight_decay': config['weight_decay']
        })
        print(f"[INFO] Other params: {sum(p.numel() for p in other_params) / 1e6:.2f}M, lr={config['lr']}\n")

    if config['optimizer'] != 'Adam':
        raise ValueError(f"Unsupported optimizer '{config['optimizer']}'. Use Adam for this cleaned baseline.")
    optimizer = optim.Adam(param_groups)

    if config['scheduler'] != 'CosineAnnealingLR':
        raise ValueError(
            f"Unsupported scheduler '{config['scheduler']}'. Use CosineAnnealingLR for this cleaned baseline."
        )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['min_lr'])

    backup_dir = os.path.join(output_dir, exp_name, 'code_backup')
    os.makedirs(backup_dir, exist_ok=True)
    backup_files = ['train.py', 'archs.py', 'utils.py', 'dataset_e.py', 'kan.py', 'FCS_attention.py']
    for fname in backup_files:
        if os.path.exists(fname):
            shutil.copy2(fname, os.path.join(backup_dir, fname))
    shutil.copy2(__file__, f'{output_dir}/{exp_name}/')
    if os.path.exists('archs.py'):
        shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')

    print("\n" + "=" * 60)
    print("Loading Complex Dataset (Regression)")
    print("=" * 60)

    def build_path(base_dir, file_path):
        if os.path.isabs(file_path):
            return file_path
        else:
            return os.path.normpath(os.path.join(base_dir, file_path))

    real_img_path = build_path(config['data_dir'], config['real_img_file'])
    imag_img_path = build_path(config['data_dir'], config['imag_img_file'])
    real_label_path = build_path(config['data_dir'], config['real_label_file'])
    imag_label_path = build_path(config['data_dir'], config['imag_label_file'])

    print(f"\nData files:")
    print(f"  Real image:  {real_img_path}")
    print(f"  Imag image:  {imag_img_path}")
    print(f"  Real target: {real_label_path}")
    print(f"  Imag target: {imag_label_path}")

    train_transform = Compose([
        RandomRotate90(),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Resize(config['input_h'], config['input_w']),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
    ])

    full_dataset = ComplexMatDataset(
        real_img_path=real_img_path,
        imag_img_path=imag_img_path,
        real_label_path=real_label_path,
        imag_label_path=imag_label_path,
        img_size=(config['original_img_size'], config['original_img_size']),
        transform=None,
        normalize_method='z-score'
    )

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    np.random.seed(config['dataseed'])
    np.random.shuffle(indices)


    test_split = int(np.floor(0.1 * dataset_size))  # 10%
    val_split = int(np.floor(0.1 * dataset_size))  # 10%

    test_indices = indices[:test_split]  # 0~199
    val_indices = indices[test_split: test_split + val_split]  # 200~399
    train_indices = indices[test_split + val_split:]  # 400~1999

    print("\n[Auto-Fix] Calculating normalization stats using Training indices...")
    full_dataset.calculate_normalization_stats(train_indices)
    full_dataset.save_stats(os.path.join(output_dir, exp_name, 'norm_stats.json'))

    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_transform)

    print(f"\nDataset split (8:1:1):")
    print(f"  Total:      {dataset_size}")
    print(f"  Training:   {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test:       {len(test_indices)} (reserved for evaluation)")

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('mse', []),
        ('mae', []),
        ('rmse', []),
        ('val_loss', []),
        ('val_mse', []),
        ('val_mae', []),
        ('val_rmse', []),
        ('val_ssim', []),
    ])

    best_loss = float('inf')
    trigger = 0

    print("\n" + "=" * 60)
    print("Start training (regression)")
    print("=" * 60 + "\n")

    best_loss = float('inf')
    best_mae = float('inf')
    best_mse = float('inf')
    best_rmse = float('inf')
    best_ssim = -float('inf')
    best_loss_mse = float('inf')
    best_loss_ssim = float('inf')
    trigger = 0  # Standardized technical note.
    for epoch in range(config['epochs']):
        print(f'\nEpoch [{epoch}/{config["epochs"]}]')
        print('-' * 60)

        # Train
        train_log = train(config, train_loader, model, criterion, optimizer)

        # Validate
        val_log = validate(config, val_loader, model, criterion)
        if val_log['mae'] < best_mae: best_mae = val_log['mae']
        if val_log['rmse'] < best_rmse: best_rmse = val_log['rmse']
        improved_mse = False
        improved_ssim = False

        if val_log['mse'] < best_mse:
            best_mse = val_log['mse']
            best_loss_mse = val_log['loss']
            improved_mse = True
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model_best_mse.pth')
            print(f"=> [Epoch {epoch}] Saved best-MSE model! (val_mse: {val_log['mse']:.6f})")

        if val_log['ssim'] > best_ssim:
            best_ssim = val_log['ssim']
            best_loss_ssim = val_log['loss']
            improved_ssim = True
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model_best_ssim.pth')
            print(f"=> [Epoch {epoch}] Saved best-SSIM model! (val_ssim: {val_log['ssim']:.6f})")

        if config['early_stop_metric'] == 'mse':
            improved_main = improved_mse
            best_loss = best_loss_mse
        else:
            improved_main = improved_ssim
            best_loss = best_loss_ssim

        if improved_main:
            # Keep model.pth aligned with the chosen early-stop metric.
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            trigger = 0
        else:
            trigger += 1
            print(f"=> No improvement for {trigger} epochs on val_{config['early_stop_metric']}.")

        print(
            f'   Best Results -> Loss: {best_loss:.4f} | MAE: {best_mae:.4f} | MSE: {best_mse:.4f} | RMSE: {best_rmse:.4f} | SSIM: {best_ssim:.4f}')

        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            if config['early_stop_metric'] == 'mse':
                print(f"=> Early stopping triggered! (Best val_mse: {best_mse:.6f})")
            else:
                print(f"=> Early stopping triggered! (Best val_ssim: {best_ssim:.6f})")
            break

        scheduler.step()

        print(f'\nResults:')
        print(
            f"  Train - Loss: {train_log['loss']:.6f}, MSE: {train_log['mse']:.6f}, MAE: {train_log['mae']:.6f}, RMSE: {train_log['rmse']:.6f}")
        print(
            f"  Val   - Loss: {val_log['loss']:.6f}, MSE: {val_log['mse']:.6f}, MAE: {val_log['mae']:.6f}, RMSE: {val_log['rmse']:.6f}, SSIM: {val_log['ssim']:.6f}")

        log['epoch'].append(epoch)
        log['lr'].append(optimizer.param_groups[0]['lr'])
        log['loss'].append(train_log['loss'])
        log['mse'].append(train_log['mse'])
        log['mae'].append(train_log['mae'])
        log['rmse'].append(train_log['rmse'])
        log['val_loss'].append(val_log['loss'])
        log['val_mse'].append(val_log['mse'])
        log['val_mae'].append(val_log['mae'])
        log['val_rmse'].append(val_log['rmse'])
        log['val_ssim'].append(val_log['ssim'])

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        # TensorBoard
        my_writer.add_scalar('train/loss', train_log['loss'], epoch)
        my_writer.add_scalar('train/mse', train_log['mse'], epoch)
        my_writer.add_scalar('train/mae', train_log['mae'], epoch)
        my_writer.add_scalar('train/rmse', train_log['rmse'], epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], epoch)
        my_writer.add_scalar('val/mse', val_log['mse'], epoch)
        my_writer.add_scalar('val/mae', val_log['mae'], epoch)
        my_writer.add_scalar('val/rmse', val_log['rmse'], epoch)
        my_writer.add_scalar('val/ssim', val_log['ssim'], epoch)
        my_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        my_writer.add_scalar('best/loss', best_loss, epoch)
        my_writer.add_scalar('best/mae', best_mae, epoch)
        my_writer.add_scalar('best/mse', best_mse, epoch)
        my_writer.add_scalar('best/rmse', best_rmse, epoch)
        my_writer.add_scalar('best/ssim', best_ssim, epoch)


        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    if config['early_stop_metric'] == 'mse':
        print(f"Training complete. Best val_mse: {best_mse:.6f}")
    else:
        print(f"Training complete. Best val_ssim: {best_ssim:.6f}")
    print("=" * 60)

    my_writer.close()
    result_path = f'{output_dir}/{exp_name}/best_results.txt'

    print(f"Writing summary to: {result_path}")

    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=" * 40 + "\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Finished at Epoch: {config['epochs']}\n")
        f.write("=" * 40 + "\n\n")

        f.write("Best Metrics during Training:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Loss: {best_loss:.8f}\n")
        f.write(f"Best MSE:  {best_mse:.8f}\n")
        f.write(f"Best MAE:  {best_mae:.8f}\n")
        f.write(f"Best RMSE: {best_rmse:.8f}\n")
        f.write(f"Best SSIM: {best_ssim:.8f}\n")
        f.write("-" * 30 + "\n")

    print("[INFO] Summary saved")
    # ==============================================================


if __name__ == '__main__':
    main()
