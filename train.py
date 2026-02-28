import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from collections import OrderedDict
from glob import glob
import random
import copy
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize

import archs
from dataset_e import ComplexMatDataset, TransformSubset
# [修改1] 确保导入 SSIMLoss
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

    # loss [提示] 这里可以直接传 MSELoss 或 MSE_SSIM
    parser.add_argument('--loss', default='MSE_SSIM',
                        help='loss function: MSELoss | MSE_SSIM')

    # 复数数据集路径
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
                        choices=['Adam', 'SGD'])

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate for KAN layers')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay for KAN layers')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=60, type=int,
                        metavar='N', help='early stopping (default: 60)')
    parser.add_argument('--early_stop_metric', default='mse', choices=['loss', 'mse', 'mae', 'rmse'],
                        help='metric used by early stopping and main checkpoint trigger')

    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--no_kan', action='store_true')
    parser.add_argument('--blank_strategy', default='filter',
                        choices=['none', 'filter', 'downweight'])
    parser.add_argument('--blank_list_file', default='data_quality_report/blank_list.txt')
    parser.add_argument('--blank_weight', default=0.2, type=float)
    parser.add_argument('--fg_alpha', default=0.3, type=float)
    parser.add_argument('--fg_tau', default=0.03, type=float)
    parser.add_argument('--stage2_start_epoch', default=80, type=int)
    parser.add_argument('--use_ema', default=True, type=str2bool)
    parser.add_argument('--ema_decay', default=0.999, type=float)
    parser.add_argument('--ssim_w_start', default=0.05, type=float,
                        help='stage2 SSIM weight at switch epoch')
    parser.add_argument('--ssim_w_end', default=0.2, type=float,
                        help='stage2 SSIM weight at final epoch')
    parser.add_argument('--ssim_schedule', default='linear', choices=['linear', 'constant'])
    parser.add_argument('--use_eca', default=False, type=str2bool,
                        help='enable ECA module in UKAN decoder/fusion')
    parser.add_argument('--eca_kernel', default=3, type=int,
                        help='kernel size of ECA 1D conv')

    config = parser.parse_args()

    return config

# [修改2] 定义 MSE + SSIM 组合损失类
class MSE_SSIM_Loss(nn.Module):
    def __init__(self, channel=2, ssim_weight=0.1):
        super(MSE_SSIM_Loss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=11, channel=channel)
        self.ssim_weight = ssim_weight

    def set_weight(self, w):
        self.ssim_weight = float(w)

    def forward(self, pred, target):
        return self.mse(pred, target) + self.ssim_weight * self.ssim_loss(pred, target)


def compute_stage2_ssim_weight(config, epoch):
    start = config['stage2_start_epoch']
    end_epoch = max(config['epochs'] - 1, start)
    w0 = config['ssim_w_start']
    w1 = config['ssim_w_end']
    if epoch <= start:
        return w0
    if config['ssim_schedule'] == 'constant' or end_epoch == start:
        return w1
    ratio = (epoch - start) / float(end_epoch - start)
    ratio = min(max(ratio, 0.0), 1.0)
    return w0 + (w1 - w0) * ratio


def load_blank_ids(blank_list_file):
    blank_ids = set()
    if not os.path.exists(blank_list_file):
        print(f"Warning: blank list file not found: {blank_list_file}")
        return blank_ids
    with open(blank_list_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()[1:]
    for line in lines:
        first_col = line.split(',')[0].strip()
        if first_col.isdigit():
            blank_ids.add(int(first_col))
    return blank_ids


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


def weighted_sample_mean(loss_per_sample, sample_indices, blank_id_set, blank_weight):
    if (blank_id_set is None) or (len(blank_id_set) == 0):
        return loss_per_sample.mean()
    idx_list = sample_indices.detach().cpu().tolist() if torch.is_tensor(sample_indices) else list(sample_indices)
    is_blank = torch.tensor([i in blank_id_set for i in idx_list], device=loss_per_sample.device, dtype=loss_per_sample.dtype)
    weights = torch.where(is_blank > 0, torch.full_like(is_blank, blank_weight), torch.ones_like(is_blank))
    denom = weights.sum().clamp_min(1e-8)
    return (loss_per_sample * weights).sum() / denom


def foreground_mse_loss(pred, target, tau):
    fg_mask = (target.abs() > tau).float()
    fg_num = (((pred - target) ** 2) * fg_mask).flatten(1).sum(dim=1)
    fg_den = fg_mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return fg_num / fg_den


def compute_total_loss(pred, target, criterion, config, sample_indices, blank_id_set=None):
    if config['blank_strategy'] == 'downweight' and blank_id_set:
        idx_list = sample_indices.detach().cpu().tolist() if torch.is_tensor(sample_indices) else list(sample_indices)
        is_blank = torch.tensor([i in blank_id_set for i in idx_list], device=pred.device, dtype=torch.bool)
        n_blank = int(is_blank.sum().item())
        n_total = int(is_blank.numel())
        n_nonblank = n_total - n_blank

        if n_blank == 0 or n_nonblank == 0:
            base_loss = criterion(pred, target)
        else:
            loss_nonblank = criterion(pred[~is_blank], target[~is_blank])
            loss_blank = criterion(pred[is_blank], target[is_blank])
            denom = n_nonblank + config['blank_weight'] * n_blank
            base_loss = (n_nonblank * loss_nonblank + config['blank_weight'] * n_blank * loss_blank) / max(denom, 1e-8)
    else:
        base_loss = criterion(pred, target)

    if config['fg_alpha'] > 0:
        fg_per_sample = foreground_mse_loss(pred, target, config['fg_tau'])
        if config['blank_strategy'] == 'downweight' and blank_id_set:
            fg_loss = weighted_sample_mean(fg_per_sample, sample_indices, blank_id_set, config['blank_weight'])
        else:
            fg_loss = fg_per_sample.mean()
        return base_loss + config['fg_alpha'] * fg_loss
    return base_loss


def train(config, train_loader, model, criterion, optimizer, blank_id_set=None, ema_model=None):
    """训练一个epoch（回归任务）"""
    device = config['device']
    avg_meters = {
        'loss': AverageMeter(),
        'mse': AverageMeter(),
        'mae': AverageMeter(),
        'rmse': AverageMeter()
    }

    model.train()

    pbar = tqdm(total=len(train_loader), desc='Training')
    for input, target, sample_indices in train_loader:
        input = input.to(device)
        target = target.to(device)

        # Forward
        if config['deep_supervision']:
            outputs = model(input)
            # Deep Supervision 加权 Loss
            weights = [0.4, 0.4, 1.0]
            loss = 0
            for output_item, weight in zip(outputs, weights):
                loss += weight * compute_total_loss(
                    output_item, target, criterion, config, sample_indices, blank_id_set=blank_id_set
                )
            output = outputs[-1]
        else:
            output = model(input)
            loss = compute_total_loss(
                output, target, criterion, config, sample_indices, blank_id_set=blank_id_set
            )

        # 计算回归指标
        with torch.no_grad():
            mse = F.mse_loss(output, target)
            mae = F.l1_loss(output, target)
            rmse = torch.sqrt(mse)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema_model is not None:
            update_ema(ema_model, model, config['ema_decay'])

        # 更新指标
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['mse'].update(mse.item(), input.size(0))
        avg_meters['mae'].update(mae.item(), input.size(0))
        avg_meters['rmse'].update(rmse.item(), input.size(0))

        # 进度条
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
    """验证（回归任务）"""
    device = config['device']
    avg_meters = {
        'loss': AverageMeter(),
        'mse': AverageMeter(),
        'mae': AverageMeter(),
        'rmse': AverageMeter()
    }

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc='Validation')
        for input, target, sample_indices in val_loader:
            input = input.to(device)
            target = target.to(device)

            # Forward
            if config['deep_supervision']:
                outputs = model(input)
                weights = [0.4, 0.4, 1.0]
                loss = 0
                for output_item, weight in zip(outputs, weights):
                    loss += weight * compute_total_loss(
                        output_item, target, criterion, config, sample_indices, blank_id_set=None
                    )
                output = outputs[-1]
            else:
                output = model(input)
                loss = compute_total_loss(
                    output, target, criterion, config, sample_indices, blank_id_set=None
                )

            # 计算回归指标
            mse = F.mse_loss(output, target)
            mae = F.l1_loss(output, target)
            rmse = torch.sqrt(mse)

            # 更新指标
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['mse'].update(mse.item(), input.size(0))
            avg_meters['mae'].update(mae.item(), input.size(0))
            avg_meters['rmse'].update(rmse.item(), input.size(0))

            # 进度条
            postfix = OrderedDict([
                ('val_loss', f"{avg_meters['loss'].avg:.6f}"),
                ('val_mse', f"{avg_meters['mse'].avg:.6f}"),
                ('val_mae', f"{avg_meters['mae'].avg:.6f}"),
                ('val_rmse', f"{avg_meters['rmse'].avg:.6f}"),
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

    # 自动生成实验名
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

    # GPU设置
    cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cpu':
        raise RuntimeError("❌ GPU不可用！")

    print(f"\n{'=' * 60}")
    print(f"GPU信息")
    print(f"{'=' * 60}")
    print(f"使用设备: {device}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"{'=' * 60}\n")

    config['device'] = device

    # [修改3] 损失函数选择逻辑
    criterion_stage1 = nn.MSELoss().to(device)
    if config['loss'] == 'MSELoss':
        criterion_stage2 = nn.MSELoss().to(device)
        print('Using MSE loss')
    elif config['loss'] == 'MSE_SSIM':
        criterion_stage2 = MSE_SSIM_Loss(
            channel=config['num_classes'], ssim_weight=config['ssim_w_start']
        ).to(device)
        print('Using MSE + SSIM loss')
    elif config['loss'] == 'L1Loss':
        criterion_stage2 = nn.L1Loss().to(device)
        print('Using L1 loss')
    elif config['loss'] == 'SmoothL1Loss':
        criterion_stage2 = nn.SmoothL1Loss().to(device)
        print('Using SmoothL1 loss')
    else:
        criterion_stage2 = nn.MSELoss().to(device)
        print(f"Unknown loss {config['loss']}, fallback to MSE")
    print(f"Two-stage training: stage1=MSE, stage2={config['loss']}, switch@epoch={config['stage2_start_epoch']}")

    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config['no_kan'],
        use_eca=config['use_eca'],
        eca_kernel=config['eca_kernel']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ 模型创建成功")
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"  模型位置: {next(model.parameters()).device}\n")


    ema_model = None
    if config['use_ema']:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for p_ema in ema_model.parameters():
            p_ema.requires_grad_(False)
        print(f"EMA enabled: decay={config['ema_decay']}")

    # 参数分组（KAN层用不同的学习率）
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
        print(f"✓ KAN层参数: {sum(p.numel() for p in kan_params) / 1e6:.2f}M, lr={config['kan_lr']}")

    if len(other_params) > 0:
        param_groups.append({
            'params': other_params,
            'lr': config['lr'],
            'weight_decay': config['weight_decay']
        })
        print(f"✓ 其他参数: {sum(p.numel() for p in other_params) / 1e6:.2f}M, lr={config['lr']}\n")

    # Optimizer
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            param_groups,
            momentum=config['momentum'],
            nesterov=config['nesterov']
        )
    else:
        raise NotImplementedError

    # Scheduler
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config['factor'],
            patience=config['patience'], verbose=True, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(e) for e in config['milestones'].split(',')],
            gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # 备份代码
    shutil.copy2(__file__, f'{output_dir}/{exp_name}/')
    if os.path.exists('archs.py'):
        shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')

    # 加载数据集
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

    # 数据增强
    train_transform = Compose([
        RandomRotate90(),
        transforms.HorizontalFlip(p=0.5),
        transforms.VerticalFlip(p=0.5),
        Resize(config['input_h'], config['input_w']),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
    ])

    # 创建完整数据集
    full_dataset = ComplexMatDataset(
        real_img_path=real_img_path,
        imag_img_path=imag_img_path,
        real_label_path=real_label_path,
        imag_label_path=imag_label_path,
        img_size=(config['original_img_size'], config['original_img_size']),
        transform=None,
        normalize_method='z-score'
    )

    # 划分训练集和验证集
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    np.random.seed(config['dataseed'])
    np.random.shuffle(indices)

    # 假设 dataset_size = 2000
    # Test:  0   - 199   (10%) -> 留给 test.py 用，train.py 绝对不能碰！
    # Val:   200 - 399   (10%) -> 用于验证
    # Train: 400 - 1999  (80%) -> 用于训练

    test_split = int(np.floor(0.1 * dataset_size))  # 10%
    val_split = int(np.floor(0.1 * dataset_size))  # 10%

    # 这里的切片逻辑非常关键：
    test_indices = indices[:test_split]  # 0~199
    val_indices = indices[test_split: test_split + val_split]  # 200~399
    train_indices = indices[test_split + val_split:]  # 400~1999


    blank_id_set = set()
    if config['blank_strategy'] in ['filter', 'downweight']:
        blank_id_set = load_blank_ids(config['blank_list_file'])
        print(f"Blank strategy={config['blank_strategy']}, total blank IDs={len(blank_id_set)}")
        if config['blank_strategy'] == 'filter' and len(blank_id_set) > 0:
            before_cnt = len(train_indices)
            train_indices = [i for i in train_indices if i not in blank_id_set]
            print(f"Filtered blank samples in train: {before_cnt} -> {len(train_indices)}")
        elif config['blank_strategy'] == 'downweight':
            print(f"Downweight blank samples with weight={config['blank_weight']}")

    print("\n[Auto-Fix] Calculating normalization stats using Training indices...")
    full_dataset.calculate_normalization_stats(train_indices)
    full_dataset.save_stats(os.path.join(output_dir, exp_name, 'norm_stats.json'))

    # train.py 只需要用到 train 和 val
    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_transform)

    print(f"\nDataset split (8:1:1):")
    print(f"  Total:      {dataset_size}")
    print(f"  Training:   {len(train_dataset)} (用于训练)")
    print(f"  Validation: {len(val_dataset)} (用于早停)")
    print(f"  Test:       {len(test_indices)} (保留给 test.py)")

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

    # 训练日志
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
    ])

    best_loss = float('inf')
    trigger = 0

    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练（回归任务）")
    print("=" * 60 + "\n")

    best_loss = float('inf')
    best_mae = float('inf')
    best_mse = float('inf')
    best_rmse = float('inf')
    best_monitor = float('inf')

    best_mse_ckpt = float('inf')
    trigger = 0  # 早停计数器
    for epoch in range(config['epochs']):
        print(f'\nEpoch [{epoch}/{config["epochs"]}]')
        print('-' * 60)

        # Stage switch
        if epoch < config['stage2_start_epoch']:
            active_criterion = criterion_stage1
            stage_name = 'stage1_mse'
        else:
            active_criterion = criterion_stage2
            stage_name = 'stage2_target_loss'
            if config['loss'] == 'MSE_SSIM' and hasattr(criterion_stage2, 'set_weight'):
                curr_w = compute_stage2_ssim_weight(config, epoch)
                criterion_stage2.set_weight(curr_w)

        # Reset loss-based early-stopping baseline at stage-2 boundary,
        # because stage-2 loss scale differs from stage-1.
        if epoch == config['stage2_start_epoch']:
            best_loss = float('inf')
            best_monitor = float('inf')
            trigger = 0
            print("=> Entered stage2: reset best_loss and early-stopping trigger.")

        # Train
        train_log = train(
            config, train_loader, model, active_criterion, optimizer,
            blank_id_set=blank_id_set if config['blank_strategy'] == 'downweight' else None,
            ema_model=ema_model
        )

        # Validate
        eval_model = ema_model if ema_model is not None else model
        val_log = validate(config, val_loader, eval_model, active_criterion)
        # 如果当前的 MAE 比历史最好的还小，就更新历史最好
        if val_log['mae'] < best_mae: best_mae = val_log['mae']
        if val_log['mse'] < best_mse: best_mse = val_log['mse']
        if val_log['rmse'] < best_rmse: best_rmse = val_log['rmse']

        # 【核心早停逻辑】 这里我们以 Loss (总误差) 为主要标准来决定要不要保存模型
        monitor_key = f"val_{config['early_stop_metric']}"
        monitor_val = val_log[config['early_stop_metric']]
        if monitor_val < best_monitor:
            print(f"=> [Epoch {epoch}] Saved best model! ({monitor_key}: {monitor_val:.6f})")

            # ???? Loss / Monitor
            best_loss = val_log['loss']
            best_monitor = monitor_val

            # ??????
            torch.save(eval_model.state_dict(), f'{output_dir}/{exp_name}/model.pth')

            # ???????????
            trigger = 0
        else:
            # ??????? +1
            trigger += 1
            print(f"=> No improvement for {trigger} epochs on {monitor_key}.")

        if val_log['mse'] < best_mse_ckpt:
            best_mse_ckpt = val_log['mse']
            torch.save(eval_model.state_dict(), f'{output_dir}/{exp_name}/model_best_mse.pth')
            print(f"=> [Epoch {epoch}] Saved best-MSE model! (val_mse: {best_mse_ckpt:.6f})")
        # 打印一下当前的最好模型
        print(
            f'   [{stage_name}] Best Results -> Loss: {best_loss:.4f} | MAE: {best_mae:.4f} | MSE: {best_mse:.4f} | RMSE: {best_rmse:.4f}')

        # 触发早停（Early Stopping）
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print(f"=> Early stopping triggered! (Best Loss: {best_loss:.6f})")
            break

        # 更新学习率
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        # 打印结果
        print(f'\nResults:')
        print(
            f"  Train - Loss: {train_log['loss']:.6f}, MSE: {train_log['mse']:.6f}, MAE: {train_log['mae']:.6f}, RMSE: {train_log['rmse']:.6f}")
        print(
            f"  Val   - Loss: {val_log['loss']:.6f}, MSE: {val_log['mse']:.6f}, MAE: {val_log['mae']:.6f}, RMSE: {val_log['rmse']:.6f}")

        # 记录日志
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
        my_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        if config['loss'] == 'MSE_SSIM' and hasattr(criterion_stage2, 'ssim_weight'):
            my_writer.add_scalar('loss_weight/ssim', criterion_stage2.ssim_weight, epoch)
        my_writer.add_scalar('best/loss', best_loss, epoch)
        my_writer.add_scalar('best/mae', best_mae, epoch)
        my_writer.add_scalar('best/mse', best_mse, epoch)
        my_writer.add_scalar('best/rmse', best_rmse, epoch)


        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"训练完成！最佳验证损失: {best_loss:.6f}")
    print("=" * 60)

    my_writer.close()
    # ================= 保存最佳结果到 txt 文件 =================
    result_path = f'{output_dir}/{exp_name}/best_results.txt'

    print(f"正在保存最佳指标到: {result_path}")

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
        f.write("-" * 30 + "\n")

    print("✓ 结果已保存")
    # ==============================================================


if __name__ == '__main__':
    main()
