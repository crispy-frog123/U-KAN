import argparse
import os
from collections import OrderedDict
from glob import glob
import random
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
import losses
from dataset_e import ComplexMatDataset, TransformSubset
from utils import AverageMeter, str2bool
from torch.utils.tensorboard import SummaryWriter

import shutil
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='random seed for dataset split')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)

    parser.add_argument('--input_channels', default=2, type=int,
                        help='input channels ')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')

    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='MSELoss',
                        help='loss function for regression')

    # 复数数据集路径
    parser.add_argument('--data_dir', default='inputs', help='dataset base directory')

    parser.add_argument('--real_img_file', default='input/chi_all_real_64.mat',
                        help='real part image .mat file')
    parser.add_argument('--imag_img_file', default='input/chi_all_imag_64.mat',
                        help='imaginary part image .mat file')
    parser.add_argument('--real_mask_file', default='label/chi0_all_real_64.mat',
                        help='real part mask .mat file')
    parser.add_argument('--imag_mask_file', default='label/chi0_all_imag_64.mat',
                        help='imaginary part mask .mat file')

    parser.add_argument('--img_var_name', default='chi_all_real',
                        help='variable name in .mat file for images')
    parser.add_argument('--mask_var_name', default='chi0_all_real',
                        help='variable name in .mat file for masks')

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
    parser.add_argument('--early_stopping', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--no_kan', action='store_true')

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
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
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        # Forward
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            output = outputs[-1]
        else:
            output = model(input)
            loss = criterion(output, target)

        # 计算回归指标
        with torch.no_grad():
            mse = F.mse_loss(output, target)
            mae = F.l1_loss(output, target)
            rmse = torch.sqrt(mse)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # Forward
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                output = outputs[-1]
            else:
                output = model(input)
                loss = criterion(output, target)

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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    # 自动生成实验名
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = 'regression_%s_wDS' % config['arch']
        else:
            config['name'] = 'regression_%s_woDS' % config['arch']
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

    # 损失函数（回归）
    if config['loss'] == 'MSELoss':
        criterion = nn.MSELoss().to(device)
        print("✓ 使用MSE损失（回归任务）")
    elif config['loss'] == 'L1Loss':
        criterion = nn.L1Loss().to(device)
        print("✓ 使用L1损失（回归任务）")
    elif config['loss'] == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss().to(device)
        print("✓ 使用SmoothL1损失（回归任务）")
    else:
        criterion = nn.MSELoss().to(device)
        print("⚠️  未指定损失函数，默认使用MSE")

    # 创建模型
    print("\n创建模型...")
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config['no_kan']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ 模型创建成功")
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"  模型位置: {next(model.parameters()).device}\n")

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
    real_mask_path = build_path(config['data_dir'], config['real_mask_file'])
    imag_mask_path = build_path(config['data_dir'], config['imag_mask_file'])

    print(f"\nData files:")
    print(f"  Real image:  {real_img_path}")
    print(f"  Imag image:  {imag_img_path}")
    print(f"  Real target: {real_mask_path}")
    print(f"  Imag target: {imag_mask_path}")

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
        real_mask_path=real_mask_path,
        imag_mask_path=imag_mask_path,
        img_size=(config['original_img_size'], config['original_img_size']),
        transform=None,
        normalize_method='z-score'
    )

    # 划分训练集和验证集
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    np.random.seed(config['dataseed'])
    np.random.shuffle(indices)

    split = int(np.floor(config['val_split'] * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    # 创建Subset
    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_transform)

    print(f"\nDataset split:")
    print(f"  Total:      {dataset_size} samples")
    print(f"  Training:   {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print("=" * 60 + "\n")

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

    for epoch in range(config['epochs']):
        print(f'\nEpoch [{epoch}/{config["epochs"]}]')
        print('-' * 60)

        # Train
        train_log = train(config, train_loader, model, criterion, optimizer)

        # Validate
        val_log = validate(config, val_loader, model, criterion)

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

        # 保存最佳模型
        if val_log['loss'] < best_loss:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_loss = val_log['loss']
            print(f"\n=> Saved best model (val_loss: {best_loss:.6f})")
            trigger = 0
        else:
            trigger += 1

        # Early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print(f"\n=> Early stopping triggered (no improvement for {trigger} epochs)")
            break

        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"训练完成！最佳验证损失: {best_loss:.6f}")
    print("=" * 60)

    my_writer.close()


if __name__ == '__main__':
    main()
