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
import torch.optim as optim
import yaml

from albumentations import transforms
from albumentations.augmentations import geometric

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize

import archs

import losses
from dataset import Dataset
from dataset_e import ComplexMatDataset,TransformSubset

from metrics import iou_score, indicators

from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter

import shutil
import os
import subprocess

from pdb import set_trace as st
import albumentations as A


ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    # ========== 模型相关 ==========
    parser.add_argument('--name', default='KANUNet_experiment',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='KANUNet')

    # ========== 数据集相关 ==========
    parser.add_argument('--dataset', default='complex_chi',
                        help='dataset name: complex_chi for complex magnetic susceptibility data')
    parser.add_argument('--data_dir', default='inputs',
                        help='directory containing the .mat files')

    # 复数数据的4个文件
    parser.add_argument('--real_img_file', default='/data/input/chi_all_real_64.mat',
                        help='real part image file')
    parser.add_argument('--imag_img_file', default='/data/input/chi_all_imag_64.mov',
                        help='imaginary part image file')
    parser.add_argument('--real_mask_file', default='/data/lable/chi0_all_real_64.mat',
                        help='real part mask file')
    parser.add_argument('--imag_mask_file', default='/data/lable/chi0_all_imag_64.mat',
                        help='imaginary part mask file')

    # .mat文件变量名
    parser.add_argument('--img_var_name', default='chi_all_real',
                        help='variable name for images in .mat file')
    parser.add_argument('--mask_var_name', default='chi_all_real_mask',
                        help='variable name for masks in .mat file')

    # 原始图片尺寸
    parser.add_argument('--original_img_size', default=64, type=int,
                        help='original image size in .mat file (e.g., 64 for 64x64)')

    parser.add_argument('--output_dir', default='outputs',
                        help='output directory for results')

    # ========== 输入尺寸 ==========
    parser.add_argument('--input_channels', default=2, type=int,
                        help='input channels (2 for real+imaginary)')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes (2 for real+imaginary masks)')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width for training')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height for training')

    # ========== 训练超参数 ==========
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 4)')

    # 学习率
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_lr', default=1e-3, type=float,
                        help='learning rate for KAN layers')

    # 优化器
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD', 'AdamW'],
                        help='optimizer')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum for SGD')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')

    # 学习率调度
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau',
                                 'MultiStepLR', 'ConstantLR'])

    # ========== 损失函数 ==========
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=['BCEWithLogitsLoss', 'BCEDiceLoss'])

    # ========== 早停 ==========
    parser.add_argument('--early_stopping', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')

    # ========== 数据划分 ==========
    parser.add_argument('--dataseed', default=42, type=int,
                        help='random seed for data split')
    parser.add_argument('--val_split', default=0.2, type=float,
                        help='validation split ratio (default: 0.2 = 20%)')

    # ========== 其他 ==========
    parser.add_argument('--num_workers', default=0, type=int,
                        help='number of workers for data loading')

    config = parser.parse_args()
    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)

            iou, dice, _ = iou_score(outputs[-1], target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(outputs[-1], target)
            
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

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
    seed_torch()  # 设置随机种子，确保实验可复现
    config = vars(parse_args())  # 解析命令行参数并转换为字典格式

    exp_name = config.get('name')  # 获取实验名称
    output_dir = config.get('output_dir')  # 获取输出目录路径

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')  # 创建TensorBoard日志记录器

    # 如果没有指定实验名称，则自动生成
    if config['name'] is None:
        if config['deep_supervision']:  # 如果使用深度监督
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])  # 名称包含wDS(with Deep Supervision)
        else:  # 如果不使用深度监督
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])  # 名称包含woDS(without Deep Supervision)

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)  # 创建实验输出目录，如果已存在则不报错

    # 打印所有配置参数
    print('-' * 60)
    print('Configuration:')
    for key in config:
        print(f'  {key}: {config[key]}')
    print('-' * 60)

    # 将配置保存为YAML文件
    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)  # 保存配置到YAML文件，便于复现实验

    # 定义损失函数(criterion)
    if config['loss'] == 'BCEWithLogitsLoss':  # 如果使用BCE损失
        criterion = nn.BCEWithLogitsLoss().cuda()  # 使用PyTorch内置的BCE损失并移到GPU
    else:  # 其他损失函数
        criterion = losses.__dict__[config['loss']]().cuda()  # 从自定义losses模块动态加载并移到GPU

    cudnn.benchmark = True  # 启用cuDNN自动优化，加速卷积运算

    # 创建模型
    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'],
                                           embed_dims=config['input_list'], no_kan=config['no_kan'])  # 根据配置动态创建模型架构

    model = model.cuda()  # 将模型移到GPU

    param_groups = []  # 初始化参数分组列表，用于设置不同的学习率

    kan_fc_params = []  # KAN层全连接参数列表（未使用，仅作标记）
    other_params = []  # 其他参数列表（未使用，仅作标记）

    # 遍历模型所有参数，为不同层设置不同学习率
    for name, param in model.named_parameters():
        # print(name, "=>", param.shape)  # 调试用：打印参数名和形状
        if 'layer' in name.lower() and 'fc' in name.lower():  # 如果参数名包含'layer'和'fc'（KAN层的全连接层）
            # kan_fc_params.append(name)  # 标记为KAN参数（已注释）
            param_groups.append(
                {'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']})  # 为KAN层设置更高的学习率
        else:  # 其他参数
            # other_params.append(name)  # 标记为其他参数（已注释）
            param_groups.append(
                {'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})  # 使用标准学习率

    # st()  # 调试断点（已注释）
    # 创建优化器
    if config['optimizer'] == 'Adam':  # 如果使用Adam优化器
        optimizer = optim.Adam(param_groups)  # 使用参数分组创建Adam优化器


    elif config['optimizer'] == 'SGD':  # 如果使用SGD优化器
        optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'],
                              weight_decay=config['weight_decay'])  # 创建SGD优化器
    else:  # 其他优化器
        raise NotImplementedError  # 抛出未实现异常

    # 创建学习率调度器
    if config['scheduler'] == 'CosineAnnealingLR':  # 余弦退火调度器
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])  # 学习率按余弦曲线从初始值降到最小值
    elif config['scheduler'] == 'ReduceLROnPlateau':  # 自适应学习率调度器
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])  # 验证损失不下降时降低学习率
    elif config['scheduler'] == 'MultiStepLR':  # 多步学习率调度器
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])  # 在指定epoch降低学习率
    elif config['scheduler'] == 'ConstantLR':  # 恒定学习率
        scheduler = None  # 不使用调度器
    else:  # 其他调度器
        raise NotImplementedError  # 抛出未实现异常

    # 备份训练脚本和模型架构文件到输出目录
    shutil.copy2('train.py', f'{output_dir}/{exp_name}/')  # 复制训练脚本
    shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')  # 复制模型架构文件

    # ========== 数据加载 ==========
    print('\n' + '=' * 60)
    print('Loading Dataset')
    print('=' * 60)

    # 定义数据增强
    train_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    ])

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
    ])

    # 构建完整的文件路径
    real_img_path = os.path.join(config['data_dir'], config['real_img_file'])
    imag_img_path = os.path.join(config['data_dir'], config['imag_img_file'])
    real_mask_path = os.path.join(config['data_dir'], config['real_mask_file'])
    imag_mask_path = os.path.join(config['data_dir'], config['imag_mask_file'])

    # 检查文件是否存在
    print('\nChecking files...')
    for path in [real_img_path, imag_img_path, real_mask_path, imag_mask_path]:
        if os.path.exists(path):
            print(f'  ✓ {path}')
        else:
            print(f'  ✗ {path} NOT FOUND!')
            raise FileNotFoundError(f"File not found: {path}")

    # 创建完整数据集
    full_dataset = ComplexMatDataset(
        real_img_path=real_img_path,
        imag_img_path=imag_img_path,
        real_mask_path=real_mask_path,
        imag_mask_path=imag_mask_path,
        img_var_name=config['img_var_name'],
        mask_var_name=config['mask_var_name'],
        img_size=(config['original_img_size'], config['original_img_size']),
    )

    # 划分训练/验证集
    total_samples = len(full_dataset)
    indices = np.arange(total_samples)

    train_indices, val_indices = train_test_split(
        indices,
        test_size=config['val_split'],
        random_state=config['dataseed']
    )

    print('\n' + '-' * 60)
    print('Dataset Split:')
    print(f'  Total samples: {total_samples}')
    print(f'  Training samples: {len(train_indices)} ({len(train_indices) / total_samples * 100:.1f}%)')
    print(f'  Validation samples: {len(val_indices)} ({len(val_indices) / total_samples * 100:.1f}%)')
    print(f'  Train indices: {train_indices}')
    print(f'  Val indices: {val_indices}')
    print('-' * 60)

    # 创建子集
    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_transform)

    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True
    )

    print(f'\nDataLoader created:')
    print(f'  Train batches: {len(train_loader)}')
    print(f'  Val batches: {len(val_loader)}')

    # 测试一个batch
    print(f'\nTesting DataLoader...')
    for img_batch, mask_batch, meta_batch in train_loader:
        print(f'  Image batch shape: {img_batch.shape}')  # (batch, 2, H, W)
        print(f'  Mask batch shape: {mask_batch.shape}')  # (batch, 2, H, W)
        print(f'  Value range: [{img_batch.min():.4f}, {img_batch.max():.4f}]')
        break

    # 创建训练日志字典
    log = OrderedDict([
        ('epoch', []),  # 记录epoch
        ('lr', []),  # 记录学习率
        ('loss', []),  # 记录训练损失
        ('iou', []),  # 记录训练IoU
        ('val_loss', []),  # 记录验证损失
        ('val_iou', []),  # 记录验证IoU
        ('val_dice', []),  # 记录验证Dice系数
    ])

    best_iou = 0  # 初始化最佳IoU
    best_dice = 0  # 初始化最佳Dice系数
    trigger = 0  # 初始化早停计数器
    # 开始训练循环
    for epoch in range(config['epochs']):  # 遍历所有epoch
        print('Epoch [%d/%d]' % (epoch, config['epochs']))  # 打印当前epoch

        # 训练一个epoch
        train_log = train(config, train_loader, model, criterion, optimizer)  # 调用训练函数
        # 在验证集上评估
        val_log = validate(config, val_loader, model, criterion)  # 调用验证函数

        # 更新学习率
        if config['scheduler'] == 'CosineAnnealingLR':  # 如果使用余弦退火
            scheduler.step()  # 每个epoch自动调整学习率
        elif config['scheduler'] == 'ReduceLROnPlateau':  # 如果使用自适应调度器
            scheduler.step(val_log['loss'])  # 根据验证损失调整学习率

        # 打印训练和验证指标
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        # 记录日志
        log['epoch'].append(epoch)  # 记录epoch
        log['lr'].append(config['lr'])  # 记录学习率
        log['loss'].append(train_log['loss'])  # 记录训练损失
        log['iou'].append(train_log['iou'])  # 记录训练IoU
        log['val_loss'].append(val_log['loss'])  # 记录验证损失
        log['val_iou'].append(val_log['iou'])  # 记录验证IoU
        log['val_dice'].append(val_log['dice'])  # 记录验证Dice系数

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)  # 保存日志到CSV文件

        # 记录TensorBoard日志
        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)  # 记录训练损失
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)  # 记录训练IoU
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)  # 记录验证损失
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)  # 记录验证IoU
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)  # 记录验证Dice系数

        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)  # 记录历史最佳IoU
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)  # 记录历史最佳Dice系数

        trigger += 1  # 早停计数器加1

        # 如果验证IoU提升，保存模型
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')  # 保存模型参数
            best_iou = val_log['iou']  # 更新最佳IoU
            best_dice = val_log['dice']  # 更新最佳Dice系数
            print("=> saved best model")  # 提示保存模型
            print('IoU: %.4f' % best_iou)  # 打印最佳IoU
            print('Dice: %.4f' % best_dice)  # 打印最佳Dice系数
            trigger = 0  # 重置早停计数器

        # 早停机制
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:  # 如果连续N个epoch没有提升
            print("=> early stopping")  # 提示早停
            break  # 跳出训练循环

        torch.cuda.empty_cache()  # 清空GPU缓存，防止内存泄漏


if __name__ == '__main__':
    main()
