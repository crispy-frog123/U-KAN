import torch
import torch.nn.functional as F
import math
import argparse
import numpy as np
from torch.autograd import Variable


# ==========================================================
#  1. 基础工具类 (AverageMeter, str2bool)
# ==========================================================

class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def str2bool(v):
    """用于 argparse 的布尔类型解析"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ==========================================================
#  2. SSIM Loss 实现 (Differentiable)
# ==========================================================

def gaussian(window_size, sigma):
    """生成一维高斯核"""
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """生成二维高斯窗口，用于卷积"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    SSIM 计算核心逻辑
    img1, img2: [Batch, Channel, H, W]
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    """
    SSIM 损失函数: Loss = 1 - SSIM
    """

    def __init__(self, window_size=11, size_average=True, channel=2):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # 确保 window 在正确的 device 上
        if img1.is_cuda:
            self.window = self.window.cuda(img1.get_device())
        self.window = self.window.type_as(img1)

        # 确保通道数匹配 (自动处理如果输入通道变了的情况)
        if self.channel != img1.size(1):
            self.channel = img1.size(1)
            self.window = create_window(self.window_size, self.channel)
            if img1.is_cuda:
                self.window = self.window.cuda(img1.get_device())
            self.window = self.window.type_as(img1)

        return 1 - _ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)


# ==========================================================
#  3. 其他辅助计算 (Relative Error)
# ==========================================================

def calc_relative_error(pred, target):
    """
    计算相对误差 (Relative Error)
    RE = ||pred - target||_F / ||target||_F
    """
    # 确保是 tensor
    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred)
    if not torch.is_tensor(target):
        target = torch.from_numpy(target)

    # 展平
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)
    target_norm = torch.norm(target_flat, p=2, dim=1)

    # 防止除以0
    target_norm = torch.where(target_norm < 1e-6, torch.ones_like(target_norm) * 1e-6, target_norm)

    re = diff_norm / target_norm
    return re.mean().item()