import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from math import exp

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

#SSIMloss
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, channel=1, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # 自动将 window 移动到与 img 相同的设备 (CPU/GPU)
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device).type_as(img1)

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            ssim_val = ssim_map.mean()
        else:
            ssim_val = ssim_map.mean(1).mean(1).mean(1)

        # 我们要最小化 Loss，所以返回 1 - SSIM
        return 1 - ssim_val


# [新增] 计算符合 DeepNIS 定义的 Relative Error (RMSE)
def calc_relative_error(pred, target):
    """
    计算相对误差。
    pred, target: 可以是 Tensor 或 Numpy array
    """
    import torch
    import numpy as np

    # 防止分母为 0
    epsilon = 1e-8

    # 如果是 Tensor (训练时用)
    if torch.is_tensor(pred):
        # norm 默认计算 Frobenius 范数 (即所有元素的平方和开根号)
        numerator = torch.norm(pred - target)
        denominator = torch.norm(target) + epsilon
        return (numerator / denominator).item()

    # 如果是 Numpy (测试时用)
    else:
        numerator = np.linalg.norm(pred - target)
        denominator = np.linalg.norm(target) + epsilon
        return numerator / denominator