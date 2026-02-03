import argparse
import torch.nn as nn

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