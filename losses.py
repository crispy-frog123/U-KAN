import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice Loss
    支持分通道计算，可设置每个通道的权重
    """

    def __init__(self, channel_weights=None):
        """
        Args:
            channel_weights: list或None
                例如 [0.6, 0.4] 表示实部占60%，虚部占40%
                None表示平均权重 [0.5, 0.5]
        """
        super().__init__()
        self.channel_weights = channel_weights

    def forward(self, input, target):
        """
        分通道计算loss，按权重加权平均
        """
        num_channels = input.size(1)

        # 默认权重：平均
        if self.channel_weights is None:
            weights = [1.0 / num_channels] * num_channels
        else:
            weights = self.channel_weights
            assert len(weights) == num_channels, \
                f"权重数量({len(weights)})与通道数({num_channels})不匹配"

        total_loss = 0

        # 对每个通道分别计算loss
        for c in range(num_channels):
            # 提取当前通道
            input_c = input[:, c:c + 1, :, :]
            target_c = target[:, c:c + 1, :, :]

            # BCE Loss
            bce = F.binary_cross_entropy_with_logits(input_c, target_c)

            # Dice Loss
            smooth = 1e-5
            input_c = torch.sigmoid(input_c)
            num = target_c.size(0)

            input_c = input_c.reshape(num, -1)
            target_c = target_c.reshape(num, -1)

            intersection = (input_c * target_c)
            dice = (2. * intersection.sum(1) + smooth) / (input_c.sum(1) + target_c.sum(1) + smooth)
            dice = 1 - dice.sum() / num

            # 当前通道的loss
            loss_c = 0.5 * bce + dice

            # 加权累加
            total_loss += weights[c] * loss_c

        return total_loss



class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
