import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target, threshold=0.5):
    """
    计算IoU（支持多通道）

    Args:
        output: (B, C, H, W) - 模型输出（logits或sigmoid后的概率）
        target: (B, C, H, W) - 真实标签（0-1）
        threshold: 二值化阈值

    Returns:
        iou: 平均IoU
        dice: 平均Dice
        detail: 每个通道的详细指标
    """
    # 确保output是概率
    if output.min() < 0 or output.max() > 1:
        output = torch.sigmoid(output)

    # 二值化
    output = (output > threshold).float()
    target = (target > threshold).float()

    # 打印调试信息（第一次）
    if not hasattr(iou_score, 'printed'):
        print(f"\n[IoU Debug]")
        print(f"  Output shape: {output.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  Output unique: {torch.unique(output)}")
        print(f"  Target unique: {torch.unique(target)}")
        iou_score.printed = True

    batch_size = output.shape[0]
    num_classes = output.shape[1]

    # 计算每个样本、每个通道的IoU
    ious = []
    dices = []

    for b in range(batch_size):
        for c in range(num_classes):
            pred = output[b, c].flatten()
            gt = target[b, c].flatten()

            # 交集和并集
            intersection = (pred * gt).sum()
            union = pred.sum() + gt.sum() - intersection

            # IoU
            if union > 0:
                iou = intersection / union
            else:
                iou = torch.tensor(1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0)

            # Dice
            if (pred.sum() + gt.sum()) > 0:
                dice = 2 * intersection / (pred.sum() + gt.sum())
            else:
                dice = torch.tensor(1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0)

            ious.append(iou.item())
            dices.append(dice.item())

    # 平均
    mean_iou = np.mean(ious)
    mean_dice = np.mean(dices)

    # 详细信息
    detail = {
        'iou_per_channel': ious,
        'dice_per_channel': dices,
        'num_samples': batch_size,
        'num_channels': num_classes
    }

    return mean_iou, mean_dice, detail


def dice_coef(output, target, threshold=0.5):
    """
    单独的Dice系数计算
    """
    _, dice, _ = iou_score(output, target, threshold)
    return dice
