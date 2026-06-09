import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class FocalLoss(nn.Module):
    """
    原始的 FocalLoss 实现保持不变
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class DiceFocalLoss(nn.Module):
    def __init__(self, lambda_dice=1.0, lambda_focal=1.0,
                 alpha=None, gamma=2, smooth=1e-5, size_average=True):
        super(DiceFocalLoss, self).__init__()
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, smooth=smooth, size_average=size_average)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, inputs, targets):
        # 计算 Focal
        focal_loss = self.focal(inputs, targets)

        # 计算 Dice (取第二列异常概率)
        if inputs.shape[1] == 2:
            anomaly_prob = inputs[:, 1]
        else:
            anomaly_prob = inputs
        dice_loss = self.dice(anomaly_prob, targets)

        # 加权求和
        total_loss = (self.lambda_focal * focal_loss) + (self.lambda_dice * dice_loss)

        # 返回: (总loss用于反向传播, focal数值用于打印, dice数值用于打印)
        return total_loss, focal_loss.item(), dice_loss.item()
