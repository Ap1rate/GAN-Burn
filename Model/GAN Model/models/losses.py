# models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # x 方向梯度
        pred_dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        targ_dx = target[:, :, :, :-1] - target[:, :, :, 1:]
        # y 方向梯度
        pred_dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]
        targ_dy = target[:, :, :-1, :] - target[:, :, 1:, :]
        # 用 L1 loss 约束边缘一致性
        loss = F.l1_loss(pred_dx, targ_dx) + F.l1_loss(pred_dy, targ_dy)
        return loss
