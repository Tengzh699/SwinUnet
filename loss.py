# import torch
# import torch.nn as nn
#
#
# class DiceBCELoss(nn.Module):
#     def __init__(self, smooth=1e-5):
#         super(DiceBCELoss, self).__init__()
#         self.bce_loss = nn.BCEWithLogitsLoss()
#         self.smooth = smooth
#
#     def forward(self, inputs, targets):
#         # 1. 计算 BCE Loss
#         bce = self.bce_loss(inputs, targets)
#
#         # 2. 计算 Dice Loss
#         inputs_sigmoid = torch.sigmoid(inputs)
#         inputs_flat = inputs_sigmoid.view(-1)
#         targets_flat = targets.view(-1)
#
#         intersection = (inputs_flat * targets_flat).sum()
#         dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
#         dice_loss = 1 - dice
#
#         # 组合损失 (论文第四章的 L_total)
#         return dice_loss + bce
import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        # 使用普通的 BCE
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 1. 计算 BCE 损失
        bce = self.bce_loss(inputs, targets)

        # 2. 计算 Dice 损失
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice

        # 【核心优化】：放大 Dice 损失的比重，逼迫模型去抓取病灶像素，拒绝全黑预测
        # 论文中可写：L_total = 0.4 * L_BCE + L_Dice
        return 0.4 * bce + dice_loss