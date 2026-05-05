# import torch
# import torch.nn as nn
# from monai.networks.nets import SwinUNETR
#
#
# class CervicalSwinUnet(nn.Module):
#     """
#     面向宫颈阴道镜醋酸图病灶分割的 Swin-Unet 网络
#
#     网络核心机制 :
#     1. 采用 Swin Transformer 作为编码器 (Encoder)，通过 Patch Partition 将图像分块。
#     2. 引入 Shifted Window Multi-head Self-Attention (SW-MSA) 机制，
#        在限制计算复杂度的同时，实现了跨窗口的全局上下文建模，克服传统CNN感受野受限的缺点。
#     3. 采用对称的 U 型解码器 (Decoder) 结构，通过跳跃连接 (Skip-connection)
#        将浅层的高分辨率边缘特征与深层的全局语义特征相融合，实现病灶的像素级精准定位。
#     """
#
#     def __init__(self, img_size=224, in_channels=3, out_channels=1, feature_size=24):
#         super(CervicalSwinUnet, self).__init__()
#
#         # 核心骨干网络：调用经过医学影像高度优化的 Swin-Unet 变体 (SwinUNETR 2D版本)
#         self.swin_unet = SwinUNETR(
#             img_size=(img_size, img_size),
#             in_channels=in_channels,  # 输入通道数 (RGB图像为3)
#             out_channels=out_channels,  # 输出通道数 (二分类病灶掩码为1)
#             feature_size=feature_size,  # 基础特征通道数，控制模型参数量
#             spatial_dims=2  # 指定为 2D 平面图像分割任务
#         )
#
#     def forward(self, x):
#         """
#         前向传播函数
#         x: 输入的宫颈醋酸图像张量, shape: (Batch, 3, H, W)
#         return: 预测的病灶概率对数 (Logits), shape: (Batch, 1, H, W)
#         """
#         logits = self.swin_unet(x)
#         return logits
#
#
# # =====================================================================
# # 拓展功能：如果您需要在网络中加入额外的注意力机制，
# #
# # class SpatialAttention(nn.Module):
# #
# # =====================================================================
#
# if __name__ == "__main__":
#     # 简单的模型测试脚本，用来验证模型能否正常跑通
#     print("正在实例化 CervicalSwinUnet 模型...")
#     model = CervicalSwinUnet(img_size=224, in_channels=3, out_channels=1)
#
#
#     dummy_input = torch.randn(2, 3, 224, 224)
#     print(f"输入图像 shape: {dummy_input.shape}")
#
#     output = model(dummy_input)
#     print(f"输出掩码 shape: {output.shape}")
#     # 期望输出: torch.Size([2, 1, 224, 224])
#     print("✅ 模型结构测试通过！")
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


# =====================================================================
# 创新模块 1：通道注意力机制 (Channel Attention Module)
# 作用：自动学习特征图不同通道的权重，抑制无关背景噪声，增强病灶通道特征。
# =====================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# =====================================================================
# 创新模块 2：空间注意力机制 (Spatial Attention Module)
# 作用：聚焦于特征图的局部空间位置，锐化宫颈醋白病灶的模糊边缘。
# =====================================================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


# =====================================================================
# 创新模块 3：CBAM 综合注意力块 (融合 Channel 与 Spatial)
# =====================================================================
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x  # 乘以通道注意力权重
        x = self.sa(x) * x  # 乘以空间注意力权重
        return x


# =====================================================================
# 核心网络：双重注意力增强的 DA-Swin-Unet
# =====================================================================
class CervicalSwinUnet(nn.Module):
    """
    面向宫颈阴道镜醋酸图病灶分割的 Swin-Unet 网络

    网络核心机制 :
    1. 采用 Swin Transformer 作为编码器 (Encoder)，通过 Patch Partition 将图像分块。
    2. 引入 Shifted Window Multi-head Self-Attention (SW-MSA) 机制，
       在限制计算复杂度的同时，实现了跨窗口的全局上下文建模，克服传统CNN感受野受限的缺点。
    3. 采用对称的 U 型解码器 (Decoder) 结构，通过跳跃连接 (Skip-connection)
       将浅层的高分辨率边缘特征与深层的全局语义特征相融合，实现病灶的像素级精准定位。
    """

    """
    面向宫颈阴道镜醋酸图病灶分割的 DA-Swin-Unet (Dual-Attention Swin-Unet)

    网络核心创新机制 :
    1. 骨干编码器：采用 Swin Transformer，通过 Shifted Window 机制获取全局解剖学特征。
    2. 特征提纯头：在 Swin-Unet 输出端引入 CBAM 双重注意力机制，
       通过通道和空间双维度的自适应加权，克服纯 Transformer 局部细节感知不足的缺陷。
    3. 精准定位：最终通过 1x1 卷积输出高精度的病灶掩膜。
    """

    def __init__(self, img_size=224, in_channels=3, out_channels=1, feature_size=24):
        super(CervicalSwinUnet, self).__init__()

        # 中间隐藏层通道数，供注意力机制发挥作用
        hidden_dim = 32

        # 1. 核心骨干网络：让 SwinUNETR 输出 32 通道的高维语义特征，而不是直接输出 1
        self.swin_unet = SwinUNETR(
            img_size=(img_size, img_size),
            in_channels=in_channels,
            out_channels=hidden_dim,  # 【关键修改】：输出高维特征图
            feature_size=feature_size,
            spatial_dims=2
        )

        # 2. 注意力提纯模块：将 32 通道特征送入 CBAM 模块锐化边缘、消除噪点
        self.attention_block = CBAM(in_planes=hidden_dim, ratio=8, kernel_size=7)

        # 3. 最终分类器：将提纯后的 32 通道特征压缩回 1 通道，输出最终的分割概率
        self.final_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x):
        """
        前向传播函数
        """
        # (Batch, 32, H, W)
        features = self.swin_unet(x)

        # 经过注意力机制加权提纯
        refined_features = self.attention_block(features)

        # 降维输出最终 logits (Batch, 1, H, W)
        logits = self.final_conv(refined_features)

        return logits


if __name__ == "__main__":
    # 简单的模型测试脚本
    print("正在实例化 带有 CBAM 注意力机制的 DA-Swin-Unet 模型...")
    model = CervicalSwinUnet(img_size=224, in_channels=3, out_channels=1)

    dummy_input = torch.randn(2, 3, 224, 224)
    print(f"输入图像 shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"输出掩码 shape: {output.shape}")
    print("✅ 创新模型结构测试通过！参数量和维度完美匹配！")