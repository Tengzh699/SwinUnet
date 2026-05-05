# import os
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
#
#
# class AnnoCervDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         img_name = self.images[idx]
#         mask_name = img_name.replace('.jpg', '.png')
#
#         img_path = os.path.join(self.image_dir, img_name)
#         mask_path = os.path.join(self.mask_dir, mask_name)
#
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         mask_rgba = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#
#         if mask_rgba is not None and len(mask_rgba.shape) == 3 and mask_rgba.shape[2] == 4:
#             alpha_channel = mask_rgba[:, :, 3]
#             mask = (alpha_channel > 0).astype(np.float32)
#         else:
#             mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
#
#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented['image']
#             mask = augmented['mask']
#
#         mask = mask.unsqueeze(0)
#         return image, mask
#
#
# def get_transforms(is_train=True, img_size=224):
#     if is_train:
#         return A.Compose([
#             A.Resize(img_size, img_size),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             # 【核心修改】：使用最新的 Affine 替代 ShiftScaleRotate，消除警告
#             A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-45, 45), p=0.5),
#             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensorV2()
#         ])
#     else:
#         return A.Compose([
#             A.Resize(img_size, img_size),
#             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensorV2()
#         ])
# import os
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
#
#
# class AnnoCervDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         img_name = self.images[idx]
#         mask_name = img_name.replace('.jpg', '.png')
#
#         img_path = os.path.join(self.image_dir, img_name)
#         mask_path = os.path.join(self.mask_dir, mask_name)
#
#         # 1. 读取原图
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         h, w = image.shape[:2]  # 获取原图的真实 高度 和 宽度
#
#         # 2. 读取掩码
#         mask_rgba = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#
#         if mask_rgba is not None and len(mask_rgba.shape) == 3 and mask_rgba.shape[2] == 4:
#             alpha_channel = mask_rgba[:, :, 3]
#             mask = (alpha_channel > 0).astype(np.float32)
#
#             # 【核心修复】：如果标注图尺寸和原图尺寸不一致，强制缩放对齐！
#             # 采用最近邻插值 (INTER_NEAREST) 保证二值化标签(0和1)不被破坏
#             if mask.shape != (h, w):
#                 mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
#         else:
#             # 如果没有标注（健康），生成与原图尺寸一模一样的全 0 掩码
#             mask = np.zeros((h, w), dtype=np.float32)
#
#         # 3. 数据增强
#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented['image']
#             mask = augmented['mask']
#
#         mask = mask.unsqueeze(0)
#         return image, mask
#
#
# def get_transforms(is_train=True, img_size=224):
#     if is_train:
#         return A.Compose([
#             A.Resize(img_size, img_size),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-45, 45), p=0.5),
#             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensorV2()
#         ])
#     else:
#         return A.Compose([
#             A.Resize(img_size, img_size),
#             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensorV2()
#         ])
# import os
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
#
#
# class AnnoCervDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         img_name = self.images[idx]
#         mask_name = img_name.replace('.jpg', '.png')
#
#         img_path = os.path.join(self.image_dir, img_name)
#         mask_path = os.path.join(self.mask_dir, mask_name)
#
#         # 1. 读取原图
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         h, w = image.shape[:2]
#
#         # 2. 读取掩码 (包含透明通道)
#         mask_rgba = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#
#         # 3. 【核心医学逻辑重构】：精准提取高危病变颜色
#         if mask_rgba is not None and len(mask_rgba.shape) == 3 and mask_rgba.shape[2] == 4:
#             # 提取 RGB 三个颜色通道和 Alpha 透明通道
#             bgr = mask_rgba[:, :, :3]
#             alpha = mask_rgba[:, :, 3]
#             rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#
#             R = rgb[:, :, 0].astype(np.int32)
#             G = rgb[:, :, 1].astype(np.int32)
#             B = rgb[:, :, 2].astype(np.int32)
#
#             # 【颜色过滤算法】：
#             # 目标病灶：紫色(Aceto-white), 红色(Vessels), 褐色(Mosaics)
#             # 它们共同的色彩特征是：红色(R)通道值较高，且绿色(G)通道值相对较低。
#             # 要排除的：蓝色(交界区 R很低), 黄色(良性囊肿 G很高), 黑色(腺体 RGB都很低)
#             is_visible = alpha > 0
#             is_pathological = is_visible & (R > 80) & (G < 150)
#
#             # 生成真正的、纯净的病灶二值化掩码
#             mask = is_pathological.astype(np.float32)
#
#             # 尺寸对齐保护
#             if mask.shape != (h, w):
#                 mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
#         else:
#             # 如果是全透明图片（完全健康的病例），生成全 0 掩码
#             mask = np.zeros((h, w), dtype=np.float32)
#
#         # 4. 数据增强
#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented['image']
#             mask = augmented['mask']
#
#         mask = mask.unsqueeze(0)
#         return image, mask
#
#
# def get_transforms(is_train=True, img_size=224):
#     if is_train:
#         return A.Compose([
#             A.Resize(img_size, img_size),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-45, 45), p=0.5),
#             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensorV2()
#         ])
#     else:
#         return A.Compose([
#             A.Resize(img_size, img_size),
#             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensorV2()
#         ])
#
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AnnoCervDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('.jpg', '.png')

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 1. 读取原图 (RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # 2. 读取掩膜 (保留Alpha通道)
        mask_rgba = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask_rgba is not None and len(mask_rgba.shape) == 3 and mask_rgba.shape[2] == 4:
            bgr = mask_rgba[:, :, :3]
            alpha = mask_rgba[:, :, 3]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            R = rgb[:, :, 0].astype(np.int32)
            G = rgb[:, :, 1].astype(np.int32)
            B = rgb[:, :, 2].astype(np.int32)

            # 提取原始紫线 (附加 B > R*0.6 滤除棕色杂线)
            is_purple = (alpha > 0) & (R > G + 20) & (B > G + 20) & (B > R * 0.6)
            original_edge_mask = is_purple.astype(np.uint8)

            if original_edge_mask.sum() > 0:
                # ================= 终极泛洪填充算法 =================
                kernel = np.ones((15, 15), np.uint8)
                dilated_edge = cv2.dilate(original_edge_mask, kernel, iterations=1)

                # 画1像素边框封死贴边缺口
                sealed_edge = dilated_edge.copy()
                cv2.rectangle(sealed_edge, (0, 0), (w - 1, h - 1), 1, thickness=1)

                # 泛洪倒水
                flooded = sealed_edge.copy()
                ff_mask = np.zeros((h + 2, w + 2), np.uint8)
                corners = [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)]
                for pt in corners:
                    if flooded[pt[1], pt[0]] == 0:
                        cv2.floodFill(flooded, ff_mask, pt, 1)

                # 反转提取病灶内部
                inside_mask = (flooded == 0).astype(np.uint8)
                filled_lesion = cv2.bitwise_or(inside_mask, dilated_edge)

                # 腐蚀还原
                final_filled_mask = cv2.erode(filled_lesion, kernel, iterations=1)
                mask = final_filled_mask.astype(np.float32)
                # ====================================================
            else:
                mask = np.zeros((h, w), dtype=np.float32)

            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((h, w), dtype=np.float32)

        # 3. 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 增加通道维度 [1, H, W]
        mask = mask.unsqueeze(0)
        return image, mask


# def get_transforms(is_train=True, img_size=224):
#     if is_train:
#         return A.Compose([
#             A.Resize(img_size, img_size),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-45, 45), p=0.5),
#             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensorV2()
#         ])
#     else:
#         return A.Compose([
#             A.Resize(img_size, img_size),
#             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensorV2()
#         ])
def get_transforms(is_train=True, img_size=224):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            # 【新增1】：颜色抖动，模拟不同设备的打光和醋酸反应的深浅
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            # 【新增2】：医学图像专用的弹性形变，模拟软组织视角变化
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])