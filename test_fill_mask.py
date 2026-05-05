# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.ndimage as ndimage
#
# # ================= 配置路径 =================
# # 我们去验证集里挑几张图片来测试
# VAL_IMG_DIR = "./data/val/images"
# VAL_MASK_DIR = "./data/val/masks"
# TEST_SAVE_DIR = "./test_masks_output"
#
#
# # ============================================
#
# def test_mask_filling():
#     os.makedirs(TEST_SAVE_DIR, exist_ok=True)
#
#     # 获取所有的 jpg 图片
#     test_images = [f for f in os.listdir(VAL_IMG_DIR) if f.endswith('.jpg')]
#
#     count = 0
#     for img_name in test_images:
#         mask_name = img_name.replace('.jpg', '.png')
#         img_path = os.path.join(VAL_IMG_DIR, img_name)
#         mask_path = os.path.join(VAL_MASK_DIR, mask_name)
#
#         # 1. 读取原图
#         image = cv2.imread(img_path)
#         if image is None: continue
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # 2. 读取专家标注的掩膜
#         mask_rgba = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#
#         if mask_rgba is not None and len(mask_rgba.shape) == 3 and mask_rgba.shape[2] == 4:
#             bgr = mask_rgba[:, :, :3]
#             alpha = mask_rgba[:, :, 3]
#             rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#
#             R = rgb[:, :, 0].astype(np.int32)
#             G = rgb[:, :, 1].astype(np.int32)
#             B = rgb[:, :, 2].astype(np.int32)
#
#             # 【提取原始紫线】
#             is_purple = (alpha > 0) & (R > G + 20) & (B > G + 20)
#             original_edge_mask = is_purple.astype(np.uint8)
#
#             # 如果这张图没有紫色标注，跳过，找下一张
#             if original_edge_mask.sum() == 0:
#                 continue
#
#             # ================= 核心填充算法测试 =================
#             # 步骤 A: 闭运算（加粗并连接断点）
#             kernel = np.ones((7, 7), np.uint8)
#             closed_edge = cv2.morphologyEx(original_edge_mask, cv2.MORPH_CLOSE, kernel)
#
#             # 步骤 B: 边缘补零（防止病灶贴边导致水漏出去）
#             padded_edge = np.pad(closed_edge, pad_width=1, mode='constant', constant_values=0)
#
#             # 步骤 C: SciPy 拓扑学空洞填充
#             filled_padded = ndimage.binary_fill_holes(padded_edge)
#
#             # 步骤 D: 裁掉补的那一圈，恢复原尺寸
#             final_filled_mask = filled_padded[1:-1, 1:-1].astype(np.float32)
#             # ====================================================
#
#             # 3. 画图对比
#             plt.figure(figsize=(18, 6))
#
#             # 子图1：原图
#             plt.subplot(1, 3, 1)
#             plt.imshow(image)
#             plt.title("1. Original Image", fontsize=14)
#             plt.axis("off")
#
#             # 子图2：提取出的原始空心轮廓
#             plt.subplot(1, 3, 2)
#             plt.imshow(image)
#             plt.imshow(original_edge_mask, cmap='Purples', alpha=0.7 * original_edge_mask)
#             plt.title("2. Original Hollow Contour", fontsize=14)
#             plt.axis("off")
#
#             # 子图3：算法填充后的实心掩膜
#             plt.subplot(1, 3, 3)
#             plt.imshow(image)
#             plt.imshow(final_filled_mask, cmap='Purples', alpha=0.7 * final_filled_mask)
#             plt.title("3. SciPy Filled Solid Mask !!", fontsize=14, color='red')
#             plt.axis("off")
#
#             plt.tight_layout()
#             save_path = os.path.join(TEST_SAVE_DIR, f"test_{img_name}.png")
#             plt.savefig(save_path, dpi=200)
#             plt.close()
#
#             print(f"✅ 生成测试图: {save_path}")
#
#             count += 1
#             if count >= 10:  # 挑 10 张包含紫色的图出来看看就行了
#                 break
#
#     print(f"\n🎉 测试完成！请前往 {TEST_SAVE_DIR} 文件夹查看填充效果。")
#
#
# if __name__ == "__main__":
#     test_mask_filling()
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

VAL_IMG_DIR = "./data/train/images"
VAL_MASK_DIR = "./data/train/masks"
TEST_SAVE_DIR = "./test_masks_output"


def test_mask_filling():
    os.makedirs(TEST_SAVE_DIR, exist_ok=True)
    test_images = [f for f in os.listdir(VAL_IMG_DIR) if f.endswith('.jpg')]

    count = 0
    for img_name in test_images:
        mask_name = img_name.replace('.jpg', '.png')
        img_path = os.path.join(VAL_IMG_DIR, img_name)
        mask_path = os.path.join(VAL_MASK_DIR, mask_name)

        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        mask_rgba = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask_rgba is not None and len(mask_rgba.shape) == 3 and mask_rgba.shape[2] == 4:
            bgr = mask_rgba[:, :, :3]
            alpha = mask_rgba[:, :, 3]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            R, G, B = rgb[:, :, 0].astype(np.int32), rgb[:, :, 1].astype(np.int32), rgb[:, :, 2].astype(np.int32)

            # 提取原始紫线
            is_purple = (alpha > 0) & (R > G + 20) & (B > G + 20)
            original_edge_mask = is_purple.astype(np.uint8)

            if original_edge_mask.sum() == 0: continue

            # ================= 终极魔法：反向泛洪填充 (FloodFill) =================
            # 1. 膨胀：把医生的线画粗一点，防止有细小的断点漏水
            kernel = np.ones((15, 15), np.uint8)
            dilated_edge = cv2.dilate(original_edge_mask, kernel, iterations=1)

            # 2. 强行画一个1像素的边框，把贴边的U型缺口彻底“封死”成O型
            sealed_edge = dilated_edge.copy()
            cv2.rectangle(sealed_edge, (0, 0), (w - 1, h - 1), 1, thickness=1)

            # 3. 泛洪填充：从四个角落开始“倒水”，填满所有外部背景
            flooded = sealed_edge.copy()
            ff_mask = np.zeros((h + 2, w + 2), np.uint8)

            # 四个角落的坐标（往内缩2个像素，避开刚才画的1像素边框）
            corners = [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)]
            for pt in corners:
                if flooded[pt[1], pt[0]] == 0:  # 如果这个角落是黑色的（背景）
                    cv2.floodFill(flooded, ff_mask, pt, 1)  # 倒水，把背景变成1

            # 4. 此时，外面全变成了1，线条也是1，【只有病灶内部是0】！我们把它反转过来
            inside_mask = (flooded == 0).astype(np.uint8)

            # 5. 把病灶内部，和原始的紫色线条合并
            filled_lesion = cv2.bitwise_or(inside_mask, dilated_edge)

            # 6. 腐蚀：瘦身还原到医生最初画的粗细
            final_filled_mask = cv2.erode(filled_lesion, kernel, iterations=1)
            final_filled_mask = final_filled_mask.astype(np.float32)
            # =======================================================================

            # 画图对比
            plt.figure(figsize=(18, 6))

            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("1. Original Image", fontsize=14)
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(image)
            plt.imshow(original_edge_mask, cmap='Purples', alpha=0.7 * original_edge_mask)
            plt.title("2. Original Boundary-Touching Contour", fontsize=14)
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(image)
            plt.imshow(final_filled_mask, cmap='Purples', alpha=0.6 * final_filled_mask)
            plt.title("3. FloodFill Perfect Solid Mask !!", fontsize=14, color='red')
            plt.axis("off")

            plt.tight_layout()
            save_path = os.path.join(TEST_SAVE_DIR, f"test_{img_name}.png")
            plt.savefig(save_path, dpi=200)
            plt.close()
            print(f"✅ 泛洪填充生成成功: {save_path}")

            count += 1
            if count >= 10: break


if __name__ == "__main__":
    test_mask_filling()
