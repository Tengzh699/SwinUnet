import numpy as np
import scipy.ndimage as ndimage


def clean_prediction(pred_mask, min_size=500):
    """
    医学图像预测结果的后处理 (Post-processing)
    1. 填补病灶内部的微小空洞 (Fill holes)
    2. 消除散落在背景上的孤立噪点 (Remove small objects)

    参数:
    pred_mask: 模型的二值化输出，numpy array 格式 (H, W)
    min_size: 允许的最小连通域面积，小于这个像素数的紫斑会被抹除
    """
    # 确保输入是 boolean 格式
    pred_mask = pred_mask.astype(bool)

    # 第一步：填补病灶内部可能存在的小空洞（模拟实心病灶）
    filled_mask = ndimage.binary_fill_holes(pred_mask)

    # 第二步：连通域分析，消除孤立的细小噪点
    # label 函数会给每个独立的斑块打上不同的标签 (1, 2, 3...)
    labeled_mask, num_features = ndimage.label(filled_mask)

    cleaned_mask = np.zeros_like(filled_mask, dtype=np.uint8)

    # 遍历每一个被标记的斑块
    for i in range(1, num_features + 1):
        # 计算当前斑块的像素总面积
        area = (labeled_mask == i).sum()
        # 如果面积大于设定的阈值，我们才认为它是真正的病灶，将其保留
        if area > min_size:
            cleaned_mask[labeled_mask == i] = 1

    # 转换回 float32 类型，方便后续计算或画图
    return cleaned_mask.astype(np.float32)


# 测试一下 (可删)
if __name__ == "__main__":
    print("Utils module loaded successfully.")