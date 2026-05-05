# SwinUNet_AnnoCerv

基于 **Swin-UNet** 的宫颈图像（AnnoCerv）分割项目，包含数据准备、训练与预测脚本，并提供部分训练结果与权重文件。

## 项目结构

- `model.py`：模型结构（Swin-UNet）
- `dataset.py`：数据集读取与预处理
- `loss.py`：损失函数
- `train.py`：训练入口
- `predict.py`：预测/推理入口
- `utils.py`：通用工具函数
- `prepare_annocerv.py`：将原始数据整理为训练/验证所需目录结构
- `data/`：训练/验证数据（images/masks）
- `Annocerv/`：原始数据（示例/备份）
- `weights/`：模型权重（例如 `weights/best_swin_unet.pth`）
- `results/`、`result/`：实验输出（可视化、日志、预测结果等）

## 环境依赖

建议使用 Python 3.8+（或 3.9/3.10），常见依赖包括：

- `torch`
- `torchvision`
- `numpy`
- `opencv-python`
- `Pillow`
- `tqdm`

你可以用下面的方式自行整理成 `requirements.txt`（如项目里暂时没有）：
```bash
pip freeze > requirements.txt
```

## 数据准备

项目默认使用类似如下结构（以 `data/` 为例）：

```
data/
  train/
    images/
    masks/
  val/
    images/
    masks/
```

如果你的原始数据在 `Annocerv/` 下，可以尝试运行：
```bash
python prepare_annocerv.py
```

说明：
- `images/` 为原图（如 `.jpg`）
- `masks/` 为分割标注（如 `.png`，通常是二值或多类 mask）

## 训练

直接运行：
```bash
python train.py
```

训练产物通常会输出到 `results/exp*/`（以你当前项目目录为准），可能包含：
- `best_model.pth`：最佳权重
- `train_log.txt`：训练日志
- `training_curve*.png`：曲线图

## 预测 / 推理

使用训练好的权重进行推理：
```bash
python predict.py
```

你可以在 `predict.py` 里配置：
- 权重路径（如 `weights/best_swin_unet.pth` 或 `results/exp*/best_model.pth`）
- 输入图片/数据集路径
- 输出目录（如 `result/` 或某个 `results/exp*/`）

## 已包含的权重与结果（注意体积）

仓库中包含多个 `.pth`（例如 `best_model.pth`、`best_swin_unet*.pth`），以及大量图像数据与结果图，体积可能较大。

建议（可选）：
- 训练数据与大文件用 Git LFS 管理，或从仓库剥离改为网盘/Release 提供下载
- 将中间结果（如 `results/`、`result/`）按需保留

## 常见问题

- 如果训练/预测报路径错误：先检查 `data/` 的目录结构是否与脚本一致
- 如果 mask 读取不对：检查 `dataset.py` 中对 mask 的加载方式（二值/多类、是否需要归一化、通道处理等）

## 参考与致谢

- Swin Transformer / Swin-UNet 相关工作与实现
- 医学图像分割常用训练范式（Dice/CE 等）

