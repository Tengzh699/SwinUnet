import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.optim as optim
import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import clean_prediction

from monai.losses import DiceFocalLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from dataset import AnnoCervDataset, get_transforms
from model import CervicalSwinUnet

# ================= 配置参数 =================
IMG_DIR = './data/train/images'
MASK_DIR = './data/train/masks'
VAL_IMG_DIR = './data/val/images'
VAL_MASK_DIR = './data/val/masks'

BATCH_SIZE = 8
EPOCHS =100
LR = 3e-4  # 【调优】：稍微调大初始学习率，帮模型跳出局部最优解（全黑泥潭）
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================

def create_experiment_dir(base_dir="results"):
    """
    自动创建递增的实验文件夹，例如 results/exp1, results/exp2 ...
    返回新建文件夹的路径
    """
    os.makedirs(base_dir, exist_ok=True)

    # 查找当前已有的 exp_X 文件夹
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith("exp")]

    max_num = 0
    for d in existing_dirs:
        try:
            num = int(d.replace("exp", ""))
            if num > max_num:
                max_num = num
        except ValueError:
            pass

    # 创建下一个编号的文件夹
    new_exp_name = f"exp{max_num + 1}"
    new_exp_path = os.path.join(base_dir, new_exp_name)
    os.makedirs(new_exp_path, exist_ok=True)

    return new_exp_path


def log_message(log_file, message):
    """
    将信息同时打印到控制台并写入日志文件
    """
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Training", leave=False)
    epoch_loss = 0

    for data, targets in loop:
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)


def calculate_metrics(preds, target, smooth=1e-5):
    """
    计算三个核心指标：Dice, IoU, PA
    preds: 模型预测的二值化掩膜 (N, 1, H, W)
    target: 真实的二值化金标准 (N, 1, H, W)
    """
    # 展平矩阵，方便计算交集和并集
    preds = preds.contiguous().view(-1)
    target = target.contiguous().view(-1)

    # 计算交集
    intersection = (preds * target).sum()

    # 1. 计算 Dice
    dice = (2. * intersection + smooth) / (preds.sum() + target.sum() + smooth)

    # 2. 计算 IoU (Intersection over Union) = 交集 / (预测集 + 真实集 - 交集)
    union = preds.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    # 3. 计算 PA (Pixel Accuracy, 像素准确率) = 预测对的像素总数 / 总像素数
    # 预测对的像素 = (预测为1且真实为1) + (预测为0且真实为0)
    correct = (preds == target).sum()
    pa = correct / float(preds.numel())

    return dice.item(), iou.item(), pa.item()


def check_accuracy(loader, model):
    model.eval()
    dice_score = 0
    total_dice = 0
    total_iou = 0
    total_pa = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            with torch.amp.autocast('cuda'):
                preds = torch.sigmoid(model(x))

            # 1. 原始的 0.5 阈值二值化
            preds = (preds > 0.5).float()

            # 2. 核心后处理：由于 batch_size 的存在，我们需要逐张图清理
            preds_np = preds.cpu().numpy()  # 转到 CPU
            cleaned_preds_np = np.zeros_like(preds_np)

            for i in range(preds_np.shape[0]):  # 遍历 batch 中的每一张图
                # 提取出单张图的二维矩阵 (H, W)，送入清洗函数
                single_mask = preds_np[i, 0, :, :]
                cleaned_single = clean_prediction(single_mask, min_size=500)
                cleaned_preds_np[i, 0, :, :] = cleaned_single

            # 将清理干净的结果转回 GPU Tensor
            cleaned_preds = torch.from_numpy(cleaned_preds_np).to(DEVICE)

            # 3. 使用清理后的结果计算 Dice
            # intersection = (cleaned_preds * y).sum()
            # dice = (2. * intersection + 1e-5) / (cleaned_preds.sum() + y.sum() + 1e-5)
            # dice_score += dice
            dice, iou, pa = calculate_metrics(cleaned_preds, y)

            total_dice += dice
            total_iou += iou
            total_pa += pa

    # avg_dice = dice_score / len(loader)
    # 计算平均值
    num_batches = len(loader)
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    avg_pa = total_pa / num_batches
    # return avg_dice.item()
    return avg_dice, avg_iou, avg_pa


def main():
    # os.makedirs("./weights", exist_ok=True)
    # 解决 Windows 潜在的 DLL 冲突
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # 1. 自动创建实验专属文件夹 (例如: results/exp1)
    exp_dir = create_experiment_dir("results")
    log_file = os.path.join(exp_dir, "train_log.txt")
    weight_save_path = os.path.join(exp_dir, "best_model.pth")

    # 记录实验初始信息
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message(log_file, f"===========================================")
    log_message(log_file, f"[*] 实验开始时间: {start_time}")
    log_message(log_file, f"[*] 当前保存目录: {exp_dir}")
    log_message(log_file, f"[*] 使用计算设备: {DEVICE.upper()}")
    log_message(log_file, f"[*] Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}")
    log_message(log_file, f"===========================================\n")

    if DEVICE == "cpu":
        log_message(log_file, "[!] 警告: 未检测到GPU！模型将在CPU上运行，训练速度会非常慢。")

    print(f"[*] 当前正在使用的计算设备：{DEVICE.upper()}")

    train_dataset = AnnoCervDataset(IMG_DIR, MASK_DIR, transform=get_transforms(True, IMG_SIZE))
    val_dataset = AnnoCervDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=get_transforms(False, IMG_SIZE))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CervicalSwinUnet(img_size=IMG_SIZE, in_channels=3, out_channels=1).to(DEVICE)

    # 【核心调整】：使用 Dice + Focal Loss，gamma=2.0 意味着对难预测的病灶施加成倍惩罚
    loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True, batch=True, gamma=2.0)

    # 增加权重衰减，抑制过拟合
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    # 【核心调整】：引入余弦退火调度器，让学习率像波浪一样变化，跳出"预测全黑"的舒适区
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    scaler = torch.amp.GradScaler('cuda') if DEVICE == "cuda" else None

    # best_dice = 0
    # for epoch in range(EPOCHS):
    #     print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    #
    #     train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
    #     dice = check_accuracy(val_loader, model)
    #
    #     # 学习率更新
    #     scheduler.step()
    #     current_lr = optimizer.param_groups[0]['lr']
    #
    #     print(f"Train Loss: {train_loss:.4f} | Val Dice Score: {dice:.4f} | LR: {current_lr:.6f}")
    #
    #     # 只要有一丁点提升就保存
    #     if dice > best_dice:
    #         best_dice = dice
    #         torch.save(model.state_dict(), "./weights/best_swin_unet.pth")
    #         print("=> 表现提升，已保存最佳模型！")
    best_dice = 0
    best_iou = 0
    best_pa = 0
    for epoch in range(EPOCHS):
        epoch_str = f"Epoch [{epoch + 1}/{EPOCHS}]"
        print(f"\n{epoch_str}")

        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        # dice = check_accuracy(val_loader, model)
        dice, iou, pa = check_accuracy(val_loader, model)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 将日志信息写入文件
        msg = f"{epoch_str} | Train Loss: {train_loss:.4f} | Val Dice Score: {dice:.4f} | LR: {current_lr:.6f}"
        log_message(log_file, msg)

        # 只要有一丁点提升就保存，并记录在日志里
        if dice > best_dice:
            best_dice = dice
            best_iou = iou
            best_pa = pa
            torch.save(model.state_dict(), weight_save_path)
            log_message(log_file, f"=> 表现提升 (Best Dice: {best_dice:.4f}, mIoU: {best_iou:.4f}, mPA: {best_pa:.4f})，模型已保存！\n")

    # 训练结束
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message(log_file, f"===========================================")
    log_message(log_file, f"[*] 实验结束时间: {end_time}")
    # log_message(log_file, f"[*] 最终最优 Dice Score: {best_dice:.4f}")
    log_message(log_file, f"[*] 最终最优验证指标:")
    log_message(log_file, f"    - Best Dice : {best_dice:.4f}")
    log_message(log_file, f"    - Best mIoU : {best_iou:.4f}")
    log_message(log_file, f"    - Best mPA  : {best_pa:.4f}")
    log_message(log_file, f"===========================================")


if __name__ == "__main__":
    main()
