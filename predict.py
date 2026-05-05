import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_transforms
from model import CervicalSwinUnet
from utils import clean_prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_PATH = "./results/exp4/best_model.pth"
IMG_SIZE = 224


def load_model():
    model = CervicalSwinUnet(img_size=IMG_SIZE, in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.eval()
    return model


def predict_and_plot(model, img_path, mask_path, save_name):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    mask_rgba = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_rgba is not None and len(mask_rgba.shape) == 3 and mask_rgba.shape[2] == 4:
        bgr = mask_rgba[:, :, :3]
        alpha = mask_rgba[:, :, 3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        R, G, B = rgb[:, :, 0].astype(np.int32), rgb[:, :, 1].astype(np.int32), rgb[:, :, 2].astype(np.int32)

        is_purple = (alpha > 0) & (R > G + 20) & (B > G + 20) & (B > R * 0.6)
        original_edge_mask = is_purple.astype(np.uint8)

        if original_edge_mask.sum() > 0:
            kernel = np.ones((15, 15), np.uint8)
            dilated_edge = cv2.dilate(original_edge_mask, kernel, iterations=1)
            sealed_edge = dilated_edge.copy()
            cv2.rectangle(sealed_edge, (0, 0), (w - 1, h - 1), 1, thickness=1)
            flooded = sealed_edge.copy()
            ff_mask = np.zeros((h + 2, w + 2), np.uint8)
            corners = [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)]
            for pt in corners:
                if flooded[pt[1], pt[0]] == 0:
                    cv2.floodFill(flooded, ff_mask, pt, 1)
            inside_mask = (flooded == 0).astype(np.uint8)
            filled_lesion = cv2.bitwise_or(inside_mask, dilated_edge)
            final_filled_mask = cv2.erode(filled_lesion, kernel, iterations=1)
            true_mask = final_filled_mask.astype(np.float32)
        else:
            true_mask = np.zeros((h, w), dtype=np.float32)

        if true_mask.shape != (h, w):
            true_mask = cv2.resize(true_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        true_mask = np.zeros((h, w), dtype=np.float32)

    # transform = get_transforms(is_train=False, img_size=IMG_SIZE)
    # augmented = transform(image=image)
    # tensor_img = augmented['image'].unsqueeze(0).to(DEVICE)
    #
    # with torch.no_grad():
    #     if DEVICE == "cuda":
    #         with torch.cuda.amp.autocast():
    #             pred = model(tensor_img)
    #     else:
    #         pred = model(tensor_img)
    #     pred_sigmoid = torch.sigmoid(pred)
    #     # 0.5 阈值，现在GT是实心的，模型会更有自信！
    #     pred_mask = (pred_sigmoid > 0.5).float().cpu().squeeze().numpy()
     # 3. 模型预测
    transform = get_transforms(is_train=False, img_size=IMG_SIZE)
    augmented = transform(image=image)
    tensor_img = augmented['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                pred = model(tensor_img)
        else:
            pred = model(tensor_img)

        pred_sigmoid = torch.sigmoid(pred)

        # 原始二值化输出
        raw_pred_mask = (pred_sigmoid > 0.5).float().cpu().squeeze().numpy()

        # 【核心加入】：对预测结果进行后处理清洗 (填补空洞，消除面积小于500的噪点)
        pred_mask = clean_prediction(raw_pred_mask, min_size=500)

    plt.figure(figsize=(15, 6))
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    true_mask_resized = cv2.resize(true_mask, (IMG_SIZE, IMG_SIZE))

    plt.subplot(1, 3, 1)
    plt.imshow(image_resized)
    plt.title("Original Image",fontsize=24)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(image_resized)
    plt.imshow(true_mask_resized, cmap='Purples', alpha=0.6 * true_mask_resized)
    plt.title("Ground Truth",fontsize=24)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image_resized)
    plt.imshow(pred_mask, cmap='Purples', alpha=0.6 * pred_mask)
    plt.title("Swin-Unet Prediction",fontsize=24)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"./results/exp4/{save_name}.png", dpi=300)
    plt.close()
    print(f"已保存对比图: ./results/exp4/{save_name}.png")


if __name__ == "__main__":
    # os.makedirs("./results", exist_ok=True)
    model = load_model()
    val_img_dir = "./data/val/images"
    val_mask_dir = "./data/val/masks"

    test_images = os.listdir(val_img_dir)
    count = 0
    for img_name in test_images:
        mask_name = img_name.replace('.jpg', '.png')
        img_path = os.path.join(val_img_dir, img_name)
        mask_path = os.path.join(val_mask_dir, mask_name)

        # 只画包含紫色病灶的图片
        mask_rgba = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_rgba is not None and len(mask_rgba.shape) == 3 and mask_rgba.shape[2] == 4:
            alpha = mask_rgba[:, :, 3]
            if alpha.sum() > 0:
                predict_and_plot(model, img_path, mask_path, save_name=img_name.split('.')[0])
                count += 1
                if count >= 10:
                    break