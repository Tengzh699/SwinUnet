import os
import shutil
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = './Annocerv'
BASE_DIR = './data'


def setup_dirs():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(BASE_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, split, 'masks'), exist_ok=True)


def main():
    setup_dirs()
    valid_pairs = []

    # 遍历所有的 Case 文件夹
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        for file in files:
            # 只寻找醋酸图 (包含 'Aceto') 且是 jpg 格式
            if 'Aceto' in file and file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                mask_file = file.replace('.jpg', '.png')
                mask_path = os.path.join(root, mask_file)

                # 如果同名掩码文件存在，则视为有效数据对
                if os.path.exists(mask_path):
                    valid_pairs.append((img_path, mask_path, file, mask_file))

    print(f"共找到 {len(valid_pairs)} 对包含标注的醋酸图像数据。")

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_pairs, val_pairs = train_test_split(valid_pairs, test_size=0.2, random_state=42)

    def copy_files(pairs, split_name):
        for img_path, mask_path, img_name, mask_name in pairs:
            dst_img = os.path.join(BASE_DIR, split_name, 'images', img_name)
            dst_mask = os.path.join(BASE_DIR, split_name, 'masks', mask_name)
            shutil.copy(img_path, dst_img)
            shutil.copy(mask_path, dst_mask)

    print("正在生成训练集...")
    copy_files(train_pairs, 'train')
    print("正在生成验证集...")
    copy_files(val_pairs, 'val')
    print("✅ 数据集处理完成！请检查 ./data 目录。")


if __name__ == '__main__':
    main()