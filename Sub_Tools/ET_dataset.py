import h5py
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def read_pair_txt(file_path):
    pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            img_path, lbl_path = line.strip().split()
            pairs.append((img_path, lbl_path))
    return pairs


def read_test_lst(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]


def convert_to_hdf5(dataset_dir, train_lst_path, test_lst_path, hdf5_output_path):
    # 记录已处理的图像路径，避免重复存储
    processed_images = set()

    # 创建HDF5文件，启用压缩
    with h5py.File(hdf5_output_path, 'w') as f:
        # 处理训练集
        train_pairs = read_pair_txt(train_lst_path)
        train_group = f.create_group('train')
        for img_path, lbl_path in tqdm(train_pairs, desc="Processing train"):
            try:
                img_full_path = os.path.join(dataset_dir, img_path)
                lbl_full_path = os.path.join(dataset_dir, lbl_path)
                img = Image.open(img_full_path).convert('RGB')
                lbl = Image.open(lbl_full_path).convert('L')

                img_array = np.array(img)
                lbl_array = np.array(lbl)

                # 图像键：基于图像路径
                img_key = f"img_{img_path.replace('/', '_')}"
                # 标注键：基于标注路径
                lbl_key = f"lbl_{lbl_path.replace('/', '_')}"

                # 仅在未处理过该图像时存储，使用GZIP压缩
                if img_path not in processed_images:
                    train_group.create_dataset(img_key, data=img_array, compression='gzip', compression_opts=4)
                    processed_images.add(img_path)

                # 存储标注，使用GZIP压缩
                train_group.create_dataset(lbl_key, data=lbl_array, compression='gzip', compression_opts=4)
            except Exception as e:
                print(f"Error processing {img_path} or {lbl_path}: {e}")
                continue

        # 处理测试集
        test_images = read_test_lst(test_lst_path)
        test_group = f.create_group('test')
        for img_path in tqdm(test_images, desc="Processing test"):
            try:
                img_full_path = os.path.join(dataset_dir, img_path)
                img = Image.open(img_full_path).convert('RGB')
                img_array = np.array(img)
                img_key = f"img_{img_path.replace('/', '_')}"
                test_group.create_dataset(img_key, data=img_array, compression='gzip', compression_opts=4)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue


if __name__ == "__main__":
    dataset_dir = r"D:\rawcode\data\BSDS500_flipped_rotated_pad"
    train_lst_path = r"D:\rawcode\data\BSDS500_flipped_rotated_pad\image-train.lst"
    test_lst_path = r"D:\rawcode\data\BSDS500_flipped_rotated_pad\image-test.lst"
    hdf5_output_path = r"D:\rawcode\data\BSDS500_flipped_rotated_pad\dataset.h5"

    # 删除旧文件（若存在）
    if os.path.exists(hdf5_output_path):
        os.remove(hdf5_output_path)

    convert_to_hdf5(dataset_dir, train_lst_path, test_lst_path, hdf5_output_path)