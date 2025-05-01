import h5py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def read_pair_txt(file_path):
    pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            img_path, lbl_path = line.strip().split()
            pairs.append((img_path, lbl_path))
    return pairs

class BsdsDataset(Dataset):
    def __init__(self, dataset_path="", hdf5_path="", flag="train", sub_sample=-1):
        self.dataset_dir = dataset_path
        self.hdf5_path = hdf5_path
        self.flag = flag
        self.hdf5_file = h5py.File(hdf5_path, 'r')

        if self.flag == "train":
            pairs = read_pair_txt(os.path.join(self.dataset_dir, "image-train.lst"))
            if sub_sample > 0:
                selected_indices = np.random.choice(len(pairs), sub_sample, replace=False)
                self.img_list = [pairs[i][0] for i in selected_indices]
                self.lbl_list = [pairs[i][1] for i in selected_indices]
            else:
                self.img_list = [img_name[0] for img_name in pairs]
                self.lbl_list = [img_name[1] for img_name in pairs]
        elif self.flag == "test":
            assert sub_sample <= 0
            with open(os.path.join(self.dataset_dir, "image-test.lst"), 'r') as f:
                self.img_list = [line.strip() for line in f]
            self.lbl_list = self.img_list  # 测试集无标注，使用图像路径占位

    def __len__(self):
        return len(self.img_list)

    def trans_in_train(self, sample):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        sample['images'] = trans(sample['images'])
        sample['labels'] = torch.from_numpy(np.array(sample['labels']).astype(np.float32) / 255.0).unsqueeze(0)  # 单通道，0-1
        return sample

    def trans_in_test(self, sample):
        trans = transforms.Compose([
            transforms.RandomCrop((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        sample['images'] = trans(sample['images'])
        sample['labels'] = trans(sample['labels'])  # 测试集图像作为占位标注
        return sample

    def __getitem__(self, index):
        if self.flag == "train":
            img_key = f"img_{self.img_list[index].replace('/', '_')}"
            lbl_key = f"lbl_{self.lbl_list[index].replace('/', '_')}"
            img_array = self.hdf5_file['train'][img_key][:]
            lbl_array = self.hdf5_file['train'][lbl_key][:]
            image = Image.fromarray(img_array)
            label = Image.fromarray(lbl_array)
        else:
            img_key = f"img_{self.img_list[index].replace('/', '_')}"
            img_array = self.hdf5_file['test'][img_key][:]
            image = Image.fromarray(img_array)
            label = image  # 测试集无标注，使用图像占位

        sample = {"images": image, "labels": label}

        if self.flag == "train":
            sample = self.trans_in_train(sample)
        else:
            sample = self.trans_in_test(sample)

        return sample