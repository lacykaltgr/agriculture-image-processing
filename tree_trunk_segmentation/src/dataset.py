import cv2
import os
import re
from torch.utils.data import Dataset
import torch
import glob
from tree_trunk_segmentation.src.utils import rgb_to_binary, color_dict, numericalSort


class XYDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.transpose(torch.tensor(self.x_data[idx]).float(), 0, 2)
        #y_binary = rgb_to_binary(self.y_data[idx], color_dict)
        y = torch.transpose(torch.tensor(self.y_data[idx]).float(), 0, 2)
        return x, y


def load_dataset(root, target_size_x=None, target_size_y=None, val_split_ratio=0.9):

    binary = lambda y: rgb_to_binary(y, color_dict)

    x_filelist_3_2 = sorted(glob.glob(os.path.join(root, '3_2_images', '*.JPG')), key=numericalSort)
    x_filelist_4_3 = sorted(glob.glob(os.path.join(root, '4_3_images', '*.JPG')), key=numericalSort)

    x_data, y_data = [], []

    for x_file in x_filelist_3_2:
        x_filename = os.path.basename(x_file)
        x_serial_number = re.search(r'\d+', x_filename).group()
        y_file = os.path.join(root, '3_2_images', f'DSC{x_serial_number}_train.png')

        if not os.path.exists(y_file):
            raise ValueError("Target file does not exist")

        x = cv2.imread(x_file)
        y = cv2.imread(y_file)

        w_x, h_x, c_x = x.shape
        new_w_x = int(h_x / 3 * 4)
        x1 = x[:, :new_w_x]
        x2 = x[:, w_x-new_w_x:]
        y = y[:, :, :3] # igy pont 500, alapból 512
        y1 = y[:, :new_w_x]
        y2 = y[:, w_x-new_w_x:]

        x1_resized = cv2.resize(x1, target_size_x)
        x2_resized = cv2.resize(x2, target_size_x)
        y1_resized = cv2.resize(y1, target_size_y)
        y2_resized = cv2.resize(y2, target_size_y)

        x_data.append(x1_resized)
        y_data.append(binary(y1_resized))
        x_data.append(x2_resized)
        y_data.append(binary(y2_resized))

    for x_file in x_filelist_4_3:
        x_filename = os.path.basename(x_file)
        x_serial_number = re.search(r'\d+', x_filename).group()
        y_file = os.path.join(root, '4_3_images', f'P{x_serial_number}_train.png')
        if not os.path.exists(y_file):
            raise ValueError("Target file does not exist")

        x = cv2.imread(x_file)
        y = cv2.imread(y_file)

        if x.shape != target_size_x:
            x = cv2.resize(x, target_size_x)
        if y.shape != target_size_y:
            y = cv2.resize(y, target_size_y)

        x_data.append(x)
        y_data.append(binary(y))

    split = int(len(x_data) * val_split_ratio)
    x_train = x_data[:split]
    y_train = y_data[:split]
    x_val = x_data[split:]
    y_val = y_data[split:]

    return x_train, y_train, x_val, y_val
