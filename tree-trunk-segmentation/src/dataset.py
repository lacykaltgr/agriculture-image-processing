import cv2
import os
import re
from torch.utils.data import Dataset
import torch
import glob
from utils import rgb_to_binary, color_dict, numericalSort


class XYDataset(Dataset):
    def __init__(self, root, target_size=None):
        self.x_data, self.y_data = load_dataset(root)
        self.target_size = target_size if target_size is not None else (256, 256)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.transpose(torch.tensor(self.x_data[idx].data).float(), 0, 2)
        y_binary = rgb_to_binary(self.y_data[idx], color_dict)
        y = torch.transpose(torch.tensor(y_binary).float(), 0, 2)
        return x, y


def load_dataset(root, target_size_x=None, target_size_y=None):
    x_filelist_3_2 = sorted(glob.glob(os.path.join(root, '3_2_images', 'train',  '*.JPG')), key=numericalSort)
    x_filelist_4_3 = sorted(glob.glob(os.path.join(root, '4_3_images', 'train',  '*.jpg')), key=numericalSort)

    x_data, y_data = [], []

    for x_file in x_filelist_3_2:
        x_filename = os.path.basename(x_file)
        x_serial_number = re.search(r'\d+', x_filename).group()
        y_file = os.path.join(root, f'DSC{x_serial_number}_train.png')

        if not os.path.exists(y_file):
            raise ValueError("Target file does not exist")

        x = cv2.imread(x_file)
        y = cv2.imread(y_file)

        w_x, h_x, c_x = x.shape
        new_w_x = h_x / 3 * 4
        x1 = x[:, :new_w_x]
        x2 = x[:, w_x-new_w_x:]
        y = y[1574:2074, :, :3] # igy pont 500, alapból 512
        y1 = y[:, :new_w_x]
        y2 = y[:, w_x-new_w_x]

        x1_resized = cv2.resize(x1, target_size_x)
        x2_resized = cv2.resize(x2, target_size_x)
        y1_resized = cv2.resize(y1, target_size_y)
        y2_resized = cv2.resize(y2, target_size_y)

        x_data.append(x1_resized)
        y_data.append(y1_resized)
        x_data.append(x2_resized)
        y_data.append(y2_resized)

    for x_file in x_filelist_4_3:
        x_filename = os.path.basename(x_file)
        x_serial_number = re.search(r'\d+', x_filename).group()
        y_file = os.path.join(root, f'DSC{x_serial_number}_train.png')
        if not os.path.exists(y_file):
            raise ValueError("Target file does not exist")

        x = cv2.imread(x_file)
        y = cv2.imread(y_file)

        if x.shape != target_size_x:
            x = cv2.resize(x, target_size_x)
        if y.shape != target_size_y:
            y = cv2.resize(y, target_size_y)

        x_data.append(x)
        y_data.append(y)

    return x_data, y_data
