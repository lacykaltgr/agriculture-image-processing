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


def load_dataset(root):
    filelist_trainx = sorted(glob.glob(os.path.join(root, '*.JPG')), key=numericalSort)

    x_data, y_data = [], []

    for x_file in filelist_trainx:
        x_filename = os.path.basename(x_file)
        x_serial_number = re.search(r'\d+', x_filename).group()
        y_file = os.path.join(root, f'DSC{x_serial_number}_train.png')
        #[1568:2080,:,:3]


        if os.path.exists(y_file):
            x_data.append(cv2.imread(x_file))
            y_data.append(cv2.imread(y_file))
        else:
            raise ValueError("Target file does not exist")
        #resized_image = cv2.resize(self.x_data[idx], self.target_size)

    return x_data, y_data
