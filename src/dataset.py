import numpy as np
import imageio
from torch.utils.data import Dataset
import torch
from src.utils import numericalSort, rgb_to_onehot, color_dict


class XYDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.transpose(torch.tensor(self.x_data[idx].astype('uint8'), dtype=torch.float), 0, 2)
        y = torch.tensor(self.y_data[idx].astype('uint8'), dtype=torch.float)
        y_onehot = rgb_to_onehot(y, color_dict=color_dict)
        return x, np.transpose(y_onehot, (2, 1, 0))


def load_images(fnames):
    d_list = []
    for fname in fnames:
        image_raw = imageio.read(fname)
        image = np.array(image_raw.get_data(0))
        d_list.append(image)
    return d_list


def load_dataset(root):
    import glob

    # List of file names of actual Satellite images for traininig
    filelist_trainx = sorted(glob.glob(root+'The-Eye-in-the-Sky-dataset/sat/*.tif'), key=numericalSort)
    # List of file names of classified images for traininig
    filelist_trainy = sorted(glob.glob(root+'The-Eye-in-the-Sky-dataset/gt/*.tif'), key=numericalSort)
    # List of file names of actual Satellite images for testing
    filelist_testx = sorted(glob.glob(root+'The-Eye-in-the-Sky-test-data/sat_test/*.tif'), key=numericalSort)

    # Making array of all the training sat images as it is without any cropping
    x = load_images(filelist_trainx)
    y = load_images(filelist_trainy)
    x_test = load_images(filelist_testx)

    x_train = x[:-1]
    y_train = y[:-1]
    x_val = [x[-1]]
    y_val = [y[-1]]

    return x_train, y_train, x_val, y_val, x_test
