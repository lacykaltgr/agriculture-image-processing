import numpy as np
import imageio
import os
import re
from torch.utils.data import Dataset
import torch
import glob
from utils import numericalSort, rgb_to_onehot, color_dict, crop


class XYDataset(Dataset):
    def __init__(self, root, crop_size=None, stride=None):
        self.x_data, self.y_data = load_dataset(root)
        self.crop_size = crop_size
        self.stride = stride

    def __len__(self):
        return len(self.x_data)


    def __getitem__(self, idx):
        
        print('itt')

        '''
        for i in range(len(self.y_data[idx][:, 0, 0])):
            print(i, self.y_data[idx][i,0,:])
        '''

        x = torch.transpose(torch.tensor(self.x_data[idx]).float(), 0, 2)
        y = torch.transpose(torch.tensor(self.y_data[idx][1568:2080,:,:3]).float(), 0, 2)

        y_onehot = rgb_to_onehot(y, color_dict)
        print(y_onehot.shape)
        print(x.shape)
        print(np.unique(y_onehot))

        return x, y_onehot




def load_dataset(root):
    # List of file names of actual Satellite images
    filelist_trainx = sorted(glob.glob(os.path.join(root, '*.JPG')), key=numericalSort)
    # List of file names of classified images
    filelist_trainy = sorted(glob.glob(os.path.join(root, '*_train.png')), key=numericalSort)

    x_data = []
    y_data = []

    for x_file in filelist_trainx:
        x_filename = os.path.basename(x_file)
        x_serial_number = re.search(r'\d+', x_filename).group()
        y_file = os.path.join(root, f'DSC{x_serial_number}_train.png')

        if os.path.exists(y_file):
            x_data.append(imageio.v2.imread(x_file))
            y_data.append(imageio.v2.imread(y_file))
        else:
            raise ValueError("NO Y FILE PAIR")

    return x_data, y_data


class XYDatasetTester:
    def __init__(self, root, crop_size=None, stride=None):
        self.dataset = XYDataset(root, crop_size, stride)
        self.root = root  # Store the root directory

    def test_data_format(self):
        for i in range(len(self.dataset)):
            x, y = self.dataset[i]

            # Check x and y dimensions
            print(f"x shape: {x.shape}, y shape: {y.shape}")

            # Check data type
            print(f"x data type: {x.dtype}, y data type: {y.dtype}")

            # Ensure x and y are in the correct format to be fed to the model
            assert x.shape[0] == 3, f"Error in x format for sample {i}"
            assert y.shape[0] == 1, f"Error in y format for sample {i}"

    def run_tests(self):
        print(f"Testing data format for root: {self.root}")
        self.test_data_format()
        print()


# Provide your data root
data_root = "/Users/czimbermark/DeepLearning_Projects_23/geoinformatics-image-proc/Project2-widthoftrees/data"
tester = XYDatasetTester(data_root)
tester.run_tests()

