import numpy as np
import torch


color_dict = {0: torch.tensor([0, 0, 0]),
              1: torch.tensor([0, 125, 0]),
              2: torch.tensor([150, 80, 0]),
              3: torch.tensor([255, 255, 0]),
              4: torch.tensor([100, 100, 100]),
              5: torch.tensor([0, 255, 0]),
              6: torch.tensor([0, 0, 150]),
              7: torch.tensor([150, 150, 255]),
              8: torch.tensor([255, 255, 255])}


def pad_to_square(image, fill_color=0):
    height, width = image.shape[:2]
    max_dim = max(width, height)
    # Calculate padding size for both sides
    left_padding = (max_dim - width) // 2
    top_padding = (max_dim - height) // 2
    # Create a new square canvas with the desired value for padding
    padded_image = np.full((max_dim, max_dim) + image.shape[2:], fill_color, dtype=image.dtype)
    # Paste the original image onto the new canvas
    padded_image[top_padding:top_padding+height, left_padding:left_padding+width, ...] = image
    return padded_image


def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    print(shape)
    arr = np.zeros(shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr


def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)


def to_class_no(y_hot_list):
    y_class_list = []
    n = len(y_hot_list)
    for i in range(n):
        out = np.argmax(y_hot_list[i])
        y_class_list.append(out)
    return y_class_list


def numericalSort(value):
    import re
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts