import numpy as np
import torch
import math


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
    arr = np.zeros(shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:, :, i] = torch.all(rgb_arr.reshape((-1, 3)) == color_dict[i], axis=1).reshape(shape[:2])
    return arr


def onehot_to_rgb(onehot, color_dict):
    output = np.zeros(onehot.shape+(3,))
    for k in color_dict.keys():
        output[onehot == k] = color_dict[k].numpy()
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


def add_pixals(img, h, w, c, n_h, n_w, crop_size, stride):

    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra

    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra

    img_add = np.zeros(((h+h_toadd), (w+w_toadd), c))

    img_add[:h, :w,:] = img
    img_add[h:, :w,:] = img[:h_toadd,:, :]
    img_add[:h,w:,:] = img[:,:w_toadd,:]
    img_add[h:,w:,:] = img[h-h_toadd:h,w-w_toadd:w,:]

    return img_add


def padding(img, w, h, c, crop_size, stride, n_h, n_w):

    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra

    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra

    img_pad = np.zeros(((h+h_toadd), (w+w_toadd), c))
    #img_pad[:h, :w,:] = img
    #img_pad = img_pad+img
    img_pad = np.pad(img, [(0, h_toadd), (0, w_toadd), (0,0)], mode='constant')

    return img_pad


def crop(a, crop_size=128):
    stride = 32

    croped_images = []
    h, w, c = a.shape

    n_h = int(int(h/stride))
    n_w = int(int(w/stride))

    # Padding using the padding function we wrote
    a = padding(a, w, h, c, crop_size, stride, n_h, n_w)

    # Resizing as required
    ##a = resize(a, stride, n_h, n_w)

    # Adding pixals as required
    #a = add_pixals(a, h, w, c, n_h, n_w, crop_size, stride)

    # Slicing the image into 128*128 crops with a stride of 64
    for i in range(n_h-1):
        for j in range(n_w-1):
            crop_x = a[(i*stride):((i*stride)+crop_size), (j*stride):((j*stride)+crop_size), :]
            croped_images.append(crop_x)
    return croped_images