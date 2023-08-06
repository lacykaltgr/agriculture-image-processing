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


def onehot_to_rgb(onehot):
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


def crop(img, crop_size=256, stride=None):
    if stride is None:
        stride = crop_size/2
    croped_images = []
    h, w, c = img.shape

    n_h = math.ceil((h-crop_size)/stride+1)
    n_w = math.ceil((w-crop_size)/stride+1)

    for i in range(n_h):
        h1 = min(int(i*stride), h-crop_size)
        h2 = min(int(i*stride+crop_size), h)
        for j in range(n_w):
            w1 = min(int(j*stride), w-crop_size)
            w2 = min(int(j*stride+crop_size), w)
            crop_x = img[h1:h2, w1:w2, :]
            croped_images.append(crop_x)
    return croped_images


def conf_matrix(val_dataset, predictions, plot=True):
    from sklearn.metrics import confusion_matrix
    num_classes = 9
    total_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int)

    i = 0
    for batch in predictions:
        for image in batch:
            pred = image.flatten()
            target = np.argmax(val_dataset[i][1], axis=0).flatten()

            conf_matrix_batch = confusion_matrix(pred, target, labels=range(num_classes))
            total_conf_matrix += conf_matrix_batch
            i += 1

    if plot:
        class_labels = list(range(num_classes))
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        sns.heatmap(total_conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    return total_conf_matrix