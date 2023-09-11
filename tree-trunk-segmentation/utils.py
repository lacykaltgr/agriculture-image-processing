import numpy as np
import torch
import math
import matplotlib.pyplot as plt

color_dict = {
    # 0: torch.tensor([255, 255, 255]),  # White (Background)
    0: torch.tensor([255, 0, 0]),      # Red (Not Trees)
    1: torch.tensor([0, 255, 0])       # Green (Tree Trunks)
}


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

# transform rgb pixels to binary based on 2 item colour dict
# q: why error cls is not defined?

def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = (1,)+rgb_arr.shape[1:]
    transpose = np.transpose(rgb_arr, (1,2,0))
    color_match = torch.all(transpose == color_dict[1], dim=2)
    arr = np.zeros(shape, dtype=np.uint8)
    arr[0, :, :] = color_match.int()
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

def predict_windows(model, windows):
    predicted_windows = []

    for window in windows:
        input_tensor = torch.tensor(window.data, dtype=torch.float32)
        input_tensor = input_tensor.permute(2, 0, 1)  # Change shape from (256, 256, 4) to (4, 256, 256)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        prediction = model(input_tensor)
        predicted_windows.append(prediction[0].detach().numpy())

    return predicted_windows

def plotter(model_outputs, desired_outputs, input_images, onehot_to_rgb):
    num_samples = len(model_outputs)

    for i in range(num_samples):
        # Convert model output to RGB
        predicted_output = onehot_to_rgb(model_outputs[i])

        # Get input, desired output, and original image
        input_image = input_images[i]
        desired_output = desired_outputs[i]

        # Plot images
        plt.figure(figsize=(12, 4))
        
        # Original Input
        plt.subplot(1, 4, 1)
        plt.title('Original Input')
        plt.imshow(input_image)
        
        # Desired Output
        plt.subplot(1, 4, 2)
        plt.title('Desired Output')
        plt.imshow(desired_output)

        # Predicted Output
        plt.subplot(1, 4, 3)
        plt.title('Predicted Output')
        plt.imshow(predicted_output)

        # Input with Overlapping Prediction
        plt.subplot(1, 4, 4)
        plt.title('Input with Prediction')
        plt.imshow(input_image)
        plt.imshow(predicted_output, alpha=0.5)

        plt.tight_layout()
        plt.show()