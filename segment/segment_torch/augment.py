import numpy as np


def horizontal_flip(image_array):
    # Flip along the second axis (width) for each RGB image
    flipped_images = image_array[:, :, ::-1, :]
    return flipped_images


def contrast_change(array, factor):
    # Iterate over each image in the batch
    copy_array = array.copy()
    for i in range(array.shape[0]):
        # Convert to float for precision during the contrast change
        array_float = copy_array[i].astype(float)
        # Clip values to [0, 255] in-place
        copy_array[i] = np.clip(array_float * factor, 0, 255).astype(np.uint8)

    return copy_array
