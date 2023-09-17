# Data augmentation techniques for tree-trunk-segmentation project
# using article: https://medium.com/@rahuladream/multiply-your-dataset-using-data-augmentation-techniques-381aee8ff8f6

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa



def load_and_augment_image(image_path):
    original_image = cv2.imread(image_path)
    augmented_images = []

    '''
    # Position Augmentation (Translation)
    for dx, dy in [(-20, -20), (20, -20), (-20, 20), (20, 20)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        augmented_images.append(cv2.warpAffine(original_image, M, (original_image.shape[1], original_image.shape[0])))
    '''

    '''
    # Scaling
    for scale_factor in [0.8, 1.2]:
        augmented_images.append(cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor))
    '''
        
    '''
    # Cropping
    for crop_height, crop_width in [(100, 100), (200, 200)]:
        h, w = original_image.shape[:2]
        top = np.random.randint(0, h - crop_height)
        left = np.random.randint(0, w - crop_width)
        augmented_images.append(original_image[top:top+crop_height, left:left+crop_width])
    '''

    # Flipping
    # augmented_images.append(cv2.flip(original_image, 0))  # Flip vertically
    augmented_images.append(cv2.flip(original_image, 1))  # Flip horizontally

    '''
    # Padding
    for padding_size in [20, 50]:
        augmented_images.append(cv2.copyMakeBorder(original_image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0))
    '''
        
    '''
    # Rotation
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((original_image.shape[1]//2, original_image.shape[0]//2), angle, 1)
        augmented_images.append(cv2.warpAffine(original_image, M, (original_image.shape[1], original_image.shape[0])))
    '''
        
    # Color Augmentation
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    for value in [0.8, 1.2]:
        hsv_image[..., 2] = np.clip(hsv_image[..., 2] * value, 0, 255)
        augmented_images.append(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR))

    # Gamma Correction (#also output)
    for gamma in [0.8, 1.2]:
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        augmented_images.append(cv2.LUT(original_image, table))

    # Hue Augmentation (#also output)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    for value in [-20, 20]:
        hsv_image[..., 0] = (hsv_image[..., 0] + value) % 180
        augmented_images.append(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR))

    # Saturation Augmentation (#also output)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    for value in [0.8, 1.2]:
        hsv_image[..., 1] = np.clip(hsv_image[..., 1] * value, 0, 255)
        augmented_images.append(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR))

    # Contrast Augmentation
    for alpha in [0.8, 1.2]:
        augmented_images.append(cv2.convertScaleAbs(original_image, alpha=alpha, beta=0))

    # Weather Augmentation
    seq = iaa.Sequential([
        iaa.Fog(),   # Adjust fog parameters
        iaa.Rain(),  # Adjust rain density
        iaa.Clouds()  # Use default parameters for clouds
    ])

    augmented_images.extend(seq(images=[original_image]))

    seq = iaa.Sequential([iaa.ElasticTransformation(alpha=50.0, sigma=5.0)])
    augmented_images.extend(seq(images=[original_image]))

    '''
    seq = iaa.Sequential([iaa.Affine(shear=(-20, 20))])
    augmented_images.extend(seq(images=[original_image]))
    '''
    
    # Blurring (Gaussian Blur, Median Blur)
    for kernel_size in [3, 5]:
        augmented_images.append(cv2.GaussianBlur(original_image, (kernel_size, kernel_size), 0))
        augmented_images.append(cv2.medianBlur(original_image, kernel_size))

    return augmented_images


# Load and augment the image
input_image_path = '/Users/czimbermark/DeepLearning_Projects_23/agriculture-image-processing/tree-trunk-segmentation/data/4_3_images/P1010147.JPG'  # Replace with the actual path
augmented_images = load_and_augment_image(input_image_path)

# List of custom names for augmented images
custom_names = [
    "Horizontal Flipping",
    "Color Augmentation (Darkened)",
    "Color Augmentation (Lightened)",
    "Gamma Correction (Darkened)",
    "Gamma Correction (Lightened)",
    "Hue Shift (Left)",
    "Hue Shift (Right)",
    "Saturation Decrease",
    "Saturation Increase",
    "Contrast Decrease",
    "Contrast Increase",
    "Foggy Weather",
    "Rainy Weather",
    "Cloudy Weather",
    "Elastic Transformation",
    "Gaussian Blurred (3x3 Kernel)",
    "Gaussian Blurred (5x5 Kernel)",
    "Median Blurred (3x3 Kernel)"
]
'''

# Plot the original and augmented images
num_images = len(augmented_images)
num_cols = 4  # Number of columns
num_rows = (num_images + num_cols - 1) // num_cols  # Calculate the number of rows needed
plt.figure(figsize=(15, 4 * num_rows))

# Plot the original image
plt.subplot(num_rows, num_cols, 1)
original_image = cv2.imread(input_image_path)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Plot the augmented images
for i, augmented_image in enumerate(augmented_images):
    title = custom_names[i] if i < len(custom_names) else f'Augmented Image {i+1}'
    plt.subplot(num_rows, num_cols, i+2)  # Start from the second subplot
    plt.imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

'''

def augment_dataset_with_labels(x, y):
    augmented_x = []
    augmented_y = []

    for i in range(len(x)):
        image_x = x[i]
        image_y = y[i]

        # Append y
        augmented_y.append(image_y.copy())

        # Apply horizontal flip to y and maintain labels
        augmented_y.append(cv2.flip(image_y, 1).copy())

        # Append x
        augmented_x.append(image_x.copy())

        # Apply color augmentation to x (darker version)
        hsv_image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2HSV)
        darkened_hsv_image_x = hsv_image_x.copy()
        darkened_hsv_image_x[..., 2] = np.clip(darkened_hsv_image_x[..., 2] * 0.8, 0, 255)
        augmented_x.append(cv2.cvtColor(darkened_hsv_image_x, cv2.COLOR_HSV2BGR))

        # Apply contrast augmentation to x (increase)
        augmented_x.append(cv2.convertScaleAbs(image_x, alpha=1.2, beta=0))

        # Apply horizontal flip to x and maintain labels
        augmented_x.append(cv2.flip(image_x, 1).copy())

    return augmented_x, augmented_y


import os

def load_images(folder_path):
    x_images = []
    y_images = []
    
    for filename in os.listdir(folder_path):
        if filename.startswith("DSC") and filename.endswith(".JPG"):
            x_images.append(cv2.imread(os.path.join(folder_path, filename)))
        elif filename.startswith("DSC") and filename.endswith("_train.png"):
            y_images.append(cv2.imread(os.path.join(folder_path, filename)))

    return x_images, y_images



def plot_images(x, y):
    num_images = len(x) + len(y)
    num_cols = 6  # Number of columns
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate the number of rows needed

    plt.figure(figsize=(15, 4 * num_rows))

    for i, image in enumerate(x + y):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Image {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

folder_path = '/Users/czimbermark/DeepLearning_Projects_23/agriculture-image-processing/tree-trunk-segmentation/data/3_2_images'
x_images, y_images = load_images(folder_path)

augmented_x, augmented_y = augment_dataset_with_labels(x_images, y_images)

plot_images(augmented_y, augmented_x)
