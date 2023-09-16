# Data augmentation techniques for tree-trunk-segmentation project
# using article: https://medium.com/@rahuladream/multiply-your-dataset-using-data-augmentation-techniques-381aee8ff8f6

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_augment_image(image_path):
    # Load the original image
    original_image = cv2.imread(image_path)
    augmented_images = []

    # Position Augmentation (Translation)
    for dx, dy in [(20, -20), (-20, 20)]:  # Only keep translations
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        augmented_images.append(cv2.warpAffine(original_image, M, (original_image.shape[1], original_image.shape[0])))

    # Scaling
    for scale_factor in [0.8, 1.2]:
        augmented_images.append(cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor))

    '''
    # Cropping
    for crop_height, crop_width in [(100, 100), (200, 200)]:
        h, w = original_image.shape[:2]
        top = np.random.randint(0, h - crop_height)
        left = np.random.randint(0, w - crop_width)
        augmented_images.append(original_image[top:top+crop_height, left:left+crop_width])
    '''

    # Flipping
    # (Flipping might not be relevant for this task)

    # Padding
    for padding_size in [20, 50]:
        augmented_images.append(cv2.copyMakeBorder(original_image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0))

    '''
    # Rotation
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((original_image.shape[1]//2, original_image.shape[0]//2), angle, 1)
        augmented_images.append(cv2.warpAffine(original_image, M, (original_image.shape[1], original_image.shape[0])))
    '''
        
    # Color Augmentation (#also output)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    for value in [0.8, 1.2]:
        hsv_image[..., 2] = np.clip(hsv_image[..., 2] * value, 0, 255)
        augmented_images.append(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR))

    # Blurring (Gaussian Blur, Median Blur)
    for kernel_size in [3, 5]:
        augmented_images.append(cv2.GaussianBlur(original_image, (kernel_size, kernel_size), 0))
        augmented_images.append(cv2.medianBlur(original_image, kernel_size))

    return augmented_images

# Load and augment the image
input_image_path = '/Users/czimbermark/DeepLearning_Projects_23/agriculture-image-processing/tree-trunk-segmentation/data/4_3_images/P1010147.JPG'  # Replace with the actual path
augmented_images = load_and_augment_image(input_image_path)

# Plot the original and augmented images
num_images = len(augmented_images)
num_rows = (num_images + 3) // 4  # Calculate the number of rows needed
plt.figure(figsize=(15, 4 * num_rows))

# Plot the original image
plt.subplot(num_rows, 4, 1)
original_image = cv2.imread(input_image_path)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Plot the augmented images
for i, augmented_image in enumerate(augmented_images):
    plt.subplot(num_rows, 4, i+2 if i < 11 else i+1)  # Start from the second subplot
    plt.imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Augmented Image {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()



