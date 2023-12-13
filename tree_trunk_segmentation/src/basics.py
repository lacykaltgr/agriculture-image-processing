import numpy as np

class Augment():
    # def __init__(self) -> None: # init returns None
        # pass # indication the method does nothing
    def __init__(self):
        self.x  = None
        self.y = None

    def horizontal_flip(self, image_array):
        # Get the height and width of the image
        height, width = image_array.shape[:2] # getting dimensions from tuple
        
        print("Shape:", image_array.shape)

        print("Size:", image_array.size)

        print("Data Type:", image_array.dtype)

        print("Array Type:", type(image_array))

        print("Number of Dimensions:", image_array.ndim)

        print("Item Size (bytes):", image_array.itemsize)


        # Create an empty array to store the flipped image
        flipped_image = np.empty_like(image_array) 

        # Loop on each row and reverse the order of elements
        for row in range(height):
            # [row, :] gets the entire row, (:) is a slice notation all elements along this axis
            # [row, ::-1] reverses the order of the elements in that row, slice with a step of -1
            flipped_image[row, :] = image_array[row, ::-1]  

        return flipped_image

    def contrastChange(self, array, factor):
        array = array.astype(float)

        # Apply contrast change
        augmented_array = np.clip(array * factor, 0, 255).astype(np.uint8)

        return augmented_array
        