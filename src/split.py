import numpy as np

class ImageProcessor:
    class Window:
        def __init__(self, size, band):
            self.data = np.zeros((size, size, band), dtype=np.uint8)

    def __init__(self, w, h, band, size):
        self.w = w
        self.h = h
        self.band = band
        self.size = size
        self.data = np.zeros((h, w, band), dtype=np.uint8)
        self.windows = []


    def split(self, window_size=256):
        pad_w = (window_size - (self.w % window_size)) % window_size
        pad_h = (window_size - (self.h % window_size)) % window_size

        print("Padding:", (pad_h, pad_w))

        # Calculate the padding values
        padded_w = self.w + pad_w
        padded_h = self.h + pad_h

        # Pad the image using np.pad() with constant values
        padded_data = np.pad(self.data, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

        print("Original Image Shape:", self.data.shape)
        print("Padded Image Shape:", padded_data.shape)

        for y in range(0, padded_h, window_size):
            for x in range(0, padded_w, window_size):
                w = self.Window(window_size, self.band)
                self.windows.append(w)
                for sy in range(window_size):
                    for sx in range(window_size):
                        if (y + sy) < padded_h and (x + sx) < padded_w:  # Ensure valid indices
                            for i in range(self.band):
                                w.data[sy, sx, i] = padded_data[y + sy, x + sx, i]

        return self.windows
    

    def unsplit(self):
        reconstructed_data = np.zeros((self.h, self.w, self.band), dtype=np.uint8)
        num_windows_per_row = self.w // self.size

        for y in range(self.h):
            for x in range(self.w):
                by = y // self.size
                bx = x // self.size
                w_index = by * num_windows_per_row + bx
                w_obj = self.windows[w_index]

                px = x % self.size
                py = y % self.size

                for i in range(self.band):
                    reconstructed_data[y, x, i] = w_obj.data[py, px, i]

        return reconstructed_data

