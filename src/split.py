import numpy as np
from math import ceil

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


    def split(self, window_size=256, overlap=128):
        windows = []

        for y in range(0, self.h, window_size - overlap):
            for x in range(0, self.w, window_size - overlap):
                w = self.Window(window_size, self.band)
                for sy in range(min(window_size, self.h - y)):
                    for sx in range(min(window_size, self.w - x)):
                        for i in range(self.band):
                            w.data[sy, sx, i] = self.data[y + sy, x + sx, i]
                windows.append(w)

        return windows


    def unsplit(self, predicted_windows, overlap=128):
        num_predicted_channels = predicted_windows[0].shape[0]
        reconstructed_data = np.zeros((self.h, self.w, num_predicted_channels), dtype=np.uint8)
        windows_per_row = ceil((self.w + overlap) / self.size)

        for w_index in range(len(predicted_windows)):
            by = w_index // windows_per_row
            bx = w_index % windows_per_row
            w_obj = predicted_windows[w_index]

            for py in range(self.size):
                for px in range(self.size):
                    x = bx * (self.size - overlap) + px
                    y = by * (self.size - overlap) + py

                    for i in range(num_predicted_channels):
                        if x < self.w and y < self.h:
                            reconstructed_data[y, x, i] = w_obj[i, py, px]

        return reconstructed_data
