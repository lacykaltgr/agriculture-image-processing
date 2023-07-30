
from src.utils import *
from src.utils import numericalSort
import imageio


def load_images(fnames, crop_size=128, stride=32):
    d_list = []
    for fname in fnames[:13]:
        image_raw = imageio.read(fname)
        image = np.array(image_raw.get_data(0))
        h, w, c = image.shape
        n_h = int(int(h/stride))
        n_w = int(int(w/stride))
        image = padding(image, w, h, c, crop_size, stride, n_h, n_w)
        d_list.append(image)
    return np.array(d_list, dtype=np.float32)


def load_dataset():
    import glob

    # List of file names of actual Satellite images for traininig
    filelist_trainx = sorted(glob.glob('/Users/laszlofreund/code/ai/satellite-image-segmentation/data/The-Eye-in-the-Sky-dataset/sat/*.tif'), key=numericalSort)
    # List of file names of classified images for traininig
    filelist_trainy = sorted(glob.glob('/Users/laszlofreund/code/ai/satellite-image-segmentation/data/The-Eye-in-the-Sky-dataset/gt/*.tif'), key=numericalSort)
    # List of file names of actual Satellite images for testing
    filelist_testx = sorted(glob.glob('/Users/laszlofreund/code/ai/satellite-image-segmentation/data/The-Eye-in-the-Sky-test-data/sat_test/*.tif'), key=numericalSort)

    # Making array of all the training sat images as it is without any cropping
    x_train = load_images(filelist_trainx[:13])
    ytrain = load_images(filelist_trainy[:13])
    x_test = load_images(filelist_testx)

    x_val = x_train[:-1]
    y_val = ytrain[-1]

    return x_train, ytrain, x_val, y_val, x_test


