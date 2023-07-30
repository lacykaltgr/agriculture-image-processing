import PIL
from PIL import Image
import matplotlib.pyplot as plt
from libtiff import TIFF
from scipy.misc import imresize
import numpy as np
import glob
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from unet import UNet

model = UNet()

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')





#pred_train_all, pred_test_all, Y_pred_val, Y_gt_val = testing(model, trainx, trainy, testx, testy, weights_file = "model_onehot.h5")

##pred_train_all, Y_gt_train, pred_val_all, Y_gt_val = testing(model, trainx, trainy, testx, testy, weights_file = "model_onehot.h5")

pred_train_13, Y_gt_train_13, pred_val_all, Y_gt_val = testing_diffsizes(model, x_train, y_train, x_val, y_val, weights_file = "model_onehot.h5")

print(pred_val_all[0].shape)
print(Y_gt_val[0].shape)
#print(len(pred_train_all))
#print(len(Y_gt_train))

# Convert onehot to label






confusion_matrix_train, kappa_train = conf_matrix(Y_gt_train_13, pred_train_13, num_classes = 9)
print('Confusion Matrix for training')
print(confusion_matrix_train)
print('Kappa Coeff for training without unclassified pixels')
print(kappa_train)

confusion_matrix_test, kappa_test = conf_matrix(Y_gt_val, pred_val_all, num_classes = 9)
print('Confusion Matrix for validation')
print(confusion_matrix_test)
print('Kappa Coeff for validation without unclassified pixels')
print(kappa_test)


# Pass Confusion matrix, label to which the accuracy needs to be found, number of classes to be considered
# Returns that particular class accuracy

def acc_of_class(class_label, conf_matrix, num_classes = 8):

    numerator = conf_matrix[class_label, class_label]

    denorminator = 0

    for i in range(num_classes):
        denorminator = denorminator + conf_matrix[class_label, i]

    acc_of_class = numerator/denorminator

    return acc_of_class


# On training

# Find accuray of all the classes NOT considering the unclassified pixels

for i in range(8):
    acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_train, num_classes = 8)
    print('Accuracy of class '+str(i) + ' WITHOUT unclassified pixels - Training')
    print(acc_of_cl)

# Find accuray of all the classes considering the unclassified pixels

for i in range(9):
    acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_train, num_classes = 9)
    print('Accuracy of class '+str(i) + ' WITH unclassified pixels - Training')
    print(acc_of_cl)

# On validation

# Find accuray of all the classes NOT considering the unclassified pixels

for i in range(8):
    acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_test, num_classes = 8)
    print('Accuracy of class '+str(i) + ' WITHOUT unclassified pixels - Validation')
    print(acc_of_cl)

# Find accuray of all the classes considering the unclassified pixels

for i in range(9):
    acc_of_cl = acc_of_class(class_label = i, conf_matrix = confusion_matrix_test, num_classes = 9)
    print('Accuracy of class '+str(i) + ' WITH unclassified pixels - Validation')
    print(acc_of_cl)


# Calulating over all accuracy with and without unclassified pixels

def overall_acc(conf_matrix, include_unclassified_pixels = False):

    if include_unclassified_pixels:

        numerator = 0
        for i in range(9):

            numerator = numerator + conf_matrix[i,i]

        denominator = 0
        for i in range(9):
            for j in range(9):

                denominator = denominator + conf_matrix[i,j]

        acc = numerator/denominator

        return acc

    else:

        numerator = 0
        for i in range(8):

            numerator = numerator + conf_matrix[i,i]

        denominator = 0
        for i in range(8):
            for j in range(8):

                denominator = denominator + conf_matrix[i,j]

        acc = numerator/denominator

        return acc


# Training

# Over all accuracy without unclassified pixels

print('Over all accuracy WITHOUT unclassified pixels - Training')
print(overall_acc(conf_matrix = confusion_matrix_train, include_unclassified_pixels = False))

# Over all accuracy with unclassified pixels

print('Over all accuracy WITH unclassified pixels - Training')
print(overall_acc(conf_matrix = confusion_matrix_train, include_unclassified_pixels = True))

# Validation

# Over all accuracy without unclassified pixels

print('Over all accuracy WITHOUT unclassified pixels - Validation')
print(overall_acc(conf_matrix = confusion_matrix_test, include_unclassified_pixels = False))

# Over all accuracy with unclassified pixels

print('Over all accuracy WITH unclassified pixels - Validation')
print(overall_acc(conf_matrix = confusion_matrix_test, include_unclassified_pixels = True))



# Convert decimal onehot encode from prediction to actual onehot code

def dec_to_onehot(pred_all):

    pred_all_onehot_list = []

    for img in pred_all:

        h, w, c = img.shape

        for i in range(h):
            for j in range(w):

                argmax_index = np.argmax(img[i,j])

                sudo_onehot_arr = np.zeros((9))

                sudo_onehot_arr[argmax_index] = 1

                onehot_encode = sudo_onehot_arr

                img[i,j,:] = onehot_encode

        pred_all_onehot_list.append[img]

    return pred_all_onehot_list



color_dict = {0: (0, 0, 0),
              1: (0, 125, 0),
              2: (150, 80, 0),
              3: (255, 255, 0),
              4: (100, 100, 100),
              5: (0, 255, 0),
              6: (0, 0, 150),
              7: (150, 150, 255),
              8: (255, 255, 255)}

def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    print(shape)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)


# Pred on train, val, test and save outputs

weights_file = "model_onehot.h5"
model.load_weights(weights_file)

#y_pred_test_all = []

xtrain_list.append(x_val)


for i_ in range(len(xtrain_list)):

    item = xtrain_list[i_]

    h,w,c = item.shape

    item = np.reshape(item,(1,h,w,c))

    y_pred_train_img = model.predict(item)

    ba,h,w,c = y_pred_train_img.shape

    y_pred_train_img = np.reshape(y_pred_train_img,(h,w,c))

    img = y_pred_train_img
    h, w, c = img.shape

    for i in range(h):
        for j in range(w):

            argmax_index = np.argmax(img[i,j])

            sudo_onehot_arr = np.zeros((9))

            sudo_onehot_arr[argmax_index] = 1

            onehot_encode = sudo_onehot_arr

            img[i,j,:] = onehot_encode

    y_pred_train_img = onehot_to_rgb(img, color_dict)

    tif = TIFF.open(filelist_trainx[i_])
    image2 = tif.read_image()

    h,w,c = image2.shape

    y_pred_train_img = y_pred_train_img[:h, :w, :]

    imx = Image.fromarray(y_pred_train_img)

    imx.save("train_predictions/pred"+str(i_+1)+".jpg")



for i_ in range(len(xtest_list1)):

    item = xtest_list1[i_]

    h,w,c = item.shape

    item = np.reshape(item,(1,h,w,c))

    y_pred_test_img = model.predict(item)

    ba,h,w,c = y_pred_test_img.shape

    y_pred_test_img = np.reshape(y_pred_test_img,(h,w,c))

    img = y_pred_test_img
    h, w, c = img.shape

    for i in range(h):
        for j in range(w):

            argmax_index = np.argmax(img[i,j])

            sudo_onehot_arr = np.zeros((9))

            sudo_onehot_arr[argmax_index] = 1

            onehot_encode = sudo_onehot_arr

            img[i,j,:] = onehot_encode

    y_pred_test_img = onehot_to_rgb(img, color_dict)

    tif = TIFF.open(filelist_testx[i_])
    image2 = tif.read_image()

    h,w,c = image2.shape

    y_pred_test_img = y_pred_test_img[:h, :w, :]

    imx = Image.fromarray(y_pred_test_img)

    imx.save("test_outputs/out"+str(i_+1)+".jpg")