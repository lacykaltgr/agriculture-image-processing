
from utils import *

color_dict = {0: (0, 0, 0),
              1: (0, 125, 0),
              2: (150, 80, 0),
              3: (255, 255, 0),
              4: (100, 100, 100),
              5: (0, 255, 0),
              6: (0, 0, 150),
              7: (150, 150, 255),
              8: (255, 255, 255)}


def test(model, trainx, trainy, testx, testy, weights_file="model_oneshot.h5"):
    pred_train_all = []
    pred_val_all = []

    model.load_weights(weights_file)
    Y_pred_train = model.predict(trainx)
    for k in range(Y_pred_train.shape[0]):
        pred_train_all.append(Y_pred_train[k])
    Y_gt_train = [rgb_to_onehot(arr, color_dict) for arr in trainy]
    Y_pred_val = model.predict(testx)

    for k in range(Y_pred_val.shape[0]):
        pred_val_all.append(Y_pred_val[k])
    Y_gt_val = [rgb_to_onehot(arr, color_dict) for arr in testy]

    return pred_train_all, Y_gt_train, pred_val_all, Y_gt_val


def conf_matrix(Y_gt, Y_pred, num_classes = 9):

    total_pixels = 0
    kappa_sum = 0
    sudo_confusion_matrix = np.zeros((num_classes, num_classes))

    #    if len(Y_pred.shape) == 3:
    #        h,w,c = Y_pred.shape
    #        Y_pred = np.reshape(Y_pred, (1,))

    n = len(Y_pred)

    for i in range(n):
        y_pred = Y_pred[i]
        y_gt = Y_gt[i]

        #y_pred_hotcode = hotcode(y_pred)
        #y_gt_hotcode = hotcode(y_gt)

        pred = np.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2]))
        gt = np.reshape(y_gt, (y_gt.shape[0]*y_gt.shape[1], y_gt.shape[2]))

        pred = [i for i in pred]
        gt = [i for i in gt]

        pred = to_class_no(pred)
        gt = to_class_no(gt)

        #        pred.tolist()
        #        gt.tolist()

        gt = np.asarray(gt, dtype = 'int32')
        pred = np.asarray(pred, dtype = 'int32')

        conf_matrix = confusion_matrix(gt, pred, labels=[0,1,2,3,4,5,6,7,8])

        kappa = cohen_kappa_score(gt,pred, labels=[0,1,2,3,4,5,6,7])

        pixels = len(pred)
        total_pixels = total_pixels+pixels

        sudo_confusion_matrix = sudo_confusion_matrix + conf_matrix

        kappa_sum = kappa_sum + kappa

    final_confusion_matrix = sudo_confusion_matrix

    final_kappa = kappa_sum/n

    return final_confusion_matrix, final_kappa