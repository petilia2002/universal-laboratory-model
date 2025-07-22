import numpy as np


def calc_confusion_matrix(y_real, y_pred, threshold):

    y_real = np.reshape(y_real, (-1))
    y_pred = np.reshape(y_pred, (-1))

    tp = tn = fn = fp = 0

    for i in range(len(y_real)):
        if y_pred[i] >= threshold and y_real[i] == 1:
            tp = tp + 1
        if y_pred[i] < threshold and y_real[i] == 0:
            tn = tn + 1
        if y_pred[i] >= threshold and y_real[i] == 0:
            fp = fp + 1
        if y_pred[i] < threshold and y_real[i] == 1:
            fn = fn + 1

    return tp, tn, fp, fn
