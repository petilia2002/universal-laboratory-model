import datetime
import sys
import math

import numpy as np
from configparser import ConfigParser

from data import createDatasets
from models import createUlm, createMultiTask, createSingle, createXGBoosting
from callbacks import MessagerCallback

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from data import MASKED_VALUE
from utils import calc_confusion_matrix

config_object = ConfigParser()
config_object.read("./config.ini")

fs_config = config_object["FSCONFIG"]

load_file = fs_config["load_file_weights"]
save_file = fs_config["save_file_weights"]
model_type = fs_config["model_type"]

embeddings_file = fs_config["embeddings_file"]

tests = []  # ids
labels = []  # names

if model_type == "ulm":
    tests = [1, 31, 30, 32]
    labels = ["Ferritin", "Glucose", "Cholesterol", "Uric acid"]
elif model_type == "multi":
    tests = [1, 31, 30, 32]
    labels = ["Ferritin", "Glucose", "Cholesterol", "Uric acid"]
elif model_type == "single":
    tests = [32]
    labels = ["Uric acid"]
elif model_type == "GB":
    tests = [30]
    labels = ["Cholesterol"]
else:
    print("Unknown model type. Check config, please.", model_type)
    sys.exit(1)

(
    x_train,
    w_train,
    m_train,
    y_train,
    m_pred_train,
    x_test,
    w_test,
    m_test,
    y_test,
    m_pred_test,
) = createDatasets(tests, config_object)

model = None
if model_type == "ulm":
    model = createUlm(
        embeddings_file=embeddings_file, N_train=x_train.shape[1], N_pred=len(tests)
    )
elif model_type == "multi":
    model = createMultiTask(N_train=x_train.shape[1], N_pred=len(tests))
elif model_type == "single":
    model = createSingle(N_train=x_train.shape[1])
elif model_type == "GB":
    model = createXGBoosting()
if model is None:
    print("No model is built. Check config.")
    sys.exit(1)


log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=100,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0,
)
reduce = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=20,
    verbose=0,
    mode="auto",
    min_delta=0.00001,
    cooldown=0,
    min_lr=1e-7,
)

# Load previous weights if any.

try:
    model.load_weights(load_file)
except:
    pass

if model_type == "ulm":
    model.fit(
        x=[x_train, w_train, m_train, m_pred_train],
        y=y_train,
        shuffle=True,
        batch_size=32,
        epochs=1000,
        validation_split=0.2,
        callbacks=[
            tensorboard_callback,
            early_stopping,
            reduce,
            tf.keras.callbacks.TerminateOnNaN(),
            MessagerCallback(),
        ],
    )

    y_pred = model.predict([x_train, w_train, m_train, m_pred_train])
    y2_pred = model.predict([x_test, w_test, m_test, m_pred_test])
    model.save_weights(save_file, True)

elif model_type == "multi" or model_type == "single":
    model.fit(
        x=x_train,
        y=y_train,
        shuffle=True,
        batch_size=32,
        epochs=1000,
        validation_split=0.2,
        callbacks=[
            tensorboard_callback,
            early_stopping,
            reduce,
            tf.keras.callbacks.TerminateOnNaN(),
            MessagerCallback(),
        ],
    )

    y_pred = model.predict(x_train)
    y2_pred = model.predict(x_test)
    print(y2_pred)
    print(y2_pred.shape)
    model.save_weights(save_file, True)

elif model_type == "GB":
    model.fit(x_train, y_train)

    y_pred = model.predict_proba(x_train)
    y2_pred = model.predict_proba(x_test)

    y_pred = y_pred[:, 1]
    y2_pred = y2_pred[:, 1]

    y_pred = y_pred.reshape(-1, 1)
    y2_pred = y2_pred.reshape(-1, 1)


# Calc stats
def calcRocs(prop):

    q = []
    v = []

    qe = []
    qr = []

    for i in range(len(y_train)):
        if y_train[i, prop] != MASKED_VALUE:
            q.append(y_train[i, prop])
            v.append(y_pred[i, prop])

    for i in range(len(y2_pred)):
        if y_test[i, prop] != MASKED_VALUE:
            qe.append(y_test[i, prop])
            qr.append(y2_pred[i, prop])

    fpr, tpr, thresholds = roc_curve(q, v)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, thresholds, qe, qr, roc_auc


def calc_stats(analyte, name):

    fpr, tpr, thresholds, qe, qr, roc_auc = calcRocs(analyte)

    m1 = 10
    m2 = -10

    ot1 = ot2 = 0.0
    for i in range(len(fpr)):
        r = tpr[i] - (1.0 - fpr[i])

        if r >= 0:
            if r <= m1:
                m1 = r
                ot1 = thresholds[i]
        else:
            if r > m2:
                m2 = r
                ot2 = thresholds[i]

    ot = (ot1 + ot2) / 2.0

    tp, tn, fp, fn = calc_confusion_matrix(qe, qr, ot)

    f1 = 2.0 * tp / (2 * tp + fp + fn)
    accuracy = (tp + tn) * 100.0 / (tp + tn + fp + fn)

    sensitivity = tp * 100.0 / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn * 100.0 / (tn + fp) if (tn + fp) > 0 else None

    print("AUC (", name, ") = ", roc_auc, "\n")
    print("Optimal threshold = ", ot)
    print("TP = ", tp)
    print("TN = ", tn)
    print("FP = ", fp)
    print("FN = ", fn, "\n")

    print("F1_score = ", f1)
    print("Accuracy = ", accuracy)
    print("Sensitivity = ", sensitivity)
    print("Specificity = ", specificity)

    print("")

    ppv = tp / (tp + fp) if (tp + fp) > 0 else None
    npv = tn / (tn + fn) if (tn + fn) > 0 else None

    print("Positive Predictive Value = ", round(ppv, 2))
    print("Negative Predictive Value = ", round(npv, 2))

    print("")

    if fp > 0 and fn > 0 and tp > 0 and tn > 0:

        dor = tp * tn / fp / fn

        sigma = 1.96 * math.sqrt(1.0 / tp + 1.0 / tn + 1.0 / fp + 1.0 / fn)
        low = math.exp(math.log(dor) - sigma)
        high = math.exp(math.log(dor) + sigma)

        print(
            "DOR = ", round(dor, 1), " CI = [", round(low, 1), ", ", round(high, 1), "]"
        )

    print("\n\n")


for i in range(len(tests)):
    calc_stats(i, labels[i])

print("Relax")
