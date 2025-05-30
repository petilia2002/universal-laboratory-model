from configparser import ConfigParser
import fdb
import numpy as np
import math
import sys
import pickle
import datetime
import os

import keras

from keras import Model, regularizers
from keras.layers import Input, Dense, Embedding
from keras.layers import Dropout, Flatten, Lambda
from keras.layers import Concatenate, Multiply, Add
from keras.layers import GaussianNoise, LayerNormalization

from keras.callbacks import EarlyStopping
from keras.utils import set_random_seed

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.metrics import roc_curve, auc

import tensorflow
import tensorflow.python.keras.backend as K

config_object = ConfigParser()
config_object.read("./config.ini")

user_info = config_object["USERINFO"]
server_config = config_object["SERVERCONFIG"]

dsn = server_config["dsn"]
user = user_info["user"]
password = user_info["password"]

con = fdb.connect(dsn=dsn, user=user, password=password)

print(tensorflow.config.list_physical_devices("GPU"))

idx = [
    34,
    35,
    19,
    20,
    21,
    23,
    22,
    24,
    25,
    26,
    27,
    28,
    36,
    37,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    18,
    29,
]
tests = [1, 31, 30, 32]

ANALYTE = 0

NN = len(idx)
MASKED_VALUE = -1

Limits = {}

for i in range(len(idx)):
    Limits[idx[i]] = [sys.float_info.max, -sys.float_info.max, 0]
for i in range(len(tests)):
    Limits[tests[i]] = [sys.float_info.max, -sys.float_info.max, 0]

X = []
W = []
Y = []

cnt = 0.0

for (
    gender,
    age,
    hgb,
    rbc,
    mcv,
    plt,
    wbc,
    neut,
    lymph,
    eo,
    baso,
    mono,
    mid,
    gra,
    b12,
    folic,
    ast,
    alt,
    bil_direct,
    bil_indirect,
    bil_total,
    crea,
    urea,
    pro,
    ldg,
    alb,
    crp,
    ferritin,
    glu,
    chol,
    uric,
) in con.cursor().execute(
    """select a.gender, a.age, a.hgb, a.rbc, a.mcv, a.plt, 
                                      a.wbc, a.neut, a.lymph, a.eo, a.baso, a.mono,
                                      a.mid, a.gra, a.b12, a.folic, a.ast, a.alt, 
                                      a.bil_direct, a.bil_indirect, a.bil_total, 
                                      a.crea, a.urea, a.pro, a.ldg, a.alb, a.crp, 
                                      a.ferritin, a.glu, a.chol, a.uric
                                from lab2 a 
                                where a.ferritin is not null or a.glu is not null 
                                      or a.chol is not null or a.uric is not null 
                                      and a.id <= 1671132 """
):

    y = [MASKED_VALUE, MASKED_VALUE, MASKED_VALUE, MASKED_VALUE]

    if ferritin is not None:
        y[0] = 1 if ferritin <= 12 else 0
    if glu is not None:
        y[1] = 1 if glu >= 7 else 0
    if chol is not None:
        y[2] = 1 if chol >= 5.2 else 0
    if uric is not None:
        y[3] = (
            1
            if ((gender == 1 and uric >= 0.48) or (gender == 0 and uric >= 0.38))
            else 0
        )

    if y[ANALYTE] == MASKED_VALUE:
        continue

    x = [0 for i in range(NN)]
    w = [0 for i in range(NN)]

    def add_value(analyte, ind):

        if analyte is not None:
            v = float(analyte)
            if ind != 34:
                v = math.log(v)

            if Limits[ind][0] > v:
                Limits[ind][0] = v
            if Limits[ind][1] < v:
                Limits[ind][1] = v

            Limits[ind][2] = Limits[ind][2] + 1

            if ind not in tests:

                z = idx.index(ind)

                x[z] = v
                w[z] = ind

    add_value(gender, 34)
    add_value(age, 35)

    add_value(hgb, 19)
    add_value(rbc, 20)
    add_value(mcv, 21)
    add_value(plt, 23)
    add_value(wbc, 22)
    add_value(neut, 24)
    add_value(lymph, 25)
    add_value(eo, 26)
    add_value(baso, 27)
    add_value(mono, 28)

    add_value(mid, 36)
    add_value(gra, 37),

    add_value(b12, 2)
    add_value(folic, 3)
    add_value(ast, 4)
    add_value(alt, 5)
    add_value(bil_direct, 6)
    add_value(bil_indirect, 7)
    add_value(bil_total, 8)
    add_value(crea, 9)
    add_value(urea, 10)
    add_value(pro, 11)
    add_value(ldg, 12)
    add_value(alb, 18)
    add_value(crp, 29)

    add_value(ferritin, 1)
    add_value(glu, 31)
    add_value(chol, 30)
    add_value(uric, 32)

    X.append(x)
    W.append(w)
    Y.append(y)

    # print(x);
    # print(w);
    # print(y);

for u in Limits:
    print(u, Limits[u])

l_in = Input(name="values", shape=(NN,))
l_f = Dense(units=256, activation="relu")(l_in)
l_out_cbc = Dense(units=1, activation="sigmoid")(l_f)


def cbcLoss(y_true, y_pred):
    mask = K.cast_to_floatx(K.not_equal(y_true, MASKED_VALUE))
    y_pred = tensorflow.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    r = y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred)
    return -tensorflow.reduce_mean(r * mask)


model = Model(l_in, l_out_cbc)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy"
)
model.summary()


ids = np.arange(0, len(X), 1, dtype=int)
np.random.shuffle(ids)

train_size = (int)(len(ids) * 0.8)
test_size = len(ids) - train_size

x_train = np.zeros((train_size, NN), dtype="float32")
w_train = np.zeros((train_size, NN), dtype="int32")
m_train = np.zeros((train_size, NN), dtype="int32")
y_train = np.zeros((train_size), dtype="float32")

x_test = np.zeros((test_size, NN), dtype="float32")
w_test = np.zeros((test_size, NN), dtype="int32")
m_test = np.zeros((test_size, NN), dtype="int32")
y_test = np.zeros((test_size), dtype="float32")

for i in range(train_size):
    for j in range(NN):

        v = X[ids[i]][j]
        w = W[ids[i]][j]

        if w == 0:
            break

        y_min, y_max = Limits[w][0], Limits[w][1]

        add = 0.01 * (y_max - y_min)
        y_max = y_max + add
        y_min = y_min - add

        x_train[i][j] = 0.9 - 0.8 * (y_max - v) / (y_max - y_min)

    y_train[i] = Y[ids[i]][ANALYTE]
    w_train[i] = W[ids[i]]

    m_train[i] = W[ids[i]]
    for j in range(NN):
        if m_train[i][j] > 0:
            m_train[i][j] = 1


j = 0
for i in range(train_size, len(ids)):
    for k in range(NN):

        v = X[ids[i]][k]
        w = W[ids[i]][k]

        if w == 0:
            break

        y_min, y_max = Limits[w][0], Limits[w][1]

        add = 0.01 * (y_max - y_min)
        y_max = y_max + add
        y_min = y_min - add

        x_test[j][k] = 0.9 - 0.8 * (y_max - v) / (y_max - y_min)

    y_test[j] = Y[ids[i]][ANALYTE]
    w_test[j] = W[ids[i]]

    m_test[j] = W[ids[i]]
    for k in range(NN):
        if m_test[j][k] > 0:
            m_test[j][k] = 1

    j = j + 1

print("Training sizes: ", len(x_train), len(y_train))
print("Test sizes: ", len(x_test), len(y_test))

m_pred_train = np.zeros((train_size, 4), dtype="float32")
m_pred_test = np.zeros((test_size, 4), dtype="float32")

m_pred_train[:, :] = tests
m_pred_test[:, :] = tests


a = model(x_train[0:2])

print(a.shape)
print(a)

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1
)

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

reduce = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=20,
    verbose=0,
    mode="auto",
    min_delta=0.00001,
    cooldown=0,
    min_lr=1e-7,
)


class MessagerCallback(tensorflow.keras.callbacks.Callback):

    def __init__(self, **kwargs):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if os.path.exists("stop"):
            self.model.stop_training = True
        return


# model.load_weights("base-multi.weights.h5");

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
        tensorflow.keras.callbacks.TerminateOnNaN(),
        MessagerCallback(),
    ],
)

model.save_weights("one/ferritin-3.weights.h5", True)

y_pred = model.predict(x_train)
y2_pred = model.predict(x_test)


def calcRocs():

    q = []
    v = []

    qe = []
    qr = []

    for i in range(len(y_train)):
        if y_train[i] != MASKED_VALUE:
            q.append(y_train[i])
            v.append(y_pred[i])

    for i in range(len(y2_pred)):
        if y_test[i] != MASKED_VALUE:
            qe.append(y_test[i])
            qr.append(y2_pred[i])

    fpr, tpr, thresholds = roc_curve(q, v)
    roc_auc = auc(fpr, tpr)

    label = ""

    if ANALYTE == 0:
        label = "Ferritin"
    elif ANALYTE == 1:
        label = "Glucose"
    elif ANALYTE == 2:
        label = "Cholesterol"
    elif ANALYTE == 3:
        label = "Uric"

    fp = open("/home/ilya/oak/stats/" + label + "-one.txt", "w")
    for i in range(len(fpr)):
        print(fpr[i], tpr[i], file=fp)
    fp.close()

    return fpr, tpr, thresholds, qe, qr, roc_auc


calcRocs()


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


def calc_stats():

    fpr, tpr, thresholds, qe, qr, roc_auc = calcRocs()

    m1 = 10  # некие значение явно больше и меньше диапазона FPR / TPR
    m2 = -10

    ot1 = ot2 = 0.0
    for i in range(len(fpr)):
        r = tpr[i] - (1.0 - fpr[i])

        if r >= 0:
            # значение положительное, то есть мы движемся сверху от 1 - x
            if r <= m1:
                m1 = r
                ot1 = thresholds[i]
        else:
            # мы в под прямой 1 - x
            if r > m2:
                m2 = r
                ot2 = thresholds[i]

    ot = (ot1 + ot2) / 2.0

    tp, tn, fp, fn = calc_confusion_matrix(qe, qr, ot)

    f1 = 2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None
    accuracy = (
        (tp + tn) * 100.0 / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else None
    )

    sensitivity = tp * 100.0 / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn * 100.0 / (tn + fp) if (tn + fp) > 0 else None

    name = ""

    if ANALYTE == 0:
        name = "Ferritin"
    if ANALYTE == 1:
        name = "Glucose"
    if ANALYTE == 2:
        name = "Cholesterol"
    if ANALYTE == 3:
        name = "Uric Acid"

    print("AUC (", name, ") = ", roc_auc, "\n")
    print("Optimal threshold = ", ot)
    print("TP = ", tp)
    print("TN = ", tn)
    print("FP = ", fp)
    print("FN = ", fn, "\n")
    print("N = ", tp + tn + fp + fn)

    print("F1_score = ", f1)
    print("Accuracy = ", accuracy)
    print("Sensitivity = ", sensitivity)
    print("Specificity = ", specificity)

    print("")

    ppv = tp / (tp + fp) if (tp + fp) > 0 else None
    npv = tn / (tn + fn) if (tn + fn) > 0 else None

    print("Positive Predictive Value = ", ppv)
    print("Negative Predictive Value = ", npv)

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


calc_stats()

print("Relax")
