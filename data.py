import fdb
import sys

import math
import pickle
import numpy as np

from layers import MASKED_VALUE

# We use a FireBird database, the schema is provided.
# As we cannot share the data, you might implement a similar function
# to get data from your data lake.


def createDatasets(tests, config_object):

    user_info = config_object["USERINFO"]
    server_config = config_object["SERVERCONFIG"]

    fs_config = config_object["FSCONFIG"]
    analyte_limits_file = fs_config["analyte_limits_file"]

    dsn = server_config["dsn"]
    user = user_info["user"]
    password = user_info["password"]

    con = fdb.connect(dsn=dsn, user=user, password=password)

    # Our training parameters, values correspond to analytes.id

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

    NN = len(idx)

    # Define min / max values for each parameter.
    Limits = {}

    for i in range(len(idx)):
        Limits[idx[i]] = [sys.float_info.max, -sys.float_info.max, 0]
    for i in range(len(tests)):
        Limits[tests[i]] = [sys.float_info.max, -sys.float_info.max, 0]

    X = []  # values
    W = []  # words, i.e. analytes.id
    Y = []  # outputs 0 or 1

    # For reproducibility we use time split, all the models are being built for
    # sample.id <= 1671132.

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
                                where (a.ferritin is not null or a.glu is not null 
                                      or a.chol is not null or a.uric is not null)
                                      and a.id <= 1671132 """
    ):

        # Set all output values as unknown first.

        y = [MASKED_VALUE for i in range(len(tests))]

        if len(tests) == 1:
            # Single task settings.
            t = tests[0]

            if t == 1 and ferritin is not None:
                y[0] = 1 if ferritin <= 12 else 0
            elif t == 31 and glu is not None:
                y[0] = 1 if glu >= 7 else 0
            elif t == 30 and chol is not None:
                y[0] = 1 if chol >= 5.2 else 0
            elif t == 32 and uric is not None:
                y[0] = (
                    1
                    if (
                        (gender == 1 and uric >= 0.48) or (gender == 0 and uric >= 0.38)
                    )
                    else 0
                )

            if y[0] == MASKED_VALUE:
                continue

        else:
            if ferritin is not None:
                y[0] = 1 if ferritin <= 12 else 0
            if glu is not None:
                y[1] = 1 if glu >= 7 else 0
            if chol is not None:
                y[2] = 1 if chol >= 5.2 else 0
            if uric is not None:
                y[3] = (
                    1
                    if (
                        (gender == 1 and uric >= 0.48) or (gender == 0 and uric >= 0.38)
                    )
                    else 0
                )

        # Initialize input / output params.

        x = [0 for i in range(NN)]
        w = [0 for i in range(NN)]

        def add_value(analyte, ind):

            if analyte is not None:

                if ind not in Limits:
                    return

                v = float(analyte)
                if ind != 34:  # Do not log age.
                    v = math.log(v)

                if Limits[ind][0] > v:
                    Limits[ind][0] = v
                if Limits[ind][1] < v:
                    Limits[ind][1] = v

                # Count (for statistical purpose).
                Limits[ind][2] = Limits[ind][2] + 1

                if ind not in tests:
                    # Input params: fill X and W arrays.

                    z = idx.index(ind)
                    x[z] = v
                    w[z] = ind

        add_value(gender, 34)
        add_value(age, 35)

        # Basic CBC 5-DIFF
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

        # 3-DIFF if any
        add_value(mid, 36)
        add_value(gra, 37),

        # Additional parameters if any.
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

        # Add values to the dataset.
        X.append(x)
        W.append(w)
        Y.append(y)

    # End of for select cycle.
    with open(analyte_limits_file, "wb") as file:
        pickle.dump(Limits, file)

    # Shuffle the data.
    ids = np.arange(0, len(X), 1, dtype=int)
    np.random.shuffle(ids)

    # 20% for testing.

    train_size = (int)(len(ids) * 0.8)
    test_size = len(ids) - train_size

    x_train = np.zeros((train_size, NN), dtype="float32")
    w_train = np.zeros((train_size, NN), dtype="int32")
    m_train = np.zeros((train_size, NN), dtype="int32")
    y_train = np.zeros((train_size, len(tests)), dtype="float32")

    x_test = np.zeros((test_size, NN), dtype="float32")
    w_test = np.zeros((test_size, NN), dtype="int32")
    m_test = np.zeros((test_size, NN), dtype="int32")
    y_test = np.zeros((test_size, len(tests)), dtype="float32")

    for i in range(train_size):
        for j in range(NN):

            v = X[ids[i]][j]
            w = W[ids[i]][j]

            # No more values here.
            if w == 0:
                break

            y_min, y_max = Limits[w][0], Limits[w][1]

            add = 0.01 * (y_max - y_min)
            y_max = y_max + add
            y_min = y_min - add

            # Scale.
            x_train[i][j] = 0.9 - 0.8 * (y_max - v) / (y_max - y_min)

        y_train[i] = Y[ids[i]]
        w_train[i] = W[ids[i]]
        m_train[i] = W[ids[i]]

        for j in range(NN):
            if m_train[i][j] > 0:
                m_train[i][j] = 1

    # and for test ...
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

        y_test[j] = Y[ids[i]]
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

    return (
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
    )
