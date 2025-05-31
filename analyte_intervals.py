from configparser import ConfigParser
import fdb
import numpy as np

from scipy.stats import gaussian_kde
from scipy import optimize


def find_min_max(x):

    P = 0.9999

    x = sorted(x)
    st = {}

    for i in x:
        if i in st:
            st[i] = st[i] + 1
        else:
            st[i] = 1

    idx = [i for i in st]

    m = st[idx[0]]
    im = 0

    for i in range(1, len(idx)):
        if st[idx[i]] > m:
            m = st[idx[i]]
            im = i

    x1 = x2 = idx[im]
    p = 0.0

    l = r = -1
    for i in range(len(idx)):
        if idx[i] >= x1 and l == -1:
            l = i - 1
        if idx[i] >= x2 and r == -1:
            r = i + 1
            break

    p = float(st[idx[im]])

    while p / len(x) <= P and (r < len(idx) or l >= 0):

        left_area = 0
        right_area = 0

        if l >= 0:
            a = idx[l]
            left_area = st[a]

        if r < len(idx):
            b = idx[r]
            right_area = st[b]

        if left_area > right_area:
            x1 = a
            p += left_area
            l -= 1
        elif left_area == right_area:
            x1, x2 = a, b
            p = p + left_area + right_area
            l -= 1
            r += 1
        else:
            x2 = b
            p += right_area
            r += 1

    return x1, x2, idx[im]


def applicabilityDomain():

    config_object = ConfigParser()
    config_object.read("./config_file.ini")

    user_info = config_object["USERINFO"]
    server_config = config_object["SERVERCONFIG"]

    dsn = server_config["dsn"]
    user = user_info["user"]
    password = user_info["password"]

    ml = fdb.connect(dsn=dsn, user=user, password=password)

    analytes = []

    for aid, analyte, view_name in ml.cursor().execute(
        "select id, analyte, view_name from analytes where view_name is not null"
    ):
        analytes.append([aid, analyte, view_name])

    for analyte in analytes:

        X = []

        for (x,) in ml.cursor().execute(
            "select " + analyte[2] + " from lab  where " + analyte[2] + "  > 0 "
        ):
            X.append(float(x))

        print(analyte[1], len(X))

        ints = find_min_max(X)
        if ints and len(ints) == 3:
            print(ints)

            x1 = ints[0]
            x2 = ints[1]

            ml.cursor().execute(
                "update analytes set min_val = ?, max_val = ? where id = ?",
                (
                    x1,
                    x2,
                    analyte[0],
                ),
            )

    ml.commit()


# applicabilityDomain();
