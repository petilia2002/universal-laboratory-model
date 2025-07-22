import numpy as np

# enumerate all variants up to N
# imitating addition in binary system, but from the left

# 1 0 0
# 0 1 0
# 1 1 0
# 0 0 1
# 1 0 1

# etc


def get_variants(n):

    v = []

    a = [0 for i in range(n)]
    a[0] = 1

    v.append(a[:])
    while True:

        i = -1
        for j in range(n):

            if a[j] == 1:
                a[j] = 0
                i = j + 1
            else:
                if i == -1:
                    i = j
                a[i] = 1

                v.append(a[:])
                break
        if i >= n:
            break
    return v


# Preparation augmented dataset.


def prepareData(start, end, ids, X, W, Y, source, tests, x_map, y_map, mask_value):

    # input
    x_d = []
    w_d = []
    m_d = []

    # output
    y_d = []
    m_pt = []
    r_d = []

    for i in range(start, end):

        a = [0] * len(source)
        for j in range(len(source)):

            v = X[ids[i]][j]
            w = W[ids[i]][j]

            if w == 0:
                break
            a[j] = source[x_map[w]].scale(v)

        y_t = Y[ids[i]]
        w_t = W[ids[i]]

        words = []
        for j in range(len(tests)):
            if y_t[j] != mask_value:
                words.append(tests[j].getId())

        # augment input
        for p in get_variants(len(words)):

            b = [0] * len(tests)
            c = [mask_value] * len(tests)
            e = [0] * len(tests)

            ind = 0
            dels = []

            for j in range(len(p)):
                if p[j] == 1:
                    b[ind] = words[j]
                    c[ind] = y_t[y_map[words[j]]]

                    ind_source = w_t.index(words[j])
                    e[ind] = source[x_map[words[j]]].scale(X[ids[i]][ind_source])

                    ind = ind + 1
                    dels.append(words[j])

            t = [0] * len(source)
            m = [0] * len(source)

            w = w_t[:]

            for d in dels:
                w[w.index(d)] = 0

            ind = 0
            for j in range(len(source)):
                if w[j] != 0:
                    t[ind] = a[j]
                    m[ind] = w[j]
                    ind = ind + 1

            m_t = [0] * len(source)
            for j in range(len(source)):
                if m[j] > 0:
                    m_t[j] = 1

            x_d.append(t)
            w_d.append(m)
            m_d.append(m_t)

            y_d.append(c)
            m_pt.append(b)
            r_d.append(e)

    x_d = np.asarray(x_d, dtype="float32")
    w_d = np.asarray(w_d, dtype="int8")
    m_d = np.asarray(m_d, dtype="int8")

    y_d = np.asarray(y_d, dtype="int8")
    m_pt = np.asarray(m_pt, dtype="int8")
    r_d = np.asarray(r_d, dtype="float32")

    return x_d, w_d, m_d, y_d, m_pt, r_d
