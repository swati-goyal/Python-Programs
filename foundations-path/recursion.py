result = 0


def b_n(k):
    if k == 1:
        return -5

    elif n <= 0:
        return 0

    return b_n(k-1) + 9


def b_2_n(k):
    if k == 1:
        return -5

    elif n <= 0:
        return 0

    return -5 + 9 * (k-1)


def c_n(k):
    if k == 1:
        return 20

    elif n <= 0:
        return 0

    return c_n(k-1) - 17


def d_n(k):
    if k == 1:
        return 2

    elif n <= 0:
        return 0

    return d_n(k-1) + 0.4

n = 10
print(d_n(21))
