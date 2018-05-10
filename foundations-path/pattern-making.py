def pattern2(n):
    t = [make_pattern(n)]
    k = 1
    while k < n:
        t.append(t[0][:len(t[0])-len(make_pattern(k))])
        k += 1
    return t


def make_pattern(x):
    s = ''
    while x > 0:
        s += str(x)
        x -= 1
    return s


def pattern(n):
    return "\n".join(["".join([str(y) for y in range(n, x, -1)]) for x in range(n)])




print(pattern(69))