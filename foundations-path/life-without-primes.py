from math import sqrt, ceil


def solve(n):
    t = []
    i = 1
    while len(t) != n+1:
        s = str(i)
        l = len(s)
        count = 0
        for y in range(l):
            if s[y] in ('2', '3', '5', '7'):
                count += 1
        if not_prime(i) and count == 0:
            t.append(i)
        i += 1
    return t[n]


def not_prime(x):
    if x < 2:
        return True
    for n in range(2, int(sqrt(x)) + 1):
        if x % n == 0:
            return True
    return False

print(solve(100))