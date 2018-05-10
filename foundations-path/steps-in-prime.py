from math import sqrt


def is_prime(x):
    if x < 2:
        return False
    if x == 2:
        return True

    for n in range(2, int(sqrt(x)) + 1):
        if x % n == 0:
            return False
    return True


def step(g, m, n):
    if n > m:
        for i in range(m, n+1):
            if is_prime(i) and is_prime(i + g):
                return [i, i + g]
    else:
        return []

print(step(200, 10000, 150000))
