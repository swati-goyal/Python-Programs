from math import gcd, log


def cpFact(A, B):
    arr = []
    if gcd(A, B) == 1:
        return A
    for x in range(A):
        if gcd(x, B) == 1 and A % x == 0:
            arr.append(x)
    return max(arr)

# print(cpFact(30, 12))
