import math


def wave(A):
    A.sort()
    for i in range(len(A)-1):
        if i % 2 == 0 and A[i] <= A[i + 1]:
            A[i], A[i + 1] = A[i + 1], A[i]
        elif A[i] >= A[i + 1]:
            A[i], A[i + 1] = A[i + 1], A[i]

    return A


def gcd(a, b):
    while b>0:
        a, b = b, a % b
    return a


# A = [5, 1, 3, 2, 4, 5, 6, 8, 0]
# print(wave(A))

print(math.log(10**84, 3/2))