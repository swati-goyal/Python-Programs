def spiral_order(A):
    a1 = 0
    a2 = 0
    b1 = len(A) - 1
    b2 = len(A[0]) - 1
    res = []
    direction = 0

    while a1 <= b1 and a2 <= b2:

        if direction == 0:
            for i in list(range(a2, b2 + 1)):
                res.append(A[a1][i])
            direction = 1
            a1 += 1

        elif direction == 1:
            for i in list(range(a1, b1 + 1)):
                res.append(A[i][b2])
            direction = 2
            b2 -= 1

        elif direction == 2:
            for i in list(reversed(range(a2, b2 + 1))):
                res.append(A[b1][i])
            direction = 3
            b1 -= 1

        elif direction == 3:
            for i in list(reversed(range(a1, b1 + 1))):
                res.append(A[i][a2])
            direction = 0
            a2 += 1

    return res


arr = [
    [1, 2, 3, 9],
    [4, 5, 6, 6],
    [7, 8, 9, 8],
    [10, 14, 3, 2],
    [1, 4, 7, 10]
    ]
print(spiral_order(arr))