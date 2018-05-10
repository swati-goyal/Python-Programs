def prettyPrint(A):
    size = 2 * A - 1
    array = [[0 for x in range(size)] for y in range(size)]
    for i in range(0, size):
        for j in range(i, size - i):
            array[i][j] = A - i
            array[j][i] = A - i
            array[size - 1 - i][j] = A - i
            array[j][size - 1 - i] = A - i
    for i in range(0, size):
        print(array[i])

prettyPrint(2)
