def binary_search(arr, i, j, v):

    if type(v) is not int:
        return 'Be sensible, search only integers!'

    if i == j and (v > arr[j-1] or v < arr[i]):
        return "Number not in the array"

    elif (i + j) >= 1:

        middle = (i + j) // 2

        if arr[middle] == v:
            return "This number {} is at position : {}".format(v, middle)
        elif v > arr[middle]:
            return binary_search(arr, middle + 1, j, v)
        else:
            return binary_search(arr, i, middle, v)


# a = range(0, 15, 1)
# n = len(a)
# print(binary_search(a, 0, n, 3725))

a = range(0, 44, 1)
n = len(a)
print(binary_search(a, 0, n, 7))
