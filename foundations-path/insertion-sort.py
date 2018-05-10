def binary_search(arr, i, j, v):
    if i == j:
        return i

    middle = i + ((j-i)//2)

    if v > arr[middle]:
        return binary_search(arr, middle + 1, j, v)
    elif v < arr[middle]:
        return binary_search(arr, i, middle, v)
    else:
        return middle


def binary_insertion_sort_incr(arr):
    for i in range(1, len(arr)):
        ind = binary_search(arr, 0, i, arr[i])
        if ind < i:
            tmp = arr[i]
            j = i-1
            while j >= ind:
                arr[j + 1] = arr[j]
                j -= 1
            arr[ind] = tmp
    return arr


def insertion_sort_incr(arr):
    count = 0
    for j in range(1, len(arr)):
        key = arr[j]
        i = j - 1
        while i >= 0 and arr[i] > key:
            arr[i + 1] = arr[i]
            i -= 1
            count += 1
        arr[i + 1] = key
    return arr, count


def insertion_sort_decr(arr):
    count = 0
    for j in range(1, len(arr)):
        key = arr[j]
        i = j-1
        while i >= 0 and arr[i] < key:
            arr[i+1] = arr[i]
            i -= 1
            count += 1
        arr[i+1] = key
    return arr, count


def insertion_sort_recursive(arr, n):
    if n <= 1:
        return arr

    insertion_sort_recursive(arr, n - 1)

    last = arr[n - 1]
    j = n - 2

    # sorting of sub-arrays of array of size n
    while j >= 0 and arr[j] > last:
        arr[j + 1] = arr[j]
        j -= 1
    arr[j + 1] = last


# print("Insertion sort increasing order (array in reverse order - worst case): ", insertion_sort_incr([59, 58, 41, 41, 31, 26]))
# print("Insertion sort increasing order (array in 2 pairs are sorted order - average case): ", insertion_sort_incr([31, 26, 42, 41, 59, 58]))
# print("Insertion sort increasing order (array in sorted order - best case): ", insertion_sort_incr([26, 31, 41, 41, 58, 59]))
# print("Insertion sort decreasing order (array in sorted order - best case): ", insertion_sort_decr([59, 58, 41, 41, 31, 26]))
# print("Insertion sort decreasing order (array in reverse order - worst case): ", insertion_sort_decr([26, 31, 41, 41, 58, 59]))

if __name__ == '__main__':
    arr = [34, 8, 90, 67, 45, 123, 58]
    print(insertion_sort_incr(arr))

# for i in range(1, -21, -1):
#     print(i)
