"""
Author : Swati Goyal
Date : 5th March 2018

Desc : Below is a function that sorts an array of distinct numbers. This doesn't work when there are duplicates in the
        array.
"""

def selection_sort(arr):
    for i in range(0, len(arr)):
        minimum = arr[i]
        for j in range(i, len(arr)):
            if arr[j] <= minimum:
                minimum = arr[j]
        a, b = arr.index(minimum), i
        arr[a], arr[b] = arr[b], arr[a]
    return arr


array = [1, 0, 2, 21, 19, 68, 67, 23, 60, 89, 90, 34, 56, 13, 7]
print(selection_sort(array))