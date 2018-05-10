def bubble_sorting(arr):
    n = len(arr)
    for i in range(1, n):
        for j in range(0, i):
            if arr[j] > arr[j+1]:
                arr[j+1], arr[j] = arr[j], arr[j+1]
    return arr

a = [34, 8, 90, 67, 45, 123, 58]
print(bubble_sorting(a))
