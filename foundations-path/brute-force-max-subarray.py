def brute_max_subarray(arr):
    t = []
    count = 0
    for i in range(len(arr)):
        j = i+1
        while j < len(arr):
            sum1 = arr[i] + arr[j]
            if sum1 >= 0:
                t.append((sum1, i, j))
            else:
                t.append((0, i, j))
            count += 1
            j += 1

    sums = sorted(t)

    return sums[-1], count

A = [-1, -5, -6, -6, -4, 10]
(s, i, j), c = brute_max_subarray(A)

print("Sum: {}, i= {}, j={}, and count: {}".format(s, i, j, c))
