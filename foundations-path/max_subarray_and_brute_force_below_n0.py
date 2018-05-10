count = 0


def find_max_crossing_sub_array(arr, low, mid, high):
    left_sum = 0
    sum1 = 0
    max_left = low
    max_right = high

    for i in range(mid, low + 1, -1):
        sum1 += arr[i]
        if sum1 > left_sum:
            left_sum = sum1
            max_left = i

    right_sum = 0
    sum2 = 0
    for j in range(mid + 1, high):
        sum2 += arr[j]
        if sum2 > right_sum:
            right_sum = sum2
            max_right = j

    return max_left, max_right, left_sum + right_sum


def brute_max_subarray(arr):
    t = []
    global count
    for i in range(len(arr)):
        j = i+1
        while j < len(arr):
            t.append((arr[i] + arr[j], i, j))
            count += 1
            j += 1

    sums = sorted(t)

    return sums[-1][1], sums[-1][2], sums[-1][0]


def find_maximum_sub_array(arr, low, high):
    global count

    if low < high <= 5:
        return brute_max_subarray(arr)

    else:
        if low == high:
            count += 1
            return low, high, arr[low]

        else:
            mid = (low+high)//2
            left_low, left_high, left_sum = find_maximum_sub_array(arr, low, mid)
            right_low, right_high, right_sum = find_maximum_sub_array(arr, mid+1, high)
            cross_low, cross_high, cross_sum = find_max_crossing_sub_array(arr, low, mid, high)

            if left_sum > right_sum and left_sum > cross_sum:
                count += 1
                return left_low, left_high, left_sum

            elif right_sum > left_sum and right_sum > cross_sum:
                count += 1
                return right_low, right_high, right_sum

            else:
                count += 1
                return cross_low, cross_high, cross_sum


A = [-1, -5, -6, -6, -4, 10]
(s, i, j), c = find_maximum_sub_array(A, 0, len(A)-1), count
print("i={}, j={}, Sum={}, and count: {}".format(s, i, j, c))
