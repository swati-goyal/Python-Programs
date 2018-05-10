from itertools import permutations


def next_bigger(n):
    s = str(n)
    l = len(s)
    if l == 1:
        return n

    '''
    if s[l-1:] > s[l-2:l-1]:
        new_s = s[:l-2] + s[l-1:] + s[l-2:l-1]
        return new_s
    elif s[l-1:] == s[l-2:l-1]:
        return -1
    '''

    #if int(new_s) > n:
    #    return int(new_s)

    arr = set(list(permutations(s, len(s))))

    arr2, arr3 = [], []

    for i in arr:
        arr2.append(''.join(i))

    arr3 = sorted(list(map(lambda x: int(x), arr2)))
    return arr3[arr3.index(n) + 1] if arr3.index(n) != len(arr3) - 1 else -1


print(next_bigger(21414780))