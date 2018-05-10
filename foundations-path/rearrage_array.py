# below uses extra space
def arrange(A):
    arr = []
    i = 0
    while i < len(A):
        j = A[i]
        arr.append(A[j])
        i += 1
    return arr


# below uses less space and applies index logic to manipulate given array
def re_arrange(A):
    i = 0
    length = len(A)
    while i < length:
        A[i] += ((A[A[i]]) % length) * length
        i += 1
    i = 0
    while i < length:
        A[i] /= length
        i += 1

print(arrange([ 4, 0, 2, 1, 3 ]))