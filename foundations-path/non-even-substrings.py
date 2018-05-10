def solve(s):
    length = len(s)
    t = [s[i: j] for i in range(length) for j in range(i + 1, length + 1)]
    x = []
    for i in t:
        if i is not '':
            q = int(i)
            if q % 2 != 0:
                x.append(q)
    return len(x)


def solve_two(s):
    return sum(i+1 for i, d in enumerate(list(s)) if d in '13579')

print(solve_two('1357'))

'''

'''