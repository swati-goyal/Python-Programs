def mysterio(n):
    if n <= 1:
        return 1
    elif n % 2 == 0:
        return mysterio(n/2)
    else:
        return mysterio(3 * n + 1)

print(mysterio(1325245758525))
