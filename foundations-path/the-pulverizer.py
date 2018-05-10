
def pulverizer(b, n):
    """
    :param b: first number
    :param n: second number
    :return: gcd, coefficient of b and coefficient of n
    """
    x0, x1, y0, y1 = 1, 0, 0, 1
    while n != 0:
        q, b, n = b // n, n, b % n
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return b, x0, y0


def gcd(a, b):
    while b > 0:
        temp = b
        b = a % b
        a = temp
        gcd(a, b)
    return a


print(pulverizer(899, 1147))
# print(gcd(259, 70))


