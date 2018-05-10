from collections import Counter, defaultdict, namedtuple

counter = Counter('SwatiGoyal')
d = defaultdict(list)

Point = namedtuple('Point', ['x', 'y'])
ThreePoint = namedtuple('ThreePoint', ['x', 'y', 'z'])


def signature(s):
    """Returns the signature of this string.

    Signature is a string that contains all of the letters in order.

    s: string
    """
    # TODO: rewrite using sorted()
    t = list(s)
    t.sort()
    t = ''.join(t)
    return t


def all_anagrams(filename):
    d = defaultdict(list)
    for line in open(filename):
        word = line.strip().lower()
        t = signature(word)
        d[t].append(word)
    return d

'''
for val, freq in counter.most_common(3):
    print(val, freq)


anagrams = all_anagrams('words.txt')

# for key, values in anagrams.items():
#     print(key, values)

# p = Point(1, 2)


def printall(*args, **kwargs):
    print(args)

# d = dict(x=1, y=1)

# printall(4, 5, 6, d)
# print(Point(**d))
'''

known_coeff = {(1, 1): 1, (1, 0): 1, (0, 1): 0, (0, 0): 1, (2, 1): 2}


def binomial_coeff(n, k):
    """Compute the binomial coefficient "n choose k".
    n: number of trials
    k: number of successes
    returns: int
    """
    tup = (n, k)

    if k == 0:
        return 1

    if n == 0:
        return 0

    if tup in known_coeff:
        return known_coeff[tup]

    res = binomial_coeff(n - 1, k) + binomial_coeff(n - 1, k - 1)
    known_coeff[tup] = res
    return res


print(binomial_coeff(100, 50))

#print(ThreePoint(3, 4, 5))
#print(ThreePoint)