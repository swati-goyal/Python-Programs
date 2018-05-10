from itertools import combinations


def longest_word(s, dic):
    """
    :param s: string
    :param dic: list of words in any form
    :return: longest substring of S in D
    """

    t = list()
    x = list()
    for i in range(len(s)):
        t.extend(list(combinations(s, i)))

    for item in t:
        x.append(''.join(item))

    y = set(dic)
    x = set(x)
    z = list(x.intersection(y))
    if z:
        z.sort(key=lambda s: len(s))
        print("Longest substring of S in D is: {}".format(z[-1]))
    else:
        print("Dictionary doesn't contain words from the sub-sequence of passed string!")
        w = list(x)
        w.sort(key=lambda s:len(s))
        print("Longest word in the subsequence list is: {}".format(w[-1]))

longest_word('abppplee', {"able", "ale", "apple", "bale", "kangaroo"})