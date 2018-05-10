from bisect import bisect_left, insort

secret = "whatisup"
triplets = [
  ['t','u','p'],
  ['w','h','i'],
  ['t','s','u'],
  ['a','t','s'],
  ['h','a','p'],
  ['t','i','s'],
  ['w','h','s']
]


def recoverSecret(t):
    i = 0
    c = {}
    while i < len(t):
        d = dict.fromkeys(t[i], 0)
        for j in t[i]:
            d[j] += 1
            # t[i].index(j)
        c.update(d)
        i += 1

    '''s = ''.join(c)
    trip = ''

    for i in range(len(t)):
        for j in range(len(t[i])):
            trip = insort([c.keys()], t[i][j])'''

    return ''.join(c.keys())


'''
i = 0
c = {}
while i < len(t):
    d = dict.fromkeys(t[i],0)
    for j in t[i]:
        d[j] += t[i].index(j)
        # t[i].index(j)
    c.update(d)
    i += 1
return c
'''
#assert recoverSecret(triplets) == secret

print(recoverSecret(triplets))