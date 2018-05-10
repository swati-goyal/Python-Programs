def cp():
    # 24-hour clock hh:mm, return all times which can be palindromes
    hour = range(0, 24)
    minute = range(0, 60)
    t = []
    x = []

    for h in hour:
        hs = str(h)
        if len(hs) == 1:
            hs = str(h).zfill(2)
        for m in minute:
            ms = str(m)
            if len(ms) == 1:
                ms = str(m).zfill(2)
            t.append(hs + ':' + ms)
    for item in t:
        rev = item[::-1]
        if item == rev and rev in t:
            x.append(item)

    return x

print(cp())
