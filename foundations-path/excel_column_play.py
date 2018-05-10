def titleToNumber(self, A):
    if A.isalpha() is False:
        return 0
    else:
        sum1 = 0
        i = 0
        rtitle = A.upper()[::-1]
        while i < len(rtitle):
            sum1 += (ord(rtitle[i]) - 64) * (26 ** i)
            i += 1
        return sum1


def convertToTitle(A):
    title = []
    if type(A) is not int:
        return title
    else:
        while A > 0:
            if A % 26 == 0:
                title.append(chr(25+ord('A')))
                A = (A//26)-1
            else:
                title.append(chr((A % 26) + ord('A') - 1))
                A //= 26

        return ''.join(title[::-1])

print(convertToTitle(52))