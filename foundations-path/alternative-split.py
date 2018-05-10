def encrypt_two(text, n):
    encrypted = text
    if n <= 0:
        return encrypted
    else:
        encrypted_text = text[1::2] + text[::2]
        k = n-1
        while k >= 1:
            encrypted_text = encrypt_two(encrypted_text, k)
            k -= n
        return encrypted_text


def encrypt(text, n):
    if n <= 0:
        return text

    else:
        while n > 0:
            desired = ''
            remaining = ''
            i = 0
            while i < len(text):
                if i % 2 == 0:
                    remaining += text[i]
                else:
                    desired += text[i]
                i += 1
            text = desired + remaining
            n -= 1
        return text


def decrypt_2(encrypted_text, n):
    if n <= 0:
        return encrypted_text
    x = len(encrypted_text)
    k = (x // 2) + 1
    if x % 2 != 0:
        return encrypt(encrypted_text, k - n)
    else:
        return encrypt(encrypted_text, k - 1 - n)


def decrypt(encrypted_text, n):
    if n <= 0:
        return encrypted_text
    x = len(encrypted_text)
    l = x/2

    while n >= 0:
        if n <= x:
            return encrypt(encrypted_text, x - n)
        else:
            n //= l

'''
print(decrypt("hsi  etTi sats!", 1))
print(decrypt("s eT ashi tist!", 2))
print(decrypt(" Tah itse sits!", 3))
print(decrypt("This is a test!", 4))
print(decrypt("This is a test!", 8))
print(decrypt("This is a test!", -1))
print(decrypt("hskt svr neetn!Ti aai eyitrsig", 1))


print(encrypt("This is a test!", 1))
print(encrypt("This is a test!", 2))
print(encrypt("This is a test!", 3))
print(encrypt("This is a test!", 4))
print(encrypt("This is a test!", 5))
'''

string = '#oGg<=NiO~V6Z*#[3Y90'
string2 = "iZRSSey7H5be"
string3 = "voY'q_<QN:7~"
string4 = "^a!Zxy|:SpPAnoGD*rQP4jY(opNOC|\\t-^pATn"

encrypted = encrypt(string, 640)
encrypted2 = encrypt(string2, 105)
encrypted3 = encrypt(string3, 183)
encrypted4 = encrypt(string4, 335)

print(string)
print(encrypted)
print(decrypt(encrypted, 640))

print(string2)
print(encrypted2)
print(decrypt(encrypted2, 105))

print(string3)
print(encrypted3)
print(decrypt(encrypted3, 183))

print(string4)
print(encrypted4)
print(decrypt(encrypted4, 335))


