def make_encryption_dict(text):
    encryption_dict = dict()
    y = 1
    c = text
    while c not in encryption_dict.values():
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
        encryption_dict[y] = text
        y += 1
    return encryption_dict


def encrypt(text, n):
    if n <= 0:
        return text
    encrypted_dict = make_encryption_dict(text)
    encrypted_dict[0] = text
    return encrypted_dict[n % (len(encrypted_dict) - 1)]


def decrypt(encrypted_text, n):
    if n <= 0:
        return encrypted_text
    encrypted_dict = make_encryption_dict(encrypted_text)
    l = len(encrypted_dict)
    return encrypted_dict[l - (n % l)]


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

'''
#print(encrypt("#oGg<=NiO~V6Z*#[3Y90", 1))
#print(decrypt("i[GV9=*#O3g60N#o~Y<Z", 17))
#print(decrypt("hsi  etTi sats!", 1))

print(encrypt("This is a test!", 15))


string = '#oGg<=NiO~V6Z*#[3Y90'
string4 = "^a!Zxy|:SpPAnoGD*rQP4jY(opNOC|\\t-^pATn"

encrypted = encrypt(string, 640)
encrypted4 = encrypt(string4, 335)

print(string)
print(encrypted)
print(decrypt(encrypted, 640))

print(string4)
print(encrypted4)
print(decrypt(encrypted4, 335))

'''

