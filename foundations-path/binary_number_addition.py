# Two array of bits

a = "1010110111001101101000"
b = "1000011011000000111100110"

n = len(a)
m = len(b)
diff = abs(n - m)

if n > m:
    b.zfill(n)
else:
    a.zfill(m)

ao = [int(i) for i in a]
bo = [int(i) for i in b]
# Initialization

carry = 0
c = []
i = 0

# reverse array for addition
a = ao[::-1]
b = bo[::-1]

# loop invariant
while i < len(a):
    if (a[i] == 1 and b[i] == 0) or (a[i] == 0 and b[i] == 1):
        if carry == 0:
            c.append(1)
        else:
            c.append(0)
            carry = 1
    elif a[i] == 1 and b[i] == 1:
        if carry == 0:
            c.append(0)
            carry = 1
        else:
            c.append(1)
            carry = 1
    else:
        c.append(carry)

    i += 1

# Put carry into the array as MSB
c.append(carry)

# Print the output
print(''.join(str(i) for i in c[::-1]))
#print(''.join(str(i) for i in a).zfill(len(b)))
#print(''.join(str(i) for i in b).zfill(len(a)))
