"""
This script is a prototype of RSA algorithm. (Only math)

Description: It takes input in form of a single character, generates a public key for it's cipher text and,
using that public key and known "private key" decrypts the message to its original form.

Date: 1st September 2017
Author: Swati Goyal
"""

from math import *
from random import *

# global variable to set value of private key exponent
exp_d = 1


def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)


def is_prime(x):
    """
    :param x: argument whose primality is to be tested
    :return: True or False if x is a prime number
    """
    if x <= 2:
        return True
    elif x % 2 == 0:
        return False

    for n in range(2, (x // 2) + 1):
        if x % n == 0:
            return False
    return True


def prime_generator():
    """
    :return: value of totient calculated from prime number
    """
    primes = []
    while True:
        n = randint(11, 100)
        if is_prime(n):
            if len(primes) < 2:
                primes.append(n)
            else:
                break
        n = randint(11, 100)
    print("Primes generated at random: {}".format(primes))
    return primes[0], primes[1]


def totient_function(p, q):
    """
    :param p: large prime
    :param q: large prime
    :return: value of their Carmichael's totient function
    """
    print("Totient function is : {}".format(lcm(p-1, q-1)))
    return lcm(p-1, q-1)


def get_public_key_exponent(totient):
    """
    :param totient: this is the totient of n = p * q
    :return: value of public key exponent
    """
    for e in range(totient):
        if gcd(e, totient) == 1 and e > 1:
            print("Value of exponent 'e' is calculated as: {}".format(e))
            return e


def get_private_key_exponent(totient, e):
    """
    :param totient: value of the Carmichael's totient function of n
    :param e: value of public key exponent
    :return: value of private key exponent
    """
    for d in range(1, 150000):
        num = d * e % totient
        if num == 1:
            print("Value of exponent 'd' is calculated as: {}".format(d))
            return d


def generate_public_key():
    """
    :return: value of public key, value of modulus, value of public key exponent
    """

    pl = prime_generator()
    n = pl[0] * pl[1]
    print("Value of modulus is: {}".format(n))

    totient_of_n = totient_function(pl[0], pl[1])
    exp_e = get_public_key_exponent(totient_of_n)
    print("Value of public key is: {} and {}".format(n, exp_e))

    global exp_d
    exp_d = get_private_key_exponent(totient_of_n, exp_e)

    return n, exp_e


def get_d():
    """
    :return: value of private key exponent, not exposed to the world!
    """
    d = exp_d
    return d


def decrypt_message(pbk, n):
    """
    :param pbk: encrypted cypher text with public key
    :param n: value of public key parameter 0
    :return: decrypted message
    """
    d = get_d()
    print("Value of message is decrypted as: {}".format(chr(pbk ** d % n)))
    return chr(pbk ** d % n)


def get_ordinance(s):
    """
    This function kinda encodes the character into an integer value, which here we consider as cypher text
    :param s: character of alphabet
    :return: ordinance of s
    """
    return ord(s)


ct = input("Enter the character that needs to be encrypted: ")
pbk_params = generate_public_key()
pbk = get_ordinance(ct) ** pbk_params[1] % pbk_params[0]
decrypt_message(pbk, pbk_params[0])
