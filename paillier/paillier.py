"""
"""
import math, random, sys, time
from gmpy2 import mpz, powmod, invert, is_prime, random_state, mpz_urandomb, rint_round, log2, gcd 
rand = random_state(random.randrange(sys.maxsize))

class PrivateKey(object):
    def __init__(self, p, q, n):
        if p==q:
            self.l = p * (p-1)
        else:
            self.l = (p-1) * (q-1)
        try:
            self.m = invert(self.l, n)
        except ZeroDivisionError as e:
            print(e)
            exit()

class PublicKey(object):
    def __init__(self, n):
        self.n = n
        self.n_sq = n * n
        self.g = n + 1
        self.bits=mpz(rint_round(log2(self.n)))

def generate_prime(bits):    
    """Will generate an integer of b bits that is prime using the gmpy2 library  """    
    while True:
        possible =  mpz(2)**(bits-1) + mpz_urandomb(rand, bits-1)
        if is_prime(possible):
            return possible

def generate_keypair(bits):
    """ Will generate a pair of paillier keys bits>5"""
    p = generate_prime(bits // 2)
    #print(p)
    q = generate_prime(bits // 2)
    #print(q)
    n = p * q
    return PrivateKey(p, q, n), PublicKey(n)

def enc(pub, plain):
    while True:
        r = mpz_urandomb(rand, pub.bits)
        if r < pub.n and r > 0 and gcd(r, pub.n) == 1:
            break
    cipher = (powmod(pub.g, plain, pub.n_sq) * powmod(r, pub.n, pub.n_sq)) % pub.n_sq
    return cipher

def dec(priv, pub, cipher):
    x = powmod(cipher, priv.l, pub.n_sq)
    plain = (((x - 1) // pub.n) * priv.m) % pub.n
    return plain

def enc_add(pub, m1, m2):
    """Add one encrypted integer to another"""
    add_result = m1 * m2 % pub.n_sq
    return add_result

def enc_add_const(pub, m, c):
    """Add constant n to an encrypted integer"""
    # Similiar to enc add
    add_const_result = m * powmod(pub.g, c, pub.n_sq) % pub.n_sq
    return add_const_result

def enc_mul_const(pub, m, c):
    """Multiplies an encrypted integer by a constant"""
    mul_result = powmod(m, c, pub.n_sq)
    return mul_result 

def test(priv, pub):
    # Get boundaries for integers
    lower_bound = 2 ** 10
    upper_bound = 2 ** 1000

    # Get time
    start_time = time.time()

    # Test

    # Set numbers
    x = random.randint(lower_bound, upper_bound)
    y = random.randint(lower_bound, upper_bound)
    s = random.randint(lower_bound, upper_bound)
    const_c = random.randint(lower_bound, upper_bound)
    const_d = 3
    crypted_x = enc(pub, x)
    crypted_y = enc(pub, y)
    crypted_s = enc(pub, s)

    # test add const 
    crypted_u = enc_add_const(pub, crypted_x, const_c)
    u = dec(priv, pub, crypted_u)
    # end_time_add_const = time.time()
    if u == x + const_c:
        print('test add const succeed')
    else:
        print('test add const failed')
    # print('Finished in ' + str(end_time_add_const - start_time) +  '  milliseconds')

    # test add
    crypted_z = enc_add(pub, crypted_x, crypted_y)
    z = dec(priv, pub, crypted_z)
    # end_time_add = time.time()
    if z == x + y:
        print('test add succeed')
    else:
        print('test add failed')
    # print('Finished in ' + str(end_time_add - start_time) +  '  milliseconds')

    # test mul const
    crypted_v = enc_mul_const(pub, crypted_s, const_d)
    v = dec(priv, pub, crypted_v)
    # end_time_mul = time.time()
    if v == s * const_d:
        print('test mul const succeed')
    else:
        print('test mul const failed')
    # print('Finished in ' + str(end_time_mul - start_time) +  '  milliseconds')

    return


if __name__ == '__main__':
    priv, pub = generate_keypair(1024)
    test(priv, pub)

