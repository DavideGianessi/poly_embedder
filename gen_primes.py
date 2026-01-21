from Crypto.Util.number import isPrime
from sympy import primitive_root

def check_prime_with_roots_of_unity(p: int, n: int):
    """
    Check if p is prime and if F_p contains primitive 2^n-th roots of unity.
    Returns (False, None) or (True, primitive_root).
    """
    if not isPrime(p):
        return False, None
    
    m = 1 << n
    
    if (p - 1) % m != 0:
        return False, None
    
    # Get primitive root of F_p using sympy
    g = primitive_root(p)
    
    # g^((p-1)/m) is a primitive m-th root
    root = pow(g, (p - 1) // m, p)

    roots = [root]

    for i in range(n):
        roots.append(pow(roots[-1],2,p))
    roots.reverse()
    
    return True, roots

def find_primes_with_roots(start: int, n: int, count: int):
    """
    Find primes going DOWN from start until we find count primes
    that have primitive 2^n-th roots of unity.
    
    Args:
        start: Starting number (search goes down from here)
        n: We want primes with primitive 2^n-th roots
        count: Number of primes to find
    
    Returns:
        List of tuples: [(prime1, roots1), (prime2, roots2), ...]
    """
    results = []
    step = 1<<n
    current = start - ((start - 1)% step)
    assert((current -1) % step == 0)
    
    while len(results) < count:
        if current<=0:
            assert(False)
        result = check_prime_with_roots_of_unity(current, n)
        if result[0]:  # If prime with roots exists
            prime = current
            roots = result[1]
            results.append((prime, roots))
            print(f"Found prime {prime} with primitive 2^{n}-th roots")
        current -= step
    
    return results

results=find_primes_with_roots(2**64,24,2)
tot=1;
for r in results:
    tot*=r[0]
print(f"crt coverage: {tot/2**128}")
print(results)
