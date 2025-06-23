

def resto(a, b):
    return a % b

def maximo_comun_divisor(a, b):
    while b:
        a, b = b, resto(a, b)
    return a

def extended_gcd(a: int, b: int) -> tuple[int,int,int]:
    """
    Retorna (x, y, d) tales que x*a + y*b = d = gcd(a,b).
    """
    if b == 0:
        return 1, 0, a
    x1, y1, d = extended_gcd(b, a % b)
    # x1*b + y1*(a % b) = d
    # pero a % b = a - (a//b)*b
    x = y1
    y = x1 - (a // b) * y1
    return x, y, d

def combinacion_lineal_dcm(a: int, b: int) -> tuple[int,int]:
    """
    Devuelve (x, y) tales que x*a + y*b = gcd(a,b).
    """
    x, y, _ = extended_gcd(a, b)
    return x, y

print(maximo_comun_divisor(27,21))
print(maximo_comun_divisor(41, 35*23))  # Output: 1
