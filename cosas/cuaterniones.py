import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm
from functools import wraps

def medir_tiempo(nombre):
    def decorador(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            inicio = time.time()
            resultado = func(*args, **kwargs)
            fin = time.time()
            print(f"⏱️ Tiempo de ejecución de '{nombre}': {fin - inicio:.5f} segundos")
            return resultado
        return wrapper
    return decorador

class Cuaternion:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __str__(self):
        return f"{self.a} + {self.b}i + {self.c}j + {self.d}k"

    def __add__(self, other):
        return Cuaternion(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)

    def __sub__(self, other):
        return Cuaternion(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)

    def __mul__(self, other):
        a = self.a * other.a - self.b * other.b - self.c * other.c - self.d * other.d
        b = self.a * other.b + self.b * other.a + self.c * other.d - self.d * other.c
        c = self.a * other.c - self.b * other.d + self.c * other.a + self.d * other.b
        d = self.a * other.d + self.b * other.c - self.c * other.b + self.d * other.a
        return Cuaternion(a, b, c, d)
    
    def square(self):
        return self * self
    def modulo(self):
        return (self.a**2 + self.b**2 + self.c**2 + self.d**2)**0.5
    
    def __abs__(self):
        return self.modulo()
    
    def __str__(self):
        return f"({self.a} + {self.b}i + {self.c}j + {self.d}k)"
    
def calcular_mandelbrot_cuaternion(res=500, lim=2.0, max_iter=30):
    imagen = np.zeros((res, res), dtype=np.uint8)
    
    for i in range(res):
        for j in range(res):
            a = (i / res) * (2 * lim) - lim  
            b = (j / res) * (2 * lim) - lim  
            c = Cuaternion(a, b, 0, 0)
            z = Cuaternion(0, 0, 0, 0)
            iteraciones = 0

            while abs(z) <= 4 and iteraciones < max_iter:
                z = z.square() + c
                iteraciones += 1

            imagen[j, i] = int(255 * iteraciones / max_iter)

    return imagen
@medir_tiempo("Generar imagen Mandelbrot Cuaterniónico")
def generar_mandelbrot_3d(res=500, lim=1.5, max_iter=30):
    puntos = []

    for x in range(res):
        for y in range(res):
            for z in range(res):
                a = (x / res) * (2 * lim) - lim
                b = (y / res) * (2 * lim) - lim
                c = (z / res) * (2 * lim) - lim
                d = 0  # Fijamos d = 0 para visualizar en 3D

                c_q = Cuaternion(a, b, c, d)
                q = Cuaternion(0, 0, 0, 0)
                i = 0
                print(x,y,z)
                while abs(q) <= 4 and i < max_iter:
                    q = q.square() + c_q
                    i += 1

                if i == max_iter:
                    puntos.append((a, b, c))  # Solo guardamos puntos que no escapan

    return puntos

@medir_tiempo("Generar imagen Mandelbrot Cuaterniónico con Numpy")
def generar_mandelbrot_3d_numpy(res=100, lim=1.5, max_iter=30):
    puntos = []

    lin = np.linspace(-lim, lim, res)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
    
    X = X.ravel()
    Y = Y.ravel()
    Z = Z.ravel()

    for a, b, c in tqdm(zip(X, Y, Z), total=len(X)):
        d = 0  # Fijamos dimensión d = 0
        c_q = Cuaternion(a, b, c, d)
        q = Cuaternion(0, 0, 0, 0)
        i = 0
        while abs(q) <= 4 and i < max_iter:
            q = q.square() + c_q
            i += 1
        if i == max_iter:
            puntos.append((a, b, c))

    return puntos

def visualizar_puntos_3d(puntos):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = zip(*puntos)
    ax.scatter(xs, ys, zs, s=1, c='black', alpha=0.8)
    ax.set_title("Conjunto de Mandelbrot Cuaterniónico en 3D")
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')
    plt.tight_layout()
    plt.show()
    
puntos = generar_mandelbrot_3d_numpy(res=200, lim=1.5, max_iter=30)
visualizar_puntos_3d(puntos)