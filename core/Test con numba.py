import numpy as np
import numba
import time
import matplotlib.pyplot as plt
from functools import wraps
from numba import prange

# Parámetros del render
width, height = 1000, 1000
max_iter = 256
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5

x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5


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

@medir_tiempo("Mandelbrot con Numba")
@numba.njit(parallel=True)
def mandelbrot_numba(width, height, xmin, xmax, ymin, ymax, max_iter):
    image = np.zeros((height, width), dtype=np.uint8)
    for y in numba.prange(height):
        for x in range(width):
            real = x_min + (x / width) * (x_max - x_min)
            imag = y_min + (y / height) * (y_max - y_min)
            c = complex(real, imag)
            z = 0j
            count = 0
            while abs(z) <= 2 and count < max_iter:
                z = z*z + c
                count += 1
            color = int(255 * count / max_iter)
            image[y, x] = color
    return image

@medir_tiempo("Mandelbrot con Numba numpy")
@numba.njit(parallel=True)
def mandelbrot_numpy(width, height, xmin, xmax, ymin, ymax, max_iter):

    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=np.int32)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        M[mask] += 1

    return M

def crear_plano_complejo(xmin, xmax, ymin, ymax, width, height):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    return X + 1j * Y  # Plano complejo C

@numba.njit(parallel=True)
def calcular_mandelbrot(C, max_iter):
    height, width = C.shape
    M = np.zeros((height, width), dtype=np.int32)

    for y in numba.prange(height):
        for x in range(width):
            c = C[y, x]
            z = 0j
            count = 0
            while abs(z) <= 2 and count < max_iter:
                z = z*z + c
                count += 1
            M[y, x] = count
    return M

@medir_tiempo("Mandelbrot (separado)")
def ejecutar():
    C = crear_plano_complejo(-2.0, 1.0, -1.5, 1.5, 800, 800)
    return calcular_mandelbrot(C, max_iter=100)

M = ejecutar()

def medir_tiempo(etiqueta):
    def decorador(func):
        def wrapper(*args, **kwargs):
            inicio = time.time()
            resultado = func(*args, **kwargs)
            fin = time.time()
            print(f"{etiqueta}: {fin - inicio:.3f} s")
            return resultado
        return wrapper
    return decorador

@numba.njit(parallel=True)
def mandelbrot_fast(width, height, xmin, xmax, ymin, ymax, max_iter):
    result = np.zeros((height, width), dtype=np.uint16)
    for y in prange(height):
        imag = ymin + y * (ymax - ymin) / height
        for x in range(width):
            real = xmin + x * (xmax - xmin) / width
            c = complex(real, imag)
            z = 0j
            count = 0
            while (z.real*z.real + z.imag*z.imag <= 4.0) and count < max_iter:
                z = z*z + c
                count += 1
            result[y, x] = count
    return result

@medir_tiempo("Mandelbrot (separado, rejilla + cálculo)")
def ejecutar():
    return mandelbrot_fast(1000, 1000, -2.0, 1.0, -1.5, 1.5, 1000)



plt.imshow(M, cmap='inferno', extent=(-2, 1, -1.5, 1.5))
plt.title("Mandelbrot separado (rejilla + cálculo)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()