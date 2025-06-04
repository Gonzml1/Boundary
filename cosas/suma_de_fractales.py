import numpy as np
import matplotlib.pyplot as plt

def fractal_escape_matrix(formula, xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C, dtype=np.complex128)
    escape = np.zeros(C.shape, dtype=np.int32)

    for i in range(max_iter):
        Z = formula(Z, C)
        mask = (np.abs(Z) > 2) & (escape == 0)
        escape[mask] = i

    escape[escape == 0] = max_iter
    return escape

# Fórmulas para z^2 + c, z^3 + c, z^4 + c
def mandelbrot(Z, C):
    return Z**2 + C

def mandelbrot_cubic(Z, C):
    return Z**3 + C

def mandelbrot_quartic(Z, C):
    return Z**4 + C



def fractal_serie_geometrica(xmin, xmax, ymin, ymax, width, height, max_iter, max_potencia):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C, dtype=np.complex128)
    escape = np.zeros(C.shape, dtype=np.int32)

    for i in range(max_iter):
        Z_next = np.zeros_like(Z, dtype=np.complex128)
        for k in range(1, max_potencia + 1):
            Z_next += np.power(Z, k) + np.power(C, k)

        Z = Z_next
        mask = (np.abs(Z) > 4) & (escape == 0)
        escape[mask] = i

    escape[escape == 0] = max_iter
    return escape

def fractal_serie_inversos(xmin, xmax, ymin, ymax, width, height, max_iter, max_potencia, epsilon=1e-8):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C, dtype=np.complex128)
    escape = np.zeros(C.shape, dtype=np.int32)

    for i in range(max_iter):
        Z_next = np.zeros_like(Z, dtype=np.complex128)
        for k in range(1, max_potencia + 1):
            # Usamos el inverso de Z^k y C^k, con epsilon para evitar divisiones por cero
            Z_k = np.power(Z, k)
            C_k = np.power(C, k)
            Z_next += 1 / (Z_k + epsilon) + 1 / (C_k + epsilon)

        Z = Z_next
        mask = (np.abs(Z) > 4) & (escape == 0)
        escape[mask] = i

    escape[escape == 0] = max_iter
    return escape

import numpy as np

def fractal_serie_mixta(xmin, xmax, ymin, ymax, width, height, max_iter, max_potencia, epsilon=1e-8):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C, dtype=np.complex128)
    escape = np.zeros(C.shape, dtype=np.int32)

    for i in range(max_iter):
        Z_next = np.zeros_like(Z, dtype=np.complex128)
        for k in range(1, max_potencia + 1):
            Z_k = np.power(Z, k)
            C_k = np.power(C, k)
            # Serie mixta: suma directa + inversa
            Z_next += Z_k + 1 / (Z_k + epsilon) + C_k + 1 / (C_k + epsilon)

        Z = Z_next
        mask = (np.abs(Z) > 4) & (escape == 0)
        escape[mask] = i

    escape[escape == 0] = max_iter
    return escape

xmin, xmax = -2, 2
ymin, ymax = -2, 2
width, height = 800, 800
max_iter = 50
max_potencia = 6

imagen = fractal_serie_mixta(xmin, xmax, ymin, ymax, width, height, max_iter, max_potencia)

plt.figure(figsize=(10, 10))
plt.imshow(imagen, extent=[xmin, xmax, ymin, ymax], cmap='cubehelix')
plt.title(f"Fractal mixto: potencias directas + inversas hasta k={max_potencia}")
plt.axis('off')
plt.tight_layout()
plt.show()