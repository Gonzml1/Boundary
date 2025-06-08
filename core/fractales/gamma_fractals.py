import numpy as np
import cupy as cp
import os
import time
from .base import register_fractal, FractalCalculator
from scipy.special import gamma
@register_fractal("Julia-Gamma", "CPU_Numpy")
def hacer_julia_gamma(self) -> np.ndarray:
    """
    Julia con función Gamma: z_{n+1} = gamma(z_n)^2 + c
    """
    inicio = time.time()

    x = np.linspace(self.xmin, self.xmax, self.width)
    y = np.linspace(self.ymin, self.ymax, self.height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)
    matriz = np.ones(C.shape, dtype=bool)

    c = self.real + 1j*self.imag

    for n in range(self.max_iter):
        # Calculá gamma(z) para los que aún no escaparon
        z_act = z[matriz]
        try:
            z_next = gamma(z_act) ** 2 + c
        except Exception as e:
            # Si gamma falla (por valores no válidos), poné infinito
            z_next = np.full_like(z_act, np.inf, dtype=np.complex128)
        z[matriz] = z_next
        matriz = np.logical_and(matriz, np.abs(z) <= 100)
        M[matriz] = n
        print(f"\rJULIA-GAMMA {n}", end="", flush=True)

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M

@register_fractal("Mandelbrot-Gamma", "CPU_Numpy")
def hacer_mandelbrot_gamma(self) -> np.ndarray:
"""
Mandelbrot con función Gamma: z_{n+1} = gamma(z_n)^2 + c
"""
inicio = time.time()

x = np.linspace(self.xmin, self.xmax, self.width)
y = np.linspace(self.ymin, self.ymax, self.height)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y   # <--- Ahora c depende de cada punto
z = np.zeros(C.shape, dtype=np.complex128)
M = np.zeros(C.shape, dtype=int)
matriz = np.ones(C.shape, dtype=bool)

for n in range(self.max_iter):
    z_act = z[matriz]
    c_act = C[matriz]
    try:
        z_next = gamma(z_act) + c_act
    except Exception as e:
        z_next = np.full_like(z_act, np.inf, dtype=np.complex128)
    z[matriz] = z_next

    matriz = np.logical_and(matriz, np.abs(z) <= 100)  # Mantener solo los que no escaparon
    M[matriz] = n
    print(f"\rMANDELBROT-GAMMA {n}", end="", flush=True)

fin = time.time()
print("\nTiempo de ejecución:", fin - inicio, "segundos")
return M
