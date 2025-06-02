import numpy as np
import time
from functools import wraps
import matplotlib.pyplot as plt
import cupy as cp
# 1. Función lógica: cálculo del fractal
def mandelbrot_numpy(C: np.ndarray, max_iter: int) -> np.ndarray:
    """Calcula el conjunto de Mandelbrot para una grilla compleja C usando NumPy."""
    z = np.zeros_like(C, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)
    mask = np.ones(C.shape, dtype=bool)
    for n in range(max_iter):
        z[mask] = z[mask] * z[mask] + C[mask]
        mask = np.logical_and(mask, np.abs(z) <= 2)
        M[mask] = n
        print(f"\rMandelbrot {n}", end="", flush=True)
    return M


def mandelbrot_cupy(C: cp.ndarray, max_iter: int)-> cp.ndarray:
    z = cp.zeros_like(C, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    mask = cp.ones(C.shape, dtype=bool)
    for n in range(max_iter):
        z[mask] = z[mask] * z[mask] + C[mask]
        mask = cp.logical_and(mask, cp.abs(z) <= 2)
        M[mask] = n
        print(f"\rMandelbrot {n}", end="", flush=True)
    return M.get()

def funcion(*x, **kwargs):
    for i in x:
        print(i)

funcion([1, 2, 3, 4, 5], 6, 7, 8, 9, a=10, b=11)

# 2. Decorador que conecta una grilla con una función de cálculo
def aplicar_logica_fractal(funcion_de_calculo, max_iter=100):
    def decorador(funcion_grilla):
        @wraps(funcion_grilla)
        def wrapper(*args, **kwargs):
            C = funcion_grilla(*args, **kwargs)
            resultado = funcion_de_calculo(C, max_iter=max_iter)
            return resultado
        return wrapper
    return decorador

# 3. Grilla decorada con la lógica fractal
@aplicar_logica_fractal(mandelbrot_numpy, max_iter=200)
def generar_grilla_compleja_np(xmin, xmax, ymin, ymax, width, height)-> np.ndarray:
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    return X + 1j * Y

@aplicar_logica_fractal(mandelbrot_cupy, max_iter=200)
def generar_grilla_compleja_cp(xmin, xmax, ymin, ymax, width, height)-> cp.ndarray:
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    return X + 1j * Y

# 4. Uso de la grilla decorada
if __name__ == "__main__":
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 800, 600
    resultado = generar_grilla_compleja_numpy(xmin, xmax, ymin, ymax, width, height)
    plt.imshow(resultado, extent=(xmin, xmax, ymin, ymax), cmap='hot')
    plt.colorbar()
    plt.title("Fractal de Mandelbrot")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.show()
    
