import time
from functools import wraps
import ctypes
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Añadir ruta raíz al path para poder importar 'core'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.modulo_de_calculo_fractales import calculos_mandelbrot

def medir_tiempo(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fin = time.time()
        print(f"⏱️ Tiempo de ejecución de '{func.__name__}': {fin - inicio:.5f} segundos")
        return resultado
    return wrapper

# Verificar si la DLL existe
dll_path = r"V:\ABoundary\Testeos\funciones.dll"
if not os.path.exists(dll_path):
    print(f"Error: No se encuentra la DLL en {dll_path}")
    exit(1)

# Cargar la DLL con manejo de errores
try:
    lib = ctypes.WinDLL(dll_path)
except OSError as e:
    print(f"Error al cargar la DLL: {e}")
    print("Verifica que la DLL y sus dependencias estén en el directorio o en el PATH.")
    raise

# Declarar tipos para las funciones

def mostrar_fractal(matriz, xmin, xmax, ymin, ymax, cmap='hot', titulo='Fractal de Mandelbrot'):
    """
    Muestra un fractal en pantalla a partir de una matriz de iteraciones.

    Parámetros:
    - matriz: np.ndarray, matriz 2D con los valores de iteración.
    - xmin, xmax, ymin, ymax: límites del plano complejo.
    - cmap: string, mapa de colores para la visualización.
    - titulo: string, título de la imagen.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(matriz, extent=[xmin, xmax, ymin, ymax], cmap=cmap, origin='lower')
    plt.colorbar(label='Número de iteraciones')
    plt.title(titulo)
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.grid(False)
    plt.show()

lib.mandelbrot.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.mandelbrot.restype = ctypes.POINTER(ctypes.c_int)
lib.free_mandelbrot.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.free_mandelbrot.restype = None

@medir_tiempo
def calcular_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    M_ptr = lib.mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
    M = np.ctypeslib.as_array(M_ptr, shape=(height * width,))
    M_copy = np.copy(M).reshape(height, width)
    lib.free_mandelbrot(M_ptr)
    return M_copy


# Calcular Mandelbrot
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
width, height = 1000, 1000
max_iter = 100
M = calcular_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
mostrar_fractal(M, xmin, xmax, ymin, ymax, cmap='hot', titulo='Fractal de Mandelbrot')
print(f"Forma de la matriz Mandelbrot: {M.shape}")