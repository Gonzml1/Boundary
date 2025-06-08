import numpy as np
import cupy as cp
import os
import ctypes
import time
from .base import register_fractal, FractalCalculator
from ..funciones_kernel import circulo_kernel
# Circulo #
###########

@register_fractal("Circulo", "GPU_Cupy_kernel")
def hacer_circulo_gpu(self) -> np.ndarray:
    inicio = time.time()

    x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
    y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y  
    C = C.ravel()   
    Z = cp.zeros_like(C, dtype=cp.complex128)  

    resultado = cp.empty(C.shape, dtype=cp.int32)

    try:
        circulo_kernel(Z, C, self.max_iter, resultado)
    except Exception as e:
        print(f"Error executing Circulo kernel: {e}")
        return None

    resultado = resultado.reshape((self.height, self.width))
    resultado_cpu = resultado.get()

    tiempo = time.time() - inicio
    print(f"{self.max_iter} iteraciones")
    print(f"Tiempo total: {tiempo:.5f} segundos")

    return resultado_cpu

@register_fractal("Circulo", "GPU_Cupy")
def hacer_circulo_cupy(self) -> np.ndarray:
    inicio = time.time()

    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros_like(C, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    matriz = cp.ones(C.shape, dtype=bool)

    for n in range(self.max_iter):
        z[matriz] = cp.exp((z[matriz]**2 - 1.00001*z[matriz]) / C[matriz]**4) 
        matriz = cp.logical_and(matriz, cp.abs(z) <= 2)
        M[matriz] = n
        print(f"\rCIRCULO {n}", end="", flush=True)

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M.get()

@register_fractal("Circulo", "CPU_Numpy")
def hacer_circulo_numpy(self) -> np.ndarray:
    inicio = time.time()

    x = np.linspace(self.xmin, self.xmax, self.width)
    y = np.linspace(self.ymin, self.ymax, self.height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    z = np.zeros_like(C, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)
    matriz = np.ones(C.shape, dtype=bool)

    for n in range(self.max_iter):
        z[matriz] = np.exp((z[matriz]**2 - 1.00001*z[matriz]) / C[matriz]**4) 
        matriz = np.logical_and(matriz, np.abs(z) <= 2)
        M[matriz] = n
        print(f"\rCIRCULO {n}", end="", flush=True)

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M


@register_fractal("Circulo", "CPU_cpp")
@FractalCalculator.medir_tiempo("Circulo CPP")
def hacer_circulo_cpp(self) -> np.ndarray:
    dll_path = r"codigos_cpp\circulo.dll"
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
    lib.circulo.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.circulo.restype = ctypes.POINTER(ctypes.c_int)
    lib.free_circulo.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.free_circulo.restype = None
    M_ptr = lib.circulo(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
    M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
    M_copy = np.copy(M).reshape(self.height, self.width)
    lib.free_circulo(M_ptr)
    return M_copy

