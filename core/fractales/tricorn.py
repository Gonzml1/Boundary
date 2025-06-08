import numpy as np
import cupy as cp
import os
import ctypes
import time
from .base import register_fractal, FractalCalculator
from ..funciones_kernel import tricorn_kernel
# Tricorn #
###########

@register_fractal("Tricorn", "GPU_Cupy_kernel")
def hacer_tricorn_gpu(self) -> np.ndarray:
    inicio = time.time()

    x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
    y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y  
    C = C.ravel()   
    Z = cp.zeros_like(C, dtype=cp.complex128)  

    resultado = cp.empty(C.shape, dtype=cp.int32)

    try:
        tricorn_kernel(Z, C, self.max_iter, resultado)
    except Exception as e:
        print(f"Error executing Tricorn kernel: {e}")
        return None

    resultado = resultado.reshape((self.height, self.width))
    resultado_cpu = resultado.get()

    tiempo = time.time() - inicio
    print(f"{self.max_iter} iteraciones")
    print(f"Tiempo total: {tiempo:.5f} segundos")

    return resultado_cpu

@register_fractal("Tricorn", "GPU_Cupy")
def hacer_tricorn_cupy(self) -> np.ndarray:
    inicio = time.time()

    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros_like(C, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    matriz = cp.ones(C.shape, dtype=bool)

    for n in range(self.max_iter):
        z_temp = cp.conj(z)**2 + C
        z[matriz] = z_temp[matriz]
        matriz = cp.logical_and(matriz, cp.abs(z) <= 2)
        M[matriz] = n
        print(f"\rTRICORN {n}", end="", flush=True)

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M.get()

@register_fractal("Tricorn", "CPU_Numpy")
def hacer_tricorn_numpy(self) -> np.ndarray:
    inicio = time.time()

    x = np.linspace(self.xmin, self.xmax, self.width)
    y = np.linspace(self.ymin, self.ymax, self.height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    z = np.zeros_like(C, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)
    matriz = np.ones(C.shape, dtype=bool)

    for n in range(self.max_iter):
        z_temp = np.conj(z)**2 + C
        z[matriz] = z_temp[matriz]
        matriz = np.logical_and(matriz, np.abs(z) <= 2)
        M[matriz] = n
        print(f"\rTRICORN {n}", end="", flush=True)

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M

@register_fractal("Tricorn", "CPU_cpp")
@FractalCalculator.medir_tiempo("Tricorn CPP")
def hacer_tricorn_cpp(self) -> np.ndarray:
    dll_path = r"codigos_cpp\tricorn.dll"
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
    lib.tricorn.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.tricorn.restype = ctypes.POINTER(ctypes.c_int)
    lib.free_tricorn.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.free_tricorn.restype = None
    M_ptr = lib.tricorn(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
    M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
    M_copy = np.copy(M).reshape(self.height, self.width)
    lib.free_tricorn(M_ptr)
    return M_copy

