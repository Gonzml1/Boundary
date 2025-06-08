import numpy as np
import cupy as cp
import os
import ctypes
import time
from .base import register_fractal, FractalCalculator
from ..funciones_kernel import burning_kernel
# Burning Ship #
################

@register_fractal("Burning Ship", "GPU_Cupy_kernel")
def hacer_burning_gpu(self) -> np.ndarray:
    inicio = time.time()

    x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
    y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y  
    Z = cp.zeros_like(C, dtype=cp.complex128)  
    C = C.ravel()
    Z = Z.ravel()

    resultado = cp.empty(C.shape, dtype=cp.int32)

    try:
        burning_kernel(Z, C, self.max_iter, resultado)
    except Exception as e:
        print(f"Error executing Burning Ship kernel: {e}")
        return None

    resultado = resultado.reshape((self.height, self.width))
    resultado_cpu = resultado.get()
    tiempo = time.time() - inicio
    print(f"{self.max_iter} iteraciones")
    print(f"Tiempo total: {tiempo:.5f} segundos")

    return resultado_cpu

@register_fractal("Burning Ship", "GPU_Cupy")
def hacer_burning_cupy(self) -> np.ndarray:
    inicio = time.time()

    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros_like(C, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    matriz = cp.ones(C.shape, dtype=bool)

    for n in range(self.max_iter):
        z_real = cp.abs(z.real)
        z_imag = cp.abs(z.imag)
        z_temp = (z_real + 1j * z_imag) ** 2 + C
        z[matriz] = z_temp[matriz]
        matriz = cp.logical_and(matriz, cp.abs(z) <= 2)
        M[matriz] = n
        print(f"\rBurning Ship {n}", end="", flush=True)

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M.get()

@register_fractal("Burning Ship", "CPU_Numpy")
def hacer_burning_numpy(self) -> np.ndarray:
    inicio = time.time()

    x = np.linspace(self.xmin, self.xmax, self.width)
    y = np.linspace(self.ymin, self.ymax, self.height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    z = np.zeros_like(C, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)
    matriz = np.ones(C.shape, dtype=bool)

    for n in range(self.max_iter):
        z_real = np.abs(z.real)
        z_imag = np.abs(z.imag)
        z_temp = (z_real + 1j * z_imag) ** 2 + C
        z[matriz] = z_temp[matriz]
        matriz = np.logical_and(matriz, np.abs(z) <= 2)
        M[matriz] = n
        print(f"\rBurning Ship {n}", end="", flush=True)

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M


@register_fractal("Burning Ship", "CPU_cpp")
@FractalCalculator.medir_tiempo("Burning Ship CPP")
def hacer_burning_cpp(self) -> np.ndarray:
    dll_path = r"codigos_cpp\burning_ship.dll"
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
    lib.burning_ship.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.burning_ship.restype = ctypes.POINTER(ctypes.c_int)
    lib.free_burning_ship.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.free_burning_ship.restype = None
    M_ptr = lib.burning_ship(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
    M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
    M_copy = np.copy(M).reshape(self.height, self.width)
    lib.free_burning_ship(M_ptr)
    return M_copy

