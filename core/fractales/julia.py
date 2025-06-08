import numpy as np
import cupy as cp
import os
import ctypes
import time
from .base import register_fractal, FractalCalculator
from ..funciones_kernel import julia_kernel
# Julia #
#########

@register_fractal("Julia", "GPU_Cupy_kernel")
def hacer_julia_gpu(self) -> np.ndarray:
    inicio = time.time()

    x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
    y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
    X, Y = cp.meshgrid(x, y)
    Z = X + 1j * Y  
    Z = Z.ravel()   

    C = cp.full(Z.shape, complex(self.real, self.imag), dtype=cp.complex128)

    resultado = cp.empty(Z.shape, dtype=cp.int32)

    try:
        julia_kernel(Z, C, self.max_iter, resultado)
    except Exception as e:
        print(f"Error executing Julia kernel: {e}")
        return None

    resultado = resultado.reshape((self.height, self.width))
    resultado_cpu = resultado.get()

    tiempo = time.time() - inicio
    print(f"{self.max_iter} iteraciones")
    print(f"Tiempo total: {tiempo:.5f} segundos")

    return resultado_cpu

#revisar
@register_fractal("Julia", "GPU_Cupy")
def hacer_julia_cupy(self) -> np.ndarray:
    inicio = time.time()

    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    matriz = cp.ones(C.shape, dtype=bool)

    for n in range(self.max_iter):
        z[matriz]=z[matriz]**2+(self.real+1j*self.imag)
        matriz = cp.logical_and(matriz, cp.abs(z) <= 2)
        M[matriz] = n
        print(f"\rJULIA {n}", end="", flush=True)

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M.get()

@register_fractal("Julia", "CPU_Numpy")
def hacer_julia_numpy(self) -> np.ndarray:
    inicio = time.time()

    x = np.linspace(self.xmin, self.xmax, self.width)
    y = np.linspace(self.ymin, self.ymax, self.height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)
    matriz = np.ones(C.shape, dtype=bool)

    for n in range(self.max_iter):
        z[matriz]=z[matriz]**2+(self.real+1j*self.imag)
        matriz = np.logical_and(matriz, np.abs(z) <= 2)
        M[matriz] = n
        print(f"\rJULIA {n}", end="", flush=True)

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M

@register_fractal("Julia", "CPU_cpp")
@FractalCalculator.medir_tiempo("Julia CPP")
def hacer_julia_cpp(self) -> np.ndarray:
    dll_path = r"codigos_cpp\julia.dll"
    if not os.path.exists(dll_path):
        print(f"Error: No se encuentra la DLL en {dll_path}")
        exit(1)
    try:
        lib = ctypes.WinDLL(dll_path)
    except OSError as e:
        print(f"Error al cargar la DLL: {e}")
        raise
    
    lib.julia.argtypes = [
        ctypes.c_double,  # xmin
        ctypes.c_double,  # xmax
        ctypes.c_double,  # ymin
        ctypes.c_double,  # ymax
        ctypes.c_int,     # width
        ctypes.c_int,     # height
        ctypes.c_int,     # max_iter
        ctypes.c_double,  # cr
        ctypes.c_double,  # ci
    ]
    lib.julia.restype = ctypes.POINTER(ctypes.c_int)

    lib.free_julia.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.free_julia.restype = None


    M_ptr = lib.julia(
        self.xmin, self.xmax,
        self.ymin, self.ymax,
        self.width, self.height,
        self.max_iter,
        self.real,   # constante real de Julia
        self.imag    # constante imaginaria de Julia
    )

    # Convertimos el puntero a un array de NumPy
    M_flat = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
    M_copy = np.copy(M_flat).reshape(self.height, self.width)

    # Liberamos la memoria en C
    lib.free_julia(M_ptr)

    return M_copy

