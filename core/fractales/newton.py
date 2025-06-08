import numpy as np
import cupy as cp
import os
import ctypes
import time
from .base import register_fractal, FractalCalculator
from ..funciones_kernel import newton_kernel
# Newton-Raphson #
##################

@register_fractal("Newton-Raphson", "GPU_Cupy_kernel")
def hacer_newton_gpu(self) -> np.ndarray:
    inicio = time.time()

    x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
    y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    C = C.ravel()

    root_index = cp.empty(C.shape, dtype=cp.int32)
    iter_count = cp.empty(C.shape, dtype=cp.int32)

    try:
        newton_kernel(C, self.max_iter, root_index, iter_count)
    except Exception as e:
        print(f"Error ejecutando el kernel de Newton: {e}")
        return None

    root_index = root_index.reshape((self.height, self.width))
    root_index_cpu = root_index.get()
    tiempo = time.time() - inicio
    print(f"{self.max_iter} iteraciones")
    print(f"Tiempo total: {tiempo:.5f} segundos")

    return root_index_cpu

@register_fractal("Newton-Raphson", "GPU_Cupy")
def hacer_newton_cupy(self) -> np.ndarray:
    inicio = time.time()
    
    def f(z):
        return z**3 - 1
    def df(z):
        return 3 * z**2

    raices = cp.array([1 + 0j,
                       -0.5 + 0.8660254j,
                       -0.5 - 0.8660254j])  

    tolerancia = 1e-6

    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    z = X + 1j * Y

    M = cp.zeros(z.shape, dtype=int)      
    N = cp.zeros(z.shape, dtype=int)      

    for n in range(self.max_iter):
        z = z - f(z) / df(z)

        for i, r in enumerate(raices):
            cerca = cp.abs(z - r) < tolerancia
            sin_color = (M == 0)
            M[cp.logical_and(cerca, sin_color)] = i + 1
            N[cerca] = n

        if cp.all(M > 0):
            break  

        print(f"\rNEWTON {n}", end="", flush=True)

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")

    return M.get() 

@register_fractal("Newton-Raphson", "CPU_Numpy")
def hacer_newton_numpy(self) -> np.ndarray:
    inicio = time.time()
    
    def f(z):
        return z**3 - 1
    def df(z):
        return 3 * z**2

    raices = np.array([1 + 0j,
                       -0.5 + 0.8660254j,
                       -0.5 - 0.8660254j])  

    tolerancia = 1e-6

    x = np.linspace(self.xmin, self.xmax, self.width)
    y = np.linspace(self.ymin, self.ymax, self.height)
    X, Y = np.meshgrid(x, y)
    z = X + 1j * Y

    M = np.zeros(z.shape, dtype=int)      
    N = np.zeros(z.shape, dtype=int)      

    for n in range(self.max_iter):
        z = z - f(z) / df(z)

        for i, r in enumerate(raices):
            cerca = np.abs(z - r) < tolerancia
            sin_color = (M == 0)
            M[np.logical_and(cerca, sin_color)] = i + 1
            N[cerca] = n

        if np.all(M > 0):
            break  

        print(f"\rNEWTON {n}", end="", flush=True)

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")

    return M  


@register_fractal("Newton-Raphson", "CPU_cpp")
@FractalCalculator.medir_tiempo("Newton CPP")
def hacer_newton_cpp(self) -> np.ndarray:
    dll_path = r"codigos_cpp\newton.dll"
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
    lib.newton.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.newton.restype = ctypes.POINTER(ctypes.c_int)
    lib.free_newton.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.free_newton.restype = None
    M_ptr = lib.newton(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
    M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
    M_copy = np.copy(M).reshape(self.height, self.width)
    lib.free_newton(M_ptr)
    return M_copy    

