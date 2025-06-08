import numpy as np
import cupy as cp
import os
import ctypes
import time
from .base import register_fractal, FractalCalculator
from ..funciones_kernel import phoenix_kernel
# Phoenix #
###########

@register_fractal("Phoenix", "CPU_Numpy")
def hacer_phoenix_numpy(self) -> np.ndarray:
    """
    Phoenix Fractal (Mandelbrot‐style): 
       z_{n+1} = z_n^2 + p*z_{n-1} + C,  con C en la malla.
    Necesita almacenar z_n y z_{n-1} (aquí z y zp).
    """
    inicio = time.time()

    x = np.linspace(self.xmin, self.xmax, self.width)
    y = np.linspace(self.ymin, self.ymax, self.height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    z = np.zeros(C.shape, dtype=np.complex128)
    zp = np.zeros(C.shape, dtype=np.complex128)

    M = np.zeros(C.shape, dtype=int)
    mask = np.ones(C.shape, dtype=bool)

    p = self.real + 1j * self.imag

    for n in range(self.max_iter):
        z_next = z[mask] * z[mask] + p * zp[mask] + C[mask]

        zp[mask] = z[mask]
        z[mask] = z_next

        escaped = np.abs(z) > 2.0
        just_escaped = mask & escaped
        M[just_escaped] = n
        mask[just_escaped] = False

        print(f"\rPHOENIX {n}", end="", flush=True)
        if not mask.any():
            break

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M

@register_fractal("Phoenix", "GPU_Cupy")
def hacer_phoenix_cupy(self) -> np.ndarray:
    """
    Phoenix Fractal (Mandelbrot‐style): 
       z_{n+1} = z_n^2 + p*z_{n-1} + C,  con C en la malla.
    Necesita almacenar z_n y z_{n-1} (aquí z y zp).
    """
    inicio = time.time()
    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros(C.shape, dtype=cp.complex128)
    zp = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    mask = cp.ones(C.shape, dtype=bool)
    p = self.imag + 1j * self.real  # p = c_im + i*c_re
    for n in range(self.max_iter):
        z_next = z[mask] * z[mask] + p * zp[mask] + C[mask]
        zp[mask] = z[mask]
        z[mask] = z_next
        escaped = cp.abs(z) > 2.0
        just_escaped = mask & escaped
        M[just_escaped] = n
        mask[just_escaped] = False
        print(f"\rPHOENIX {n}", end="", flush=True)
        if not mask.any():
            break
    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    
    return M.get()

@register_fractal("Phoenix", "GPU_Cupy_kernel")
@FractalCalculator.medir_tiempo("Phoenix GPU")
def hacer_phoenix_gpu(self):
    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros(C.shape, dtype=cp.complex128)
    zp = cp.zeros(C.shape, dtype=cp.complex128)
    mask = cp.ones(C.shape, dtype=cp.bool_)
    M = cp.zeros(C.shape, dtype=cp.int32)
    p = self.real + 1j * self.imag

    for n in range(self.max_iter):
        z_new, zp_new, mask_new = phoenix_kernel(z, zp, C, mask, p)
        just_escaped = mask & (~mask_new)
        M[just_escaped] = n
        z = z_new
        zp = zp_new
        mask = mask_new
        if not bool(mask.any()):
            break
        
    return M.get()

@register_fractal("Phoenix", "CPU_cpp")
@FractalCalculator.medir_tiempo("Phoenix CPP")
def hacer_phoenix_cpp(self) -> np.ndarray:
    dll_path = r"codigos_cpp\phoenix.dll"
    if not os.path.exists(dll_path):
        print(f"Error: No se encuentra la DLL en {dll_path}")
        exit(1)

    try:
        lib = ctypes.WinDLL(dll_path)
    except OSError as e:
        print(f"Error al cargar la DLL: {e}")
        print("Verifica que la DLL y sus dependencias estén en el directorio o en el PATH.")
        raise

    lib.phoenix.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
    ]
    lib.phoenix.restype = ctypes.POINTER(ctypes.c_int)
    lib.free_phoenix.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.free_phoenix.restype = None

    M_ptr = lib.phoenix(
        self.xmin, self.xmax, self.ymin, self.ymax,
        self.width, self.height, self.max_iter,
        self.real, self.imag
    )
    M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
    M_copy = np.copy(M).reshape(self.height, self.width)
    lib.free_phoenix(M_ptr)
    return M_copy

