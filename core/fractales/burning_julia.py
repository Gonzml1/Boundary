import numpy as np
import cupy as cp
import os
import ctypes
import time
from .base import register_fractal, FractalCalculator
from ..funciones_kernel import burning_julia_kernel
# Burning Julia #
#################

@register_fractal("Burning Julia", "CPU_Numpy")
def hacer_burning_julia_numpy(self) -> np.ndarray:
    """
    Burning Julia: z_{n+1} = (|Re(z_n)| + i|Im(z_n)|)^2 + c
    con c = self.real + 1j*self.imag y z_0 = X + iY.
    """
    inicio = time.time()

    x = np.linspace(self.xmin, self.xmax, self.width)
    y = np.linspace(self.ymin, self.ymax, self.height)
    X, Y = np.meshgrid(x, y)
    z = X + 1j * Y  # z_0
    M = np.zeros(z.shape, dtype=int)
    mask = np.ones(z.shape, dtype=bool)

    c = self.real + 1j * self.imag

    for n in range(self.max_iter):

        z_act = z[mask]

        re_abs = np.abs(z_act.real)
        im_abs = np.abs(z_act.imag)
        z_abs = re_abs + 1j * im_abs
        z_next = z_abs * z_abs + c

        z[mask] = z_next
        escaped = np.abs(z) > 2.0
        just_escaped = mask & escaped
        M[just_escaped] = n
        mask[just_escaped] = False

        print(f"\rBURNING JULIA {n}", end="", flush=True)
        if not mask.any():
            break

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M

@register_fractal("Burning Julia", "GPU_Cupy")
def hacer_burning_julia_cupy(self) -> np.ndarray:
    """
    Burning Julia: z_{n+1} = (|Re(z_n)| + i|Im(z_n)|)^2 + c
    con c = self.real + 1j*self.imag y z_0 = X + iY.
    """
    inicio = time.time()

    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    z = X + 1j * Y  # z_0
    M = cp.zeros(z.shape, dtype=int)
    mask = cp.ones(z.shape, dtype=bool)

    c = self.real + 1j * self.imag

    for n in range(self.max_iter):

        z_act = z[mask]

        re_abs = cp.abs(z_act.real)
        im_abs = cp.abs(z_act.imag)
        z_abs = re_abs + 1j * im_abs
        z_next = z_abs * z_abs + c

        z[mask] = z_next
        escaped = cp.abs(z) > 2.0
        just_escaped = mask & escaped
        M[just_escaped] = n
        mask[just_escaped] = False

        print(f"\rBURNING JULIA {n}", end="", flush=True)
        if not mask.any():
            break

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M.get()

@register_fractal("Burning Julia", "GPU_Cupy_kernel")
@FractalCalculator.medir_tiempo("Burning Julia GPU")
def hacer_burning_julia_gpu(self):
    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    z = X + 1j * Y
    mask = cp.ones(z.shape, dtype=cp.bool_)
    M = cp.zeros(z.shape, dtype=cp.int32)
    c = self.real + 1j * self.imag

    for n in range(self.max_iter):
        z_new, mask_new = burning_julia_kernel(z, c, mask)
        just_escaped = mask & (~mask_new)
        M[just_escaped] = n
        z = z_new
        mask = mask_new
        if not bool(mask.any()):
            break

    return M.get()

@register_fractal("Burning Julia", "CPU_cpp")
@FractalCalculator.medir_tiempo("Burning Julia CPP")
def hacer_burning_julia_cpp(self) -> np.ndarray:
    dll_path = r"codigos_cpp\burning_julia.dll"
    if not os.path.exists(dll_path):
        print(f"Error: No se encuentra la DLL en {dll_path}")
        exit(1)

    try:
        lib = ctypes.WinDLL(dll_path)
    except OSError as e:
        print(f"Error al cargar la DLL: {e}")
        print("Verifica que la DLL y sus dependencias estén en el directorio o en el PATH.")
        raise

    lib.burning_julia.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
    ]
    lib.burning_julia.restype = ctypes.POINTER(ctypes.c_int)
    lib.free_burning_julia.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.free_burning_julia.restype = None

    M_ptr = lib.burning_julia(
        self.xmin, self.xmax, self.ymin, self.ymax,
        self.width, self.height, self.max_iter,
        self.real, self.imag
    )
    M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
    M_copy = np.copy(M).reshape(self.height, self.width)
    lib.free_burning_julia(M_ptr)
    return M_copy
    
