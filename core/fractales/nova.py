import numpy as np
import cupy as cp
import os
import ctypes
import time
from .base import register_fractal, FractalCalculator
from ..funciones_kernel import nova_kernel
# Nova #
########

@register_fractal("Nova", "CPU_Numpy")
def hacer_nova_numpy(self) -> np.ndarray:
    """
    Nova Fractal: z_{n+1} = z_n^m + C + k * z_n^{-m}
    con C = X + iY, z_0 = C, m = self.nova_m, k = self.nova_k.
    """
    inicio = time.time()

    x = np.linspace(self.xmin, self.xmax, self.width)
    y = np.linspace(self.ymin, self.ymax, self.height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    # Inicializamos z0 = C (variante común en Nova)
    z = np.copy(C)

    M = np.zeros(C.shape, dtype=int)
    mask = np.ones(C.shape, dtype=bool)

    m = self.nova_m
    k = self.nova_k

    for n in range(self.max_iter):
        z_act = z[mask]

        # Evitar división por cero
        z_safe = np.where(z_act == 0, 1e-16 + 0j, z_act)

        # z^m y z^{-m}
        z_pow_m = z_safe ** m
        z_pow_negm = z_safe ** (-m)

        # Iteración Nova
        z_next = z_pow_m + C[mask] + k * z_pow_negm

        z[mask] = z_next

        # Escape
        escaped = np.abs(z) > 2.0
        just_escaped = mask & escaped
        M[just_escaped] = n
        mask[just_escaped] = False

        print(f"\rNOVA {n}", end="", flush=True)
        if not mask.any():
            break

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M

@register_fractal("Nova", "GPU_Cupy")
def hacer_nova_cupy(self) -> np.ndarray:
    """
    Nova Fractal: z_{n+1} = z_n^m + C + k * z_n^{-m}
    con C = X + iY, z_0 = C, m = self.nova_m, k = self.nova_k.
    """
    inicio = time.time()
    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    # Inicializamos z0 = C (variante común en Nova)
    z = cp.copy(C)
    M = cp.zeros(C.shape, dtype=int)
    mask = cp.ones(C.shape, dtype=bool)
    m = self.imag
    k = self.real
    for n in range(self.max_iter):
        z_act = z[mask]
        # Evitar división por cero
        z_safe = cp.where(z_act == 0, 1e-16 + 0j, z_act)
        # z^m y z^{-m}
        z_pow_m = z_safe ** m
        z_pow_negm = z_safe ** (-m)
        # Iteración Nova
        z_next = z_pow_m + C[mask] + k * z_pow_negm
        z[mask] = z_next
        # Escape
        escaped = cp.abs(z) > 2.0
        just_escaped = mask & escaped
        M[just_escaped] = n
        mask[just_escaped] = False
        print(f"\rNOVA {n}", end="", flush=True)
        if not mask.any():
            break
    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M.get()

@register_fractal("Nova", "GPU_Cupy_kernel")
@FractalCalculator.medir_tiempo("Nova GPU")
def hacer_nova_gpu(self) -> np.ndarray:
    """
    Nova Fractal con ElementwiseKernel en CuPy.
    z_{n+1} = z_n^m + C + k * z_n^{-m}
    con C = X + iY, z_0 = C, m = self.nova_m, k = self.nova_k.
    """
    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y

    z = cp.copy(C)  # z0 = C
    mask = cp.ones(C.shape, dtype=cp.bool_)
    M = cp.zeros(C.shape, dtype=cp.int32)

    m = self.imag           # potencia (asegúrate que sea float)
    k = self.real                  # número complejo

    for n in range(self.max_iter):
        z_new, mask_new = nova_kernel(z, C, mask, m, k)
        just_escaped = mask & (~mask_new)
        M[just_escaped] = n
        z = z_new
        mask = mask_new
        print(f"\rNOVA {n}", end="", flush=True)
        if not bool(mask.any()):
            break

    return M.get()

@register_fractal("Nova", "CPU_cpp")
@FractalCalculator.medir_tiempo("Nova CPP")
def hacer_nova_cpp(self) -> np.ndarray:
    dll_path = r"codigos_cpp\nova.dll"
    if not os.path.exists(dll_path):
        print(f"Error: No se encuentra la DLL en {dll_path}")
        exit(1)

    try:
        lib = ctypes.WinDLL(dll_path)
    except OSError as e:
        print(f"Error al cargar la DLL: {e}")
        print("Verifica que la DLL y sus dependencias estén en el directorio o en el PATH.")
        raise

    lib.nova.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
    ]
    lib.nova.restype = ctypes.POINTER(ctypes.c_int)
    lib.free_nova.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.free_nova.restype = None

    M_ptr = lib.nova(
        self.xmin, self.xmax, self.ymin, self.ymax,
        self.width, self.height, self.max_iter,
        self.nova_m, self.nova_k
    )
    M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
    M_copy = np.copy(M).reshape(self.height, self.width)
    lib.free_nova(M_ptr)
    return M_copy

