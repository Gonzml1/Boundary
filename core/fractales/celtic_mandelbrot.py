import numpy as np
import cupy as cp
import os
import ctypes
import time
from .base import register_fractal, FractalCalculator
from ..funciones_kernel import celtic_mandelbrot_kernel
# Celtic Mandelbrot #
#####################

@register_fractal("Celtic Mandelbrot", "CPU_Numpy")
def hacer_celtic_mandelbrot_numpy(self) -> np.ndarray:
    """
    Celtic Mandelbrot: z_{n+1} = sqrt(|Re(z_n)| + i|Im(z_n)|) + C
    con C = X + iY y z_0 = 0.
    """
    inicio = time.time()

    x = np.linspace(self.xmin, self.xmax, self.width)
    y = np.linspace(self.ymin, self.ymax, self.height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)
    mask = np.ones(C.shape, dtype=bool)

    for n in range(self.max_iter):
        z_act = z[mask]

        re_abs = np.abs(z_act.real)
        im_abs = np.abs(z_act.imag)
        z_abs = re_abs + 1j * im_abs


        z_sqrt = np.sqrt(z_abs)

        z_next = z_sqrt + C[mask]

        z[mask] = z_next


        escaped = np.abs(z) > 2.0
        just_escaped = mask & escaped
        M[just_escaped] = n
        mask[just_escaped] = False

        print(f"\rCELTIC MANDELBROT {n}", end="", flush=True)
        if not mask.any():
            break

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M

@register_fractal("Celtic Mandelbrot", "GPU_Cupy")
def hacer_celtic_mandelbrot_cupy(self) -> np.ndarray:
    """
    Celtic Mandelbrot: z_{n+1} = sqrt(|Re(z_n)| + i|Im(z_n)|) + C
    con C = X + iY y z_0 = 0.
    """
    inicio = time.time()
    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    mask = cp.ones(C.shape, dtype=bool)

    for n in range(self.max_iter):
        z_act = z[mask]

        re_abs = cp.abs(z_act.real)
        im_abs = cp.abs(z_act.imag)
        z_abs = re_abs + 1j * im_abs
        z_sqrt = cp.sqrt(z_abs)

        z_next = z_sqrt + C[mask]

        z[mask] = z_next

        escaped = cp.abs(z) > 2.0
        just_escaped = mask & escaped
        M[just_escaped] = n
        mask[just_escaped] = False

        print(f"\rCELTIC MANDELBROT {n}", end="", flush=True)
        if not mask.any():
            break

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M.get()

@register_fractal("Celtic Mandelbrot", "GPU_Cupy_kernel")
@FractalCalculator.medir_tiempo("Celtic Mandelbrot GPU")
def hacer_celtic_mandelbrot_gpu(self, z0=None) -> np.ndarray:
    """
    Celtic Mandelbrot/Julia con kernel, eligiendo z0.
    Si z0=None, se usa z0 = 0 (Celtic Mandelbrot).
    Si z0 es un array, puede ser X+1j*Y (Celtic Julia) o lo que quieras.
    """
    import time
    inicio = time.time()
    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y

    # Selección de z0 (valor inicial)
    if self.real is None:
        z = cp.zeros(C.shape, dtype=cp.complex128)
    elif np.isscalar(z0):
        z = cp.full(C.shape, self.real, dtype=cp.complex128)
    else:
        # Se espera que z0 tenga misma forma que C (array tipo Julia, etc)
        z = cp.array(self.real, dtype=cp.complex128)

    mask = cp.ones(C.shape, dtype=cp.bool_)
    M = cp.zeros(C.shape, dtype=cp.int32)

    for n in range(self.max_iter):
        z_new, mask_new = celtic_mandelbrot_kernel(z, C, mask)
        just_escaped = mask & (~mask_new)
        M[just_escaped] = n
        z = z_new
        mask = mask_new

        print(f"\rCELTIC MANDELBROT GPU {n}", end="", flush=True)
        if not bool(mask.any()):
            break

    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M.get()

@register_fractal("Celtic Mandelbrot", "CPU_cpp")
@FractalCalculator.medir_tiempo("Celtic Mandelbrot CPP")
def hacer_celtic_mandelbrot_cpp(self) -> np.ndarray:
    dll_path = r"codigos_cpp\celtic_mandelbrot.dll"
    if not os.path.exists(dll_path):
        print(f"Error: No se encuentra la DLL en {dll_path}")
        exit(1)

    try:
        lib = ctypes.WinDLL(dll_path)
    except OSError as e:
        print(f"Error al cargar la DLL: {e}")
        print("Verifica que la DLL y sus dependencias estén en el directorio o en el PATH.")
        raise

    lib.celtic_mandelbrot.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.celtic_mandelbrot.restype = ctypes.POINTER(ctypes.c_int)
    lib.free_celtic_mandelbrot.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.free_celtic_mandelbrot.restype = None

    M_ptr = lib.celtic_mandelbrot(
        self.xmin, self.xmax, self.ymin, self.ymax,
        self.width, self.height, self.max_iter
    )
    M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
    M_copy = np.copy(M).reshape(self.height, self.width)
    lib.free_celtic_mandelbrot(M_ptr)
    return M_copy

