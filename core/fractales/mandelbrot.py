import numpy as np
from ..funciones_kernel import mandelbrot_kernel, mandelbrot_smooth_kernel
import cupy as cp
import os
import ctypes
import time
from .base import register_fractal, FractalCalculator
#    @register_fractal("Mandelbrotfast", "GPU_Cupy_kernel_fast")
@FractalCalculator.medir_tiempo("Mandelbrot CPP (Fast)")
def hacer_mandelbrotfast_cpp(self) -> np.ndarray:
    dll_path = r"codigos_cpp\mandelbrot.dll"
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

    lib.mandelbrot.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.mandelbrot.restype = ctypes.POINTER(ctypes.c_int)
    lib.free_mandelbrot.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.free_mandelbrot.restype = None

    M_ptr = lib.mandelbrot(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
    M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
    M_copy = np.copy(M).reshape(self.height, self.width)
    lib.free_mandelbrot(M_ptr)
    return M_copy

#    @register_fractal("Mandelbrotprecision", "cpp_precision")
@FractalCalculator.medir_tiempo("Mandelbrot CPP (Precision)")
def hacer_mandelbrotprecision_cpp(self, precision: int = 100) -> np.ndarray:
    """
    Calcula el fractal Mandelbrot con una precisión específica.
    """
    dll_path = r"codigos_cpp\mandelbrotprecision.dll"
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

    lib.mandelbrot_ap.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.mandelbrot_ap.restype = ctypes.POINTER(ctypes.c_int)
    lib.free_mandelbrot_ap.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.free_mandelbrot_ap.restype = None

    M_ptr = lib.mandelbrot_ap(
        self.xmin, self.xmax, self.ymin, self.ymax,
        self.width, self.height, self.max_iter, precision
    )
    
    M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
    M_copy = np.copy(M).reshape(self.height, self.width)

    lib.free_mandelbrot_ap(M_ptr)
    
    return M_copy

##############
# Mandelbrot #
##############

@register_fractal("Mandelbrot", "GPU_Cupy_kernel_smooth")
@FractalCalculator.medir_tiempo("Mandelbrot GPU (Smooth Coloring)")
def hacer_mandelbrot_gpu_smooth(self) -> np.ndarray:
    # 1) Crear ejes reales en float64
    x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
    y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
    # 2) Malla 2D
    X, Y = cp.meshgrid(x, y)
    # 3) Construir C en complex128 y aplanar
    C = (X + 1j * Y).astype(cp.complex128).ravel()
    # 4) Buffer de salida de float64
    N = C.size
    resultado = cp.empty(N, dtype=cp.float64)

    try:
        # 5) INVOCACIÓN CORRECTA del ElementwiseKernel
        mandelbrot_smooth_kernel(
            C,
            np.int32(self.max_iter),
            resultado
        )
    except Exception as e:
        print(f"Error ejecutando mandelbrot_smooth_kernel: {e}")
        raise

    # 6) Remodelar a 2D y bajar a CPU
    resultado_2d = resultado.reshape((self.height, self.width))
    return resultado_2d.get()


@register_fractal("Mandelbrot", "GPU_Cupy_kernel")
@FractalCalculator.medir_tiempo("Mandelbrot GPU")
def hacer_mandelbrot_gpu(self) -> np.ndarray:
    x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
    y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y  
    C = C.ravel() 
    
    resultado = cp.empty(C.shape, dtype=cp.int32)
    
    try:
        mandelbrot_kernel(C, self.max_iter, resultado)
    except Exception as e:
        print(f"Error executing Julia kernel: {e}")
        return None
        
    resultado = resultado.reshape((self.height, self.width))
    resultado_cpu = resultado.get()

    return resultado_cpu

@FractalCalculator.medir_tiempo("Mandelbrot Entrada")
def hacer_mandelbrot_con_entrada(self) -> np.ndarray:
    operacion = self.transformar_expresion(self.formula, ["z", "C"])
    codigo = compile(operacion, "<string>", "exec")
    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin, self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    matriz = cp.ones(C.shape, dtype=bool)

    for n in range(self.max_iter):
        exec(codigo)
        matriz = cp.logical_and(matriz, cp.abs(z) <= 2)
        M[matriz] = n
        print(f"\rMANDELBROT {n}", end="", flush=True)

    return M.get()

@register_fractal("Mandelbrot", "GPU_Cupy")
def hacer_mandelbrot_cupy(self) -> np.ndarray:
    inicio = time.time()

    operacion = self.transformar_expresion(self.formula, ["z", "C"])
    codigo = compile(operacion, "<string>", "exec")

    x = cp.linspace(self.xmin, self.xmax, self.width)
    y = cp.linspace(self.ymin,self.ymax, self.height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    matriz = cp.ones(C.shape, dtype=bool)

    for n in range(self.max_iter):
        exec(codigo)
        matriz = cp.logical_and(matriz, cp.abs(z) <= 2)
        M[matriz] = n
        print(f"\rMANDELBROT {n}", end="", flush=True)

    fin = time.time() 
    print("\nTiempo de ejecución:", fin - inicio, "segundos")

    return M.get()

@register_fractal("Mandelbrot", "CPU_Numpy")
def hacer_mandelbrot_numpy(self) -> np.ndarray:
    inicio = time.time()
    
    operacion = self.transformar_expresion(self.formula, ["z", "C"])
    codigo = compile(operacion, "<string>", "exec")
    
    x = np.linspace(self.xmin, self.xmax, self.width)
    y = np.linspace(self.ymin, self.ymax, self.height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)
    matriz = np.ones(C.shape, dtype=bool)
    
    for n in range(self.max_iter):
        exec(codigo)
        matriz = np.logical_and(matriz, np.abs(z) <= 2)
        M[matriz] = n
        print(f"\rMANDELBROT {n}", end="", flush=True)
        
    fin = time.time() 
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    
    return M


@register_fractal("Mandelbrot", "CPU_cpp")
@FractalCalculator.medir_tiempo("Mandelbrot CPP")
def hacer_mandelbrot_cpp(self) -> np.ndarray:
    dll_path = r"codigos_cpp\mandelbrot.dll"
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
    lib.mandelbrot.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.mandelbrot.restype = ctypes.POINTER(ctypes.c_int)
    lib.free_mandelbrot.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.free_mandelbrot.restype = None
    M_ptr = lib.mandelbrot(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
    M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
    M_copy = np.copy(M).reshape(self.height, self.width)
    lib.free_mandelbrot(M_ptr)
    return M_copy

################
# Para olvidar #
################

@FractalCalculator.medir_tiempo("Mandelbrot C")
def hacer_mandelbrot_c(self, show_plot: bool = False) -> np.ndarray:
    _here = os.path.dirname(__file__)
    lib = ctypes.CDLL(os.path.join(_here, "libmandelbrot.so"))

    # Firma para double-precision SIMD
    lib.mandelbrot_simd_double.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.mandelbrot_simd_double.restype = None

    # Ejes en float64
    xs = np.linspace(self.xmin, self.xmax, self.width, dtype=np.float64)
    ys = np.linspace(self.ymin, self.ymax, self.height, dtype=np.float64)
    N  = self.width * self.height

    # Grid ravelizado
    cx = np.repeat(xs[np.newaxis, :], self.height, axis=0).ravel()
    cy = np.repeat(ys[:, np.newaxis], self.width, axis=1).ravel()
    cx = np.ascontiguousarray(cx, dtype=np.float64)
    cy = np.ascontiguousarray(cy, dtype=np.float64)

    # Memoria para el resultado
    iters = np.empty(N, dtype=np.int32)

    # Llamada al C
    lib.mandelbrot_simd_double(
        cx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        iters.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(N),
        ctypes.c_int(self.max_iter)
    )

    # Devolvemos la matriz en 2D
    img = iters.reshape((self.height, self.width))
    return img
