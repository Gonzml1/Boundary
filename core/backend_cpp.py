import os
import ctypes
import numpy as np

DIRECCION_DE_CARPETA = "codigos_cpp"
# --- Helper para carga de DLLs de fractales C++ ---
class CPPBackend:
    def __init__(self, dll_name: str, dll_dir: str = None):
        # Determinar carpeta base de DLLs: por defecto ../codigos_cpp desde este archivo
        base_dir = dll_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, DIRECCION_DE_CARPETA)
        )
        dll_path = os.path.join(base_dir, dll_name)
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"DLL no encontrada: {dll_path}")
        try:
            self.lib = ctypes.CDLL(dll_path)
        except OSError as e:
            raise RuntimeError(f"Error cargando {dll_name}: {e}")

    def wrap_func(self, func_name: str, argtypes: list, restype):
        func = getattr(self.lib, func_name)
        func.argtypes = argtypes
        func.restype = restype
        return func

# --- Instanciación de backends específicos ---
# Asegúrate de que la carpeta 'codigos_cpp' esté en el nivel superior del proyecto
# y contiene las DLL: mandelbrot.dll, circulo.dll, etc.
_mandelbrot_cpp = CPPBackend('mandelbrot.dll')
mandelbrot_cpp = _mandelbrot_cpp.wrap_func(
    'mandelbrot',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.POINTER(ctypes.c_int)
)
free_mandelbrot = _mandelbrot_cpp.wrap_func(
    'free_mandelbrot', [ctypes.POINTER(ctypes.c_int)], None
)


_julia_cpp = CPPBackend('julia.dll')
julia_cpp = _julia_cpp.wrap_func(
    'julia',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int, 
     ctypes.c_double, ctypes.c_double],
    ctypes.POINTER(ctypes.c_int)
)
free_julia = _julia_cpp.wrap_func(
    'free_julia', [ctypes.POINTER(ctypes.c_int)], None
)

_burning_ship_cpp = CPPBackend('burning_ship.dll')
burning_ship_cpp = _burning_ship_cpp.wrap_func(
    'burning_ship',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.POINTER(ctypes.c_int)
)
free_burning_ship = _burning_ship_cpp.wrap_func(
    'free_burning_ship', [ctypes.POINTER(ctypes.c_int)], None
)

_newton_cpp = CPPBackend('newton.dll')
newton_cpp = _newton_cpp.wrap_func(
    'newton',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.POINTER(ctypes.c_int)
)
free_newton = _newton_cpp.wrap_func(
    'free_newton', [ctypes.POINTER(ctypes.c_int)], None
)

_tricorn_cpp = CPPBackend('tricorn.dll')
tricorn_cpp = _tricorn_cpp.wrap_func(
    'tricorn',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.POINTER(ctypes.c_int)
)
free_tricorn = _tricorn_cpp.wrap_func(
    'free_tricorn', [ctypes.POINTER(ctypes.c_int)], None
)

_phoenix_cpp = CPPBackend('phoenix.dll')
phoenix_cpp = _phoenix_cpp.wrap_func(
    'phoenix',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int,
     ctypes.c_double, ctypes.c_double],
    ctypes.POINTER(ctypes.c_int)
)
free_phoenix = _phoenix_cpp.wrap_func(
    'free_phoenix', [ctypes.POINTER(ctypes.c_int)], None
)

_burning_julia_cpp = CPPBackend('burning_julia.dll')
burning_julia_cpp = _burning_julia_cpp.wrap_func(
    'burning_julia',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int,
     ctypes.c_double, ctypes.c_double],
    ctypes.POINTER(ctypes.c_int)
)
free_burning_julia = _burning_julia_cpp.wrap_func(
    'free_burning_julia', [ctypes.POINTER(ctypes.c_int)], None
)

_nova_cpp = CPPBackend('nova.dll')
nova_cpp = _nova_cpp.wrap_func(
    'nova',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int,
     ctypes.c_double, ctypes.c_double],
    ctypes.POINTER(ctypes.c_int)
)
free_nova = _nova_cpp.wrap_func(
    'free_nova', [ctypes.POINTER(ctypes.c_int)], None
)

_coseno_cpp = CPPBackend('coseno.dll')
coseno_cpp = _coseno_cpp.wrap_func(
    'coseno',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.POINTER(ctypes.c_int)
)
free_coseno = _coseno_cpp.wrap_func(
    'free_coseno', [ctypes.POINTER(ctypes.c_int)], None
)

_coseno_inv_cpp = CPPBackend('coseno_inv.dll')
coseno_inv_cpp = _coseno_inv_cpp.wrap_func(
    'coseno_inv',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.POINTER(ctypes.c_int)
)
free_coseno_inv = _coseno_inv_cpp.wrap_func(
    'free_coseno_inv', [ctypes.POINTER(ctypes.c_int)], None
)

_celtic_cpp = CPPBackend('celtic_mandelbrot.dll')
celtic_mandelbrot_cpp = _celtic_cpp.wrap_func(
    'celtic_mandelbrot',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.POINTER(ctypes.c_int)
)
free_celtic_mandelbrot = _celtic_cpp.wrap_func(
    'free_celtic_mandelbrot', [ctypes.POINTER(ctypes.c_int)], None
)

_circulo_cpp = CPPBackend('circulo.dll')
circulo_cpp = _circulo_cpp.wrap_func(
    'circulo',
    [ctypes.c_double, ctypes.c_double,
     ctypes.c_double, ctypes.c_double,
     ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.POINTER(ctypes.c_int)
)
free_circulo = _circulo_cpp.wrap_func(
    'free_circulo', [ctypes.POINTER(ctypes.c_int)], None
)

# Otros backends (coseno, burning_ship, etc.) se pueden agregar igual:
# _coseno_cpp = CPPBackend('coseno.dll')
# coseno_cpp = _coseno_cpp.wrap_func('coseno', [...], ctypes.POINTER(ctypes.c_int))
# free_coseno = _coseno_cpp.wrap_func('free_coseno', [ctypes.POINTER(ctypes.c_int)], None)
