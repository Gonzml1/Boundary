import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time 
from OpenGL.GL import *
from .funciones_kernel import *
import os
from functools import wraps
import ctypes
from gui.MandelbrotGUI import Ui_Boundary
from scipy.special import gamma
#from  .  import mandelbrot
from .backend_cpp import *

# cp.exp((z[matriz]**2 - 1.00001*z[matriz]) / C[matriz]**4) 
# z[matriz] = z[matriz]**2 + C[matriz]    

FRACTAL_REGISTRY: dict[str, dict[str, callable]] = {}

def register_fractal(fractal: str, calc: str) -> callable:
    """
    Decorador: registra la función en FRACTAL_REGISTRY bajo
    FRACTAL_REGISTRY[fractal][calc] = fn
    """
    def deco(fn) -> callable:
        # Si no existe la clave 'fractal', la creamos:
        if fractal not in FRACTAL_REGISTRY:
            FRACTAL_REGISTRY[fractal] = {}
        # Asociamos el nombre de cálculo a la función concreta:
        FRACTAL_REGISTRY[fractal][calc] = fn
        return fn
    return deco

#para añadir en un futuro
class calculos_mandelbrot:
    def __init__(self, xmin: float, xmax: float , ymin: float, ymax: float, 
                 width: int, height: int, max_iter: int, formula: str, 
                 tipo_calculo: str, tipo_fractal: str, real: float, imag: float , 
                 ui=None) -> None:
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.formula = formula
        self.tipo_calculo = tipo_calculo
        self.tipo_fractal = tipo_fractal
        self.real = real
        self.imag = imag
        self.x_np = np.linspace(self.xmin, self.xmax, self.width, dtype=np.float64)
        self.y_np = np.linspace(self.ymin, self.ymax, self.height, dtype=np.float64)
        self.ui   = ui
        self.p  = 1.0
        self.nova_m = 1.0
        self.nova_k = 1.0
#        self.x_cp = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
#        self.y_cp = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
        if self.ui is not None:
            self._llenar_combo_fractales()

    def _llenar_combo_fractales(self) -> None:

        # Primero, limpias el comboBox por las dudas:
        self.ui.tipo_fractal_comboBox.clear()
        self.ui.tipo_calculo_comboBox.clear()

        for fractal in FRACTAL_REGISTRY:
            self.ui.tipo_fractal_comboBox.addItem(fractal)

        if FRACTAL_REGISTRY:
            primer_fractal = next(iter(FRACTAL_REGISTRY))
            for calc in FRACTAL_REGISTRY[primer_fractal]:
                self.ui.tipo_calculo_comboBox.addItem(calc)

        self.ui.tipo_fractal_comboBox.currentTextChanged.connect(
            self._on_fractal_cambiado
        )
    
    def _on_fractal_cambiado(self, nombre_fractal: str) -> None:
        """
        Se ejecuta cuando el usuario elige otro fractal;
        recarga el combo de 'cálculos' según lo registrado.
        """
        self.ui.tipo_calculo_comboBox.clear()
        if nombre_fractal in FRACTAL_REGISTRY:
            for calc in FRACTAL_REGISTRY[nombre_fractal]:
                self.ui.tipo_calculo_comboBox.addItem(calc)
    
    @staticmethod
    def medir_tiempo(nombre) -> callable:
        """
        Decorador para medir el tiempo de ejecución de una función.
        """
        def decorador(func) -> callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> any:
                inicio = time.time()
                resultado = func(*args, **kwargs)
                fin = time.time()
                print(f"⏱️ Tiempo de ejecución de '{nombre}': {fin - inicio:.5f} segundos")
                return resultado
            return wrapper
        return decorador
        
    def actualizar_fractal(self,
            xmin: float,  xmax: float,
            ymin: float,  ymax: float,
            width: int,   height: int,
            max_iter: int, formula: str,
            tipo_calculo: str, tipo_fractal: str,
            real: float, imag: float) -> None:
        """
        Actualiza los parámetros del fractal.
        """
        
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.formula = formula
        self.tipo_calculo = tipo_calculo
        self.tipo_fractal = tipo_fractal
        self.real = real
        self.imag = imag
        return None

    def calcular_fractal(self) -> np.ndarray:
        if self.tipo_fractal in FRACTAL_REGISTRY:
            if self.tipo_calculo in FRACTAL_REGISTRY[self.tipo_fractal]:
                M = FRACTAL_REGISTRY[self.tipo_fractal][self.tipo_calculo](self)
                return M
            else:
                raise ValueError(f"Tipo de cálculo '{self.tipo_calculo}' no soportado para el fractal '{self.tipo_fractal}'.")

    @staticmethod
    def convertir_formula_compleja(formula: str)-> tuple [str, str]:
        """
        Convierte una fórmula compleja como 'z**2 + C' en dos fórmulas para partes reales e imaginarias,
        usando variables zr, zi, Cr, Ci.
        """
        # Solo soportamos polinomios y suma con C por ahora.
        if formula.strip() == "z**2 + C":
            # (zr + i zi)^2 = (zr^2 - zi^2) + i(2*zr*zi)
            real_expr = "zr**2 - zi**2 + Cr"
            imag_expr = "2 * zr * zi + Ci"
            return real_expr, imag_expr
        elif formula.strip() == "z**2 + 0":  # Julia con constante embebida
            real_expr = "zr**2 - zi**2"
            imag_expr = "2 * zr * zi"
            return real_expr, imag_expr
        else:
            raise NotImplementedError(f"Fórmula no soportada todavía: {formula}")
    
    @staticmethod
    def transformar_expresion(expression: str, variables: str, mask_name :str ="matriz") -> str:
        """
        Aplica una máscara a las variables en la expresión.
        """
        for var in variables:
            expression = expression.replace(var, f"{var}[{mask_name}]")
        return expression

    ###################################
    # Métodos de cálculo de fractales #
    ###################################
    @register_fractal("Mandelbrot", "CPU_cpp_entrada")
    @medir_tiempo("Mandelbrot CPP (Entrada)")
    def hacer_mandelbrot_entrada_cpp(self) -> np.ndarray:
        """
        Llama a la DLL de Mandelbrot dinámica, pasándole self.formula
        (por ejemplo "z**3 + c**2 + 2*c**3") para que el C++ la parseé.
        """
        dll_path = r"codigos_cpp\mandelbrot_dynamic.dll"
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"No se encuentra la DLL en {dll_path}")

        # Carga la DLL
        lib = ctypes.WinDLL(dll_path)

        # Mandamos:
        #  xmin, xmax, ymin, ymax: double
        #  width, height, max_iter: int
        #  formula: const char*
        lib.mandelbrot.argtypes = [
            ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,
            ctypes.c_int,    ctypes.c_int,
            ctypes.c_int,    ctypes.c_char_p
        ]
        lib.mandelbrot.restype = ctypes.POINTER(ctypes.c_int)

        lib.free_mandelbrot.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.free_mandelbrot.restype  = None

        # Prepara la fórmula como bytes UTF-8
        formula_bytes = self.formula.encode('utf-8')

        # Llamada al DLL
        M_ptr = lib.mandelbrot(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter,
            ctypes.c_char_p(formula_bytes)
        )
        if not M_ptr:
            raise RuntimeError("mandelbrot() devolvió nullptr")

        # Convierte el buffer en un ndarray y libera la memoria C++
        flat = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
        M = flat.copy().reshape(self.height, self.width)
        lib.free_mandelbrot(M_ptr)

        return M
    
#    @register_fractal("Mandelbrotfast", "GPU_Cupy_kernel_fast")
    @medir_tiempo("Mandelbrot CPP (Fast)")
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

    @register_fractal("Mandelbrotprecision", "cpp_precision")
    @medir_tiempo("Mandelbrot CPP (Precision)")
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
    
   # @register_fractal("Mandelbrot", "GPU_Cupy_kernel_smooth")
    @medir_tiempo("Mandelbrot GPU (Smooth Coloring)")
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

    @medir_tiempo("Mandelbrot Entrada")
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
    
    #@register_fractal("Mandelbrot", "CPU_Numpy")
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
    
    #@register_fractal("Mandelbrot", "GPU_Cupy")
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
    @register_fractal("Mandelbrot", "GPU_Cupy_kernel")
    @medir_tiempo("Mandelbrot GPU")
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
    
    
    @register_fractal("Mandelbrot", "CPU_cpp")
    @medir_tiempo("Mandelbrot CPP")
    def hacer_mandelbrot_cpp(self) -> np.ndarray:
        ptr = mandelbrot_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_mandelbrot(ptr)
        return img
    
    ################
    # Para olvidar #
    ################
    
    @medir_tiempo("Mandelbrot C")
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

    #########
    # Julia #
    #########
    
    #@register_fractal("Julia", "CPU_Numpy")
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
    
    #@register_fractal("Julia", "GPU_Cupy")
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
    
    @register_fractal("Julia", "CPU_cpp")
    @medir_tiempo("Julia CPP")
    def hacer_julia_cpp(self) -> np.ndarray:
        ptr = julia_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter,
            self.real,   
            self.imag    
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_julia(ptr)
        return img

    ################
    # Burning Ship #
    ################
    
    #@register_fractal("Burning Ship", "CPU_Numpy")
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
    
    #@register_fractal("Burning Ship", "GPU_Cupy")
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
    
    @register_fractal("Burning Ship", "CPU_cpp")
    @medir_tiempo("Burning Ship CPP")
    def hacer_burning_cpp2(self) -> np.ndarray:
        ptr = burning_ship_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_burning_ship(ptr)
        return img
    
    ###########
    # Tricorn #
    ###########

    #@register_fractal("Tricorn", "CPU_Numpy")
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
        
    #@register_fractal("Tricorn", "GPU_Cupy")
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
    
    @register_fractal("Tricorn", "CPU_cpp")
    @medir_tiempo("Tricorn CPP")
    def hacer_tricorn_cpp(self) -> np.ndarray:
        ptr = tricorn_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_tricorn(ptr)
        return img
    
    ###########
    # Circulo #
    ###########
    
    #@register_fractal("Circulo", "CPU_Numpy")
    def hacer_circulo_numpy(self) -> np.ndarray:
        inicio = time.time()

        x = np.linspace(self.xmin, self.xmax, self.width)
        y = np.linspace(self.ymin, self.ymax, self.height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        z = np.zeros_like(C, dtype=np.complex128)
        M = np.zeros(C.shape, dtype=int)
        matriz = np.ones(C.shape, dtype=bool)

        for n in range(self.max_iter):
            z[matriz] = np.exp((z[matriz]**2 - 1.00001*z[matriz]) / C[matriz]**4) 
            matriz = np.logical_and(matriz, np.abs(z) <= 2)
            M[matriz] = n
            print(f"\rCIRCULO {n}", end="", flush=True)

        fin = time.time()
        print("\nTiempo de ejecución:", fin - inicio, "segundos")
        return M 
    
    #@register_fractal("Circulo", "GPU_Cupy")
    def hacer_circulo_cupy(self) -> np.ndarray:
        inicio = time.time()

        x = cp.linspace(self.xmin, self.xmax, self.width)
        y = cp.linspace(self.ymin, self.ymax, self.height)
        X, Y = cp.meshgrid(x, y)
        C = X + 1j * Y
        z = cp.zeros_like(C, dtype=cp.complex128)
        M = cp.zeros(C.shape, dtype=int)
        matriz = cp.ones(C.shape, dtype=bool)

        for n in range(self.max_iter):
            z[matriz] = cp.exp((z[matriz]**2 - 1.00001*z[matriz]) / C[matriz]**4) 
            matriz = cp.logical_and(matriz, cp.abs(z) <= 2)
            M[matriz] = n
            print(f"\rCIRCULO {n}", end="", flush=True)

        fin = time.time()
        print("\nTiempo de ejecución:", fin - inicio, "segundos")
        return M.get()
    
    @register_fractal("Circulo", "GPU_Cupy_kernel")
    def hacer_circulo_gpu(self) -> np.ndarray:
        inicio = time.time()

        x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
        y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
        X, Y = cp.meshgrid(x, y)
        C = X + 1j * Y  
        C = C.ravel()   
        Z = cp.zeros_like(C, dtype=cp.complex128)  

        resultado = cp.empty(C.shape, dtype=cp.int32)

        try:
            circulo_kernel(Z, C, self.max_iter, resultado)
        except Exception as e:
            print(f"Error executing Circulo kernel: {e}")
            return None

        resultado = resultado.reshape((self.height, self.width))
        resultado_cpu = resultado.get()

        tiempo = time.time() - inicio
        print(f"{self.max_iter} iteraciones")
        print(f"Tiempo total: {tiempo:.5f} segundos")

        return resultado_cpu

    @register_fractal("Circulo", "CPU_cpp")
    @medir_tiempo("Circulo CPP")
    def hacer_circulo_cpp2(self) -> np.ndarray:
        ptr = circulo_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_circulo(ptr)
        return img
    
    ##################
    # Newton-Raphson #
    ##################
    
    #@register_fractal("Newton-Raphson", "CPU_Numpy")
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
    
    #@register_fractal("Newton-Raphson", "GPU_Cupy")
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

    @register_fractal("Newton-Raphson", "CPU_cpp")
    @medir_tiempo("Newton CPP")
    def hacer_newton_cpp2(self) -> np.ndarray:
        ptr = newton_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_newton(ptr)
        return img
    
    ###########
    # Phoenix #
    ###########

    #@register_fractal("Phoenix", "CPU_Numpy")
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
    
    #@register_fractal("Phoenix", "GPU_Cupy")
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
    @medir_tiempo("Phoenix GPU")
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
    @medir_tiempo("Phoenix CPP")
    def hacer_phoenix_cpp(self) -> np.ndarray:
        ptr = phoenix_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter,
            self.real,  
            self.imag    
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_phoenix(ptr)
        return img
    
    #################
    # Burning Julia #
    #################

    #register_fractal("Burning Julia", "CPU_Numpy")
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

    #@register_fractal("Burning Julia", "GPU_Cupy")
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
    @medir_tiempo("Burning Julia GPU")
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
    @medir_tiempo("Burning Julia CPP")
    def hacer_burning_julia_cpp(self) -> np.ndarray:
        ptr = burning_julia_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter,
            self.real,  
            self.imag    
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_burning_julia(ptr)
        return img
    
    #####################
    # Celtic Mandelbrot #
    #####################
    
   # @register_fractal("Celtic Mandelbrot", "CPU_Numpy")
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
    
    #@register_fractal("Celtic Mandelbrot", "GPU_Cupy")
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
    @medir_tiempo("Celtic Mandelbrot GPU")
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
    @medir_tiempo("Celtic Mandelbrot CPP")
    def hacer_celtic_mandelbrot_cpp2(self) -> np.ndarray:
        ptr = celtic_mandelbrot_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_celtic_mandelbrot(ptr)
        return img
    
    ########
    # Nova #
    ########

    #@register_fractal("Nova", "CPU_Numpy")
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
    
    #@register_fractal("Nova", "GPU_Cupy")
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
    @medir_tiempo("Nova GPU")
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
    @medir_tiempo("Nova CPP")
    def hacer_nova_cpp(self) -> np.ndarray:
        ptr = nova_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter,
            self.imag,  
            self.real   
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_nova(ptr)
        return img
    
   # @register_fractal("Julia-Gamma", "CPU_Numpy")
    def hacer_julia_gamma(self) -> np.ndarray:
        """
        Julia con función Gamma: z_{n+1} = gamma(z_n)^2 + c
        """
        inicio = time.time()

        x = np.linspace(self.xmin, self.xmax, self.width)
        y = np.linspace(self.ymin, self.ymax, self.height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        z = np.zeros(C.shape, dtype=np.complex128)
        M = np.zeros(C.shape, dtype=int)
        matriz = np.ones(C.shape, dtype=bool)

        c = self.real + 1j*self.imag

        for n in range(self.max_iter):
            # Calculá gamma(z) para los que aún no escaparon
            z_act = z[matriz]
            try:
                z_next = gamma(z_act) ** 2 + c
            except Exception as e:
                # Si gamma falla (por valores no válidos), poné infinito
                z_next = np.full_like(z_act, np.inf, dtype=np.complex128)
            z[matriz] = z_next
            matriz = np.logical_and(matriz, np.abs(z) <= 100)
            M[matriz] = n
            print(f"\rJULIA-GAMMA {n}", end="", flush=True)

        fin = time.time()
        print("\nTiempo de ejecución:", fin - inicio, "segundos")
        return M

    #@register_fractal("Mandelbrot-Gamma", "CPU_Numpy")
    def hacer_mandelbrot_gamma(self) -> np.ndarray:
        """
        Mandelbrot con función Gamma: z_{n+1} = gamma(z_n)^2 + c
        """
        inicio = time.time()

        x = np.linspace(self.xmin, self.xmax, self.width)
        y = np.linspace(self.ymin, self.ymax, self.height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y   # <--- Ahora c depende de cada punto
        z = np.zeros(C.shape, dtype=np.complex128)
        M = np.zeros(C.shape, dtype=int)
        matriz = np.ones(C.shape, dtype=bool)

        for n in range(self.max_iter):
            z_act = z[matriz]
            c_act = C[matriz]
            try:
                z_next = gamma(z_act) + c_act
            except Exception as e:
                z_next = np.full_like(z_act, np.inf, dtype=np.complex128)
            z[matriz] = z_next

            matriz = np.logical_and(matriz, np.abs(z) <= 100)  # Mantener solo los que no escaparon
            M[matriz] = n
            print(f"\rMANDELBROT-GAMMA {n}", end="", flush=True)

        fin = time.time()
        print("\nTiempo de ejecución:", fin - inicio, "segundos")
        return M

    #@register_fractal("Julia-GPU", "GPU_Cupy")
    @medir_tiempo("Julia GPU")
    def hacer_julia_cupy2(self) -> np.ndarray:
        inicio = time.time()
        x = cp.linspace(self.xmin, self.xmax, self.width)
        y = cp.linspace(self.ymin, self.ymax, self.height)
        X, Y = cp.meshgrid(x, y)
        C = X + 1j * Y
        z = cp.zeros(C.shape, dtype=cp.complex128)
        M = cp.zeros(C.shape, dtype=int)
        matriz = cp.ones(C.shape, dtype=bool)
        a = self.real
        b = self.imag
        for n in range(self.max_iter):
            z[matriz]=a*z[matriz]**2+C[matriz] + b*cp.exp((z[matriz]**2 - 1.00001*z[matriz]) / C[matriz]**4)
            matriz = cp.logical_and(matriz, cp.abs(z) <= 2)
            M[matriz] = n
            print(f"\rJULIA {n}", end="", flush=True)
        fin = time.time()
        print("\nTiempo de ejecución:", fin - inicio, "segundos")
        return M.get()

    @register_fractal("Julia-GPU", "GPU_Cupy_kernel")
    @medir_tiempo("Julia GPU")
    def hacer_julia_cupy_kernel_2(self) -> np.ndarray:
        x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
        y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
        X, Y = cp.meshgrid(x, y)
        C = (X + 1j * Y).ravel()

        resultado = cp.empty(C.shape, dtype=cp.int32)
        try:
            julia_custom_kernel(C, float(self.real), float(self.imag), self.max_iter, resultado)
        except Exception as e:
            print(f"Error ejecutando kernel Julia: {e}")
            return None

        resultado = resultado.reshape((self.height, self.width))
        return resultado.get()
    
    @register_fractal("Mandelbrot-Fortran", "CPU_Fortran")
    @medir_tiempo("Mandelbrot Fortran")
    def hacer_mandelbrot_fortran(self) -> np.ndarray:
        # Llamada al módulo Fortran: devuelve directamente la matriz M
        M = mandelbrot.mandelbrot(
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.width,
            self.height,
            self.max_iter
        )
        return M
    
    ##################
    # Coseno Fractal #
    ##################
    
    #@register_fractal("Coseno Fractal", "CPU_Numpy")
    def hacer_cos_numpy(self) -> np.ndarray:
        """
        Coseno fractal: z_{n+1} = cos(z_n) + C
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
            z[mask] = np.cos(z[mask]/C[mask]) 
            escaped = np.abs(z) > 2.0
            just_escaped = mask & escaped
            M[just_escaped] = n
            mask[just_escaped] = False

            print(f"\rCOSENO {n}", end="", flush=True)
            if not mask.any():
                break

        fin = time.time()
        print("\nTiempo de ejecución:", fin - inicio, "segundos")
        return M
    
    #@register_fractal("Coseno Fractal", "GPU_Cupy")
    def hacer_cos_cupy(self) -> np.ndarray:
        """
        Coseno fractal: z_{n+1} = cos(z_n) + C
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
            z[mask] = cp.cos(z[mask]/C[mask]) 
            escaped = cp.abs(z) > 2.0
            just_escaped = mask & escaped
            M[just_escaped] = n
            mask[just_escaped] = False

            print(f"\rCOSENO {n}", end="", flush=True)
            if not mask.any():
                break

        fin = time.time()
        print("\nTiempo de ejecución:", fin - inicio, "segundos")
        return M.get()
    
    @register_fractal("Coseno Fractal", "GPU_Cupy_kernel")
    @medir_tiempo("Coseno Fractal GPU")
    def hacer_cos_cupy_kernel(self) -> np.ndarray:
        x = cp.linspace(self.xmin, self.xmax, self.width)
        y = cp.linspace(self.ymin, self.ymax, self.height)
        X, Y = cp.meshgrid(x, y)
        C = X + 1j * Y

        resultado = cp.empty(C.shape, dtype=cp.int32)
        try:
            coseno_kernel(C, self.max_iter, resultado)
        except Exception as e:
            print(f"Error ejecutando kernel Coseno: {e}")
            return None

        return resultado.get()
    
    @register_fractal("Coseno Fractal", "CPU_cpp")
    @medir_tiempo("Coseno Fractal CPP")
    def hacer_cos_cpp2(self) -> np.ndarray:
        ptr = coseno_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_coseno(ptr)
        return img
    
    ######################
    # Coseno+1/C Fractal #
    ######################
    
    #@register_fractal("Coseno+1/C Fractal", "CPU_Numpy")
    def hacer_cos_inv_numpy(self) -> np.ndarray:
        """
        Fractal: z_{n+1} = cos(z_n) + 1/C
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
            z[mask] = np.cos(z[mask]) + 1.0/C[mask]
            esc = np.abs(z) > 16.0
            js = mask & esc
            M[js] = n
            mask[js] = False
            if not mask.any(): break
        print("⏱", time.time() - inicio, "segundos")
        return M


    #@register_fractal("Coseno+1/C Fractal", "GPU_Cupy")
    def hacer_cos_inv_cupy(self) -> np.ndarray:
        """
        Fractal: z_{n+1} = cos(z_n) + 1/C en GPU
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
            z[mask] = cp.cos(z[mask]) + 1.0/C[mask]
            esc = cp.abs(z) > 16.0
            js = mask & esc
            M[js] = n
            mask[js] = False
            if not mask.any(): break
        print("⏱", time.time() - inicio, "segundos")
        return M.get()
    
    @register_fractal("Coseno+1/C Fractal", "GPU_Cupy_kernel")
    @medir_tiempo("Coseno+1/C Fractal GPU")
    def hacer_cos_inv_cupy_kernel(self) -> np.ndarray:
        x = cp.linspace(self.xmin, self.xmax, self.width)
        y = cp.linspace(self.ymin, self.ymax, self.height)
        X, Y = cp.meshgrid(x, y)
        C = X + 1j * Y

        resultado = cp.empty(C.shape, dtype=cp.int32)
        try:
            cos_inv_kernel(C, self.max_iter, resultado)
        except Exception as e:
            print(f"Error ejecutando kernel: {e}")
            return np.zeros(C.shape, dtype=np.int32)
        return resultado.get()

    @register_fractal("Coseno+1/C Fractal", "CPU_cpp")
    @medir_tiempo("Coseno+1/C Fractal CPP")
    def hacer_cos_inv_cpp2(self) -> np.ndarray:
        ptr = coseno_inv_cpp(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter
        )
        flat = np.ctypeslib.as_array(ptr, shape=(self.width*self.height,))
        img = flat.copy().reshape(self.height, self.width)
        free_coseno_inv(ptr)
        return img

    #@register_fractal("Julia Corazón", "CPU_Numpy")
    @medir_tiempo("Julia Corazón CPU")
    def julia_corazon(self):
        x = np.linspace(self.xmin, self.xmax, self.width)
        y = np.linspace(self.ymin, self.ymax, self.height)
        X, Y = np.meshgrid(x, y)
        z = X + 1j*Y
        c = -0.123 + 0.745j

        M = np.zeros(z.shape, int)
        mask = np.ones(z.shape, bool)
        for n in range(self.max_iter):
            z[mask] = z[mask]**2 + c
            escaped = np.abs(z) > 2
            M[mask & escaped] = n
            mask &= ~escaped
            if not mask.any():
                break
        return M

