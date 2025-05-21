#import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time 
from OpenGL.GL import *
#from .funciones_kernel import *
import os
from functools import wraps
from PyQt5.QtWidgets import QFileDialog

# cp.exp((z[matriz]**2 - 1.00001*z[matriz]) / C[matriz]**4) 
# z[matriz] = z[matriz]**2 + C[matriz]    

class calculos_mandelbrot:
    def __init__(self, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag):
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
        self.fractales= {
        "Mandelbrot" :  {"GPU_Cupy": self.hacer_mandelbrot_cupy, "GPU_Cupy_kernel": self.hacer_mandelbrot_gpu, "CPU_Numpy": self.hacer_mandelbrot_numpy,"CPU_cpp": self.hacer_mandelbrot_cpp
        },
        "Julia":        {"GPU_Cupy": self.hacer_julia_cupy, "GPU_Cupy_kernel": self.hacer_julia_gpu, "CPU_Numpy": self.hacer_julia_numpy, "CPU_cpp": self.hacer_julia_cpp
        },
        "Burning Ship": {"GPU_Cupy": self.hacer_burning_cupy, "GPU_Cupy_kernel": self.hacer_burning_gpu, "CPU_Numpy": self.hacer_burning_numpy, "CPU_cpp": self.hacer_burning_cpp
        }, 
        "Tricorn":      {"GPU_Cupy" : self.hacer_tricorn_cupy, "GPU_Cupy_kernel" : self.hacer_tricorn_gpu,"CPU_Numpy" : self.hacer_tricorn_numpy, "CPU_cpp": self.hacer_tricorn_cpp
        },
        "Circulo":      {"GPU_Cupy" : self.hacer_circulo_cupy, "GPU_Cupy_kernel" : self.hacer_circulo_gpu,"CPU_Numpy" : self.hacer_circulo_numpy, "CPU_cpp": self.hacer_circulo_cpp
        }
        }
        
    def guardar_imagen(self):
        ruta, _ = QFileDialog.getSaveFileName(
            None,
            "Guardar imagen",
            "fractal.png",
            "Imágenes PNG (*.png);;JPEG (*.jpg *.jpeg);;Todos los archivos (*)"
        )
        
        if ruta:
            # Reemplazá esto por tu lógica de fractal real
            imagen_array = self.calcular_fractal()
            plt.imsave(ruta, imagen_array,cmap='twilight_shifted')
            print(f"Imagen guardada en: {ruta}")

    def medir_tiempo(nombre):
        def decorador(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                inicio = time.time()
                resultado = func(*args, **kwargs)
                fin = time.time()
                print(f"⏱️ Tiempo de ejecución de '{nombre}': {fin - inicio:.5f} segundos")
                return resultado
            return wrapper
        return decorador
        
    def actualizar_fractal(self, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag):
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
        
    def calcular_fractal(self):
        if self.tipo_fractal in self.fractales:
            if self.tipo_calculo in self.fractales[self.tipo_fractal]:
                M = self.fractales[self.tipo_fractal][self.tipo_calculo]()
                return M
            else:
                raise ValueError(f"Tipo de cálculo '{self.tipo_calculo}' no soportado para el fractal '{self.tipo_fractal}'.")
        else:
            raise ValueError(f"Fractal '{self.tipo_fractal}' no soportado.")

    @staticmethod
    def transformar_expresion(expression, variables, mask_name="matriz"):
        for var in variables:
            expression = expression.replace(var, f"{var}[{mask_name}]")
        return expression
    
    def guardar_mandelbrot(self, M,filepath, cmap1, dpi):
        figsize = ((self.width) / dpi, self.height / dpi)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.axis('off')
        
        ax.imshow(M , extent=(self.xmin, self.xmax, self.ymin, self.ymax), cmap=cmap1)
        
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0, format="png")
        plt.close()
        return None
    
    def hacer_mandelbrot_gpu(self):
        inicio = time.time()
        
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
        tiempo = time.time() - inicio
        print(f"{self.max_iter} iteraciones")
        print(f"Tiempo total: {tiempo:.5f} segundos")

        return resultado_cpu
    
    def hacer_mandelbrot_con_entrada(self):
        inicio=time.time()

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

        fin = time.time() 
        print("\nTiempo de ejecución:", fin - inicio, "segundos")
        return M.get()

    
    def hacer_mandelbrot_cupy(self):
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

    def hacer_mandelbrot_numpy(self):
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
    
    
    ##############################################################
    @medir_tiempo("Mandelbrot CPP")
    def hacer_mandelbrot_cpp(self):
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
 
    ###################################################################
    def hacer_julia_gpu(self):
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
    
    def hacer_julia_cupy(self):
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
    
    def hacer_julia_numpy(self):
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
    
    @medir_tiempo("Julia CPP")
    def hacer_julia_cpp(self):
        dll_path = r"codigos_cpp\julia.dll"
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
        lib.julia.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        lib.julia.restype = ctypes.POINTER(ctypes.c_int)
        lib.free_julia.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.free_julia.restype = None
        M_ptr = lib.julia(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
        M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
        M_copy = np.copy(M).reshape(self.height, self.width)
        lib.free_julia(M_ptr)
        return M_copy
        
    
    def hacer_burning_gpu(self):
        inicio = time.time()

        x = cp.linspace(self.xmin, self.xmax, self.width, dtype=cp.float64)
        y = cp.linspace(self.ymin, self.ymax, self.height, dtype=cp.float64)
        X, Y = cp.meshgrid(x, y)
        C = X + 1j * Y  
        C = C.ravel()   
        Z = cp.zeros_like(C, dtype=cp.complex128)  

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
    
    def hacer_burning_cupy(self):
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
    
    def hacer_burning_numpy(self):
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
    
    @medir_tiempo("Burning Ship CPP")
    def hacer_burning_cpp(self):
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
    
    def hacer_tricorn_gpu(self):
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
    
    def hacer_tricorn_cupy(self):
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
    
    def hacer_tricorn_numpy(self):
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
    
    @medir_tiempo("Tricorn CPP")
    def hacer_tricorn_cpp(self):
        dll_path = r"codigos_cpp\tricorn.dll"
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
        lib.tricorn.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        lib.tricorn.restype = ctypes.POINTER(ctypes.c_int)
        lib.free_tricorn.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.free_tricorn.restype = None
        M_ptr = lib.tricorn(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
        M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
        M_copy = np.copy(M).reshape(self.height, self.width)
        lib.free_tricorn(M_ptr)
        return M_copy
    
    
    def hacer_circulo_gpu(self):
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
    
    
    def hacer_circulo_cupy(self):
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
    
    def hacer_circulo_numpy(self):
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
    
    
    @medir_tiempo("Circulo CPP")
    def hacer_circulo_cpp(self):
        dll_path = r"codigos_cpp\circulo.dll"
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
        lib.circulo.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        lib.circulo.restype = ctypes.POINTER(ctypes.c_int)
        lib.free_circulo.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.free_circulo.restype = None
        M_ptr = lib.circulo(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
        M = np.ctypeslib.as_array(M_ptr, shape=(self.height * self.width,))
        M_copy = np.copy(M).reshape(self.height, self.width)
        lib.free_circulo(M_ptr)
        return M_copy
    
    
    
