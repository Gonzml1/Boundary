import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time 
from OpenGL.GL import *
from .funciones_kernel import *

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
        self.fractales= fractales = {
    
    "Mandelbrot" :{ "GPU_Cupy": self.hacer_mandelbrot_cupy, "GPU_Cupy_kernel": self.hacer_mandelbrot_gpu, "CPU_Numpy": self.hacer_mandelbrot_numpy,
    },
    "Julia":{       "GPU_Cupy": self.hacer_julia_cupy,"GPU_Cupy_kernel": self.hacer_julia_gpu,"CPU_Numpy": self.hacer_julia_numpy 
    },
    "Burning Ship":{"GPU_Cupy": self.hacer_burning_cupy,"GPU_Cupy_kernel": self.hacer_burning_gpu, "CPU_Numpy": self.hacer_burning_numpy
    }, 
    "Tricorn":{     "GPU_Cupy" : self.hacer_tricorn_cupy,"GPU_Cupy_kernel" : self.hacer_tricorn_gpu,"CPU_Numpy" : self.hacer_tricorn_numpy
    },
    "Circulo":{     "GPU_Cupy" : self.hacer_circulo_cupy,"GPU_Cupy_kernel" : self.hacer_circulo_gpu,"CPU_Numpy" : self.hacer_circulo_numpy
    }
    }
        
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
    
    def guardar_mandelbrot(self, M, filepath, cmap1, dpi):
        figsize = ((self.width) / dpi, self.height / dpi)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.axis('off')
        
        ax.imshow(M, extent=(self.xmin, self.xmax, self.ymin, self.ymax), cmap=cmap1)
        
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
        z[matriz] = z[matriz]**2 + C[matriz]

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
        z[matriz] = z[matriz]**2 + C[matriz]

        for n in range(self.max_iter):
            z[matriz]=z[matriz]**2+(self.real+1j*self.imag)
            matriz = np.logical_and(matriz, np.abs(z) <= 2)
            M[matriz] = n
            print(f"\rJULIA {n}", end="", flush=True)

        fin = time.time()
        print("\nTiempo de ejecución:", fin - inicio, "segundos")
        return M
    
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
            z_temp = (z.real**2 + z.imag**2) + C
            z[matriz] = z_temp[matriz]
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
            z_temp = (z.real**2 + z.imag**2) + C
            z[matriz] = z_temp[matriz]
            matriz = np.logical_and(matriz, np.abs(z) <= 2)
            M[matriz] = n
            print(f"\rCIRCULO {n}", end="", flush=True)

        fin = time.time()
        print("\nTiempo de ejecución:", fin - inicio, "segundos")
        return M
    
    
