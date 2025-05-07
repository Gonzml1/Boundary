import mpmath as mp
import cupy as cp
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import time 

# cp.exp((z[mascara]**2 - 1.00001*z[mascara]) / C[mascara]**4) 
# z[mascara] = z[mascara]**2 + C[mascara]    

def calcular_fractal(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag):
    if tipo_fractal in fractales:
        if tipo_calculo in fractales[tipo_fractal]:
            M = fractales[tipo_fractal][tipo_calculo](xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag)
            return M
        else:
            raise ValueError(f"Tipo de cálculo '{tipo_calculo}' no soportado para el fractal '{tipo_fractal}'.")
    else:
        raise ValueError(f"Fractal '{tipo_fractal}' no soportado.")
    
def calcular_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, formula,tipo_calculo):
    
    if tipo_calculo in calculos:
        M = calculos[tipo_calculo](xmin, xmax, ymin, ymax, width, height, max_iter, formula)
        return M
    else:
        raise ValueError(f"Tipo de cálculo '{tipo_calculo}' no soportado.")    


def transformar_expresion(expression, variables, mask_name="mascara"):
    for var in variables:
        expression = expression.replace(var, f"{var}[{mask_name}]")
    return expression

def guardar_mandelbrot(M,xmin,xmax,ymin,ymax,filepath, width, height, cmap1,dpi ):
    figsize = ((width) / dpi, height / dpi)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis('off')
    
    ax.imshow(M, extent=(xmin, xmax, ymin, ymax), cmap=cmap1)
    
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0, format="png")
    plt.close()
 
    
#########################
#       MANDELBROT      #
#########################


mandelbrot_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='int32 result',
    operation="""
        complex<double> z = 0.0;  
        for (int i = 0; i < max_iter; ++i) {
            z = z*z + c;  
            if (real(z)*real(z) + imag(z)*imag(z) > 4.0) {
                result = i;
                return;
            }
        }
        result = max_iter;  
    """,
    name='mandelbrot_kernel'
)

def hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag):
    inicio = time.time()

    x = cp.linspace(xmin, xmax, width, dtype=cp.float64)
    y = cp.linspace(ymin, ymax, height, dtype=cp.float64)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y  
    C = C.ravel()   
    resultado = cp.empty(C.shape, dtype=cp.int32)
    mandelbrot_kernel(C, max_iter, resultado)
    resultado = resultado.reshape((height, width))
    resultado_cpu = resultado.get()
    tiempo = time.time() - inicio
    
    print(f"{max_iter} iteraciones")
    print(f"Tiempo total: {tiempo:.5f} segundos")

    return resultado_cpu

def hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag):
    inicio=time.time()
    
    operacion = transformar_expresion(formula, ["z", "C"])
    codigo = compile(operacion, "<string>", "exec")
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    mascara = cp.ones(C.shape, dtype=bool)
    
    for n in range(max_iter):
        exec(codigo)
        mascara = cp.logical_and(mascara, cp.abs(z) <= 2)
        M[mascara] = n
        print(f"\rMANDELBROT {n}", end="", flush=True)
        
    fin = time.time() 
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M.get()

def hacer_mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag):
    inicio = time.time()
    
    operacion = transformar_expresion(formula, ["z", "C"])
    codigo = compile(operacion, "<string>", "exec")
    
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)
    mascara = np.ones(C.shape, dtype=bool)
    
    for n in range(max_iter):
        exec(codigo)
        mascara = np.logical_and(mascara, np.abs(z) <= 2)
        M[mascara] = n
        print(f"\rMANDELBROT {n}", end="", flush=True)
        
    fin = time.time() 
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    
    return M

def hacer_mandelbrot_cupy(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag):
    inicio = time.time()
    
    operacion = transformar_expresion(formula, ["z", "C"])
    codigo = compile(operacion, "<string>", "exec")
    
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    mascara = cp.ones(C.shape, dtype=bool)
    
    for n in range(max_iter):
        exec(codigo)
        mascara = cp.logical_and(mascara, cp.abs(z) <= 2)
        M[mascara] = n
        print(f"\rMANDELBROT {n}", end="", flush=True)
        
    fin = time.time() 
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    
    return M.get()


#########################
#         JULIA         #
#########################
 
             
julia_kernel = cp.ElementwiseKernel(
    in_params='complex128 z, complex128 c, int32 max_iter',
    out_params='int32 result',
    operation="""
        complex<double> z_temp = z;  
        for (int i = 0; i < max_iter; ++i) {
            z_temp = z_temp * z_temp + c;  
            if (real(z_temp) * real(z_temp) + imag(z_temp) * imag(z_temp) > 4.0) {
                result = i;
                return;
            }
        }
        result = max        result = max_iter;  
    """,
    name='julia_kernel'
)

def hacer_julia_gpu(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag):
    inicio = time.time()

    # Create grid of complex numbers (initial z values)
    x = cp.linspace(xmin, xmax, width, dtype=cp.float64)
    y = cp.linspace(ymin, ymax, height, dtype=cp.float64)
    X, Y = cp.meshgrid(x, y)
    Z = X + 1j * Y  # Initial z values for Julia set
    Z = Z.ravel()   # Flatten for kernel processing
    
    # Fixed complex parameter c for Julia set
    C = cp.full(Z.shape, complex(real, imag), dtype=cp.complex128)
    
    # Array to store results
    resultado = cp.empty(Z.shape, dtype=cp.int32)
    
    # Execute the Julia kernel
    julia_kernel(Z, C, max_iter, resultado)
    
    # Reshape result to 2D grid
    resultado = resultado.reshape((height, width))
    
    # Transfer result back to CPU
    resultado_cpu = resultado.get()
    
    tiempo = time.time() - inicio
    print(f"{max_iter} iteraciones")
    print(f"Tiempo total: {tiempo:.5f} segundos")

    return resultado_cpu
    
def hacer_julia_cupy(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag):
    inicio = time.time()
    
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    matriz = cp.ones(C.shape, dtype=bool)
    z[matriz] = z[matriz]**2 + C[matriz]
    
    for n in range(max_iter):
        z[matriz]=z[matriz]**2+(real+1j*imag)
        matriz = cp.logical_and(matriz, cp.abs(z) <= 2)
        M[matriz] = n
        print(f"\rJULIA {n}", end="", flush=True)
        print(n)
    
    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M.get() 

def hacer_julia_numpy(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag):
    inicio = time.time()
    
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)
    matriz = np.ones(C.shape, dtype=bool)
    z[matriz] = z[matriz]**2 + C[matriz]
    
    for n in range(max_iter):
        z[matriz]=z[matriz]**2+(real+1j*imag)
        matriz = np.logical_and(matriz, np.abs(z) <= 2)
        M[matriz] = n
        print(f"\rJULIA {n}", end="", flush=True)
        print(n)
    
    fin = time.time()
    print("\nTiempo de ejecución:", fin - inicio, "segundos")
    return M


calculos ={}

fractales = {
    
    "Mandelbrot" :{
    
    "GPU_Cupy": hacer_mandelbrot_cupy,
    "GPU_Cupy_kernel": hacer_mandelbrot_gpu, 
    "CPU_Numpy": hacer_mandelbrot_numpy 
    },
    
    "Julia":{ 
        
    "GPU_Cupy": hacer_julia_cupy,
    "GPU_Cupy_kernel": hacer_julia_gpu,
    "CPU_Numpy": hacer_julia_numpy 
    }
    
}