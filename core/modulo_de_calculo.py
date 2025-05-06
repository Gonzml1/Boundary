import mpmath as mp
import cupy as cp
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import time 

# cp.exp((z[mascara]**2 - 1.00001*z[mascara]) / C[mascara]**4) 
# z[mascara] = z[mascara]**2 + C[mascara]    



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

def hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter):
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

def transformar_expresion(expression, variables, mask_name="mascara"):
    for var in variables:
        expression = expression.replace(var, f"{var}[{mask_name}]")
    return expression

def hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax, width, height, max_iter,formula):
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

def hacer_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    mascara = cp.ones(C.shape, dtype=bool)
    
    for n in range(max_iter):
        z[mascara] = z[mascara]**2 + C[mascara]
        mascara = cp.logical_and(mascara, cp.abs(z) <= 2)
        M[mascara] = n
        print(f"\rMANDELBROT {n}", end="", flush=True)
    fin = time.time()  # Fin del cronómetro
    return M.get()

def mandelbrot_gpu_cp(C, max_iter):
    z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    mascara = cp.ones(C.shape, dtype=bool)
    
    for n in range(max_iter):
        z[mascara] = cp.exp((z[mascara]**2 - 1.00001*z[mascara]) / C[mascara]**4) 
        mascara = cp.logical_and(mascara, cp.abs(z) <= 2)
        M[mascara] = n
        print(f"\rMANDELBROT {n}", end="", flush=True)
    return M.get()

def guardar_mandelbrot(M,xmin,xmax,ymin,ymax,filepath, width, height, cmap1,dpi ):
    figsize = ((width) / dpi, height / dpi)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis('off')
    
    ax.imshow(M, extent=(xmin, xmax, ymin, ymax), cmap=cmap1)
    
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0, format="png")
    plt.close()