import mpmath as mp
import cupy as cp
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import time 

# cp.exp((z[mascara]**2 - 1.00001*z[mascara]) / C[mascara]**4) 
# z[mascara] = z[mascara]**2 + C[mascara]    

#Funcion basica con uso de la gpu
def hacer():
    print("HOLA")
    return 0

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

def mandelbrot_cp2(xmin, xmax, ymin, ymax, width, height, max_iter):
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

    return M.get()

#implementa la funcion anterior
def generate_mandelbrot_gpu_cp(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    
    M = mandelbrot_gpu_cp(C, max_iter)
    return M

#Permite variar el exponente del fractal
def mandelbrot_gpu_exponente_cp(C, max_iter,exponente):
    z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    mascara = cp.ones(C.shape, dtype=bool)
    
    for n in range(max_iter):
        z[mascara] = z[mascara]**exponente + C[mascara]        
        mascara = cp.logical_and(mascara, cp.abs(z) <= 2)
        M[mascara] = n

    return M.get()

def mandelbrot_gpu_exponente_np(C, max_iter,exponente):
    z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)
    mascara = np.ones(C.shape, dtype=bool)
    
    for n in range(max_iter):
        z[mascara] = z[mascara]**exponente + C[mascara]        
        mascara = np.logical_and(mascara, np.abs(z) <= 2)
        M[mascara] = n
        
    return M

#permite dividr el fractal en matrices mas pequeñas
def generate_mandelbrot_block(xmin, xmax, ymin, ymax, width, height, max_iter, block_size):
    a = 0
    total_blocks = (height // block_size) * (width // block_size)
    full_image = np.zeros((height, width), dtype=np.uint16)
    
    start_time = time.time()  # Timer start
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            x_block_size = min(block_size, width - j)
            y_block_size = min(block_size, height - i)

            x = cp.linspace(xmin + (xmax - xmin) * j / width,
                            xmin + (xmax - xmin) * (j + x_block_size) / width,
                            x_block_size)
            y = cp.linspace(ymin + (ymax - ymin) * i / height,
                            ymin + (ymax - ymin) * (i + y_block_size) / height,
                            y_block_size)
            X, Y = cp.meshgrid(x, y)
            C = X + 1j * Y

            M_block = mandelbrot_gpu_cp(C, max_iter)
            full_image[i:i + y_block_size, j:j + x_block_size] = M_block

            a += 1
            progress = (a / total_blocks) * 100
            elapsed = time.time() - start_time  # Timer for elapsed time
            print(f'Porcentaje: {a}/{total_blocks} ({progress:.2f}%) - Time elapsed: {elapsed:.2f} seconds', flush=True)

    return full_image

#Una implementacion basica de la libreria mpmath
def mandelbrot_high_precision_for(xmin, xmax, ymin, ymax, width, height, max_iter,precision):
    # Convertir a números de alta precisión
    mp.mp.dps = precision  # Número de dígitos decimales de precisión
    x = mp.linspace(xmin, xmax, width)
    y = mp.linspace(ymin, ymax, height)
    
    M = np.zeros((height, width), dtype=int)
    
    for i in range(height):
        for j in range(width):
            c = mp.mpc(x[j], y[i])
            z = mp.mpc(0,0)
            for n in range(max_iter):
                if abs(z) > 2:
                    M[i, j] = n
                    break
                z = z**2 + c
                print(n,i,j)
    return M

#Implementa numpy para un mejor renderizado del fractal 
def mandelbrot_high_precision_np(xmin, xmax, ymin, ymax, width, height, max_iter,precision):
    # Convertir a números de alta precisión
    mp.mp.dps = precision  # Número de dígitos decimales de precisión

    x = np.array([mp.mpf(val) for val in mp.linspace(xmin, xmax, width)])
    y = np.array([mp.mpf(val) for val in mp.linspace(ymin, ymax, height)])
    
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    
    z = np.zeros(C.shape, dtype=object)
    M = np.zeros(C.shape, dtype=int)
    
    mascara = np.ones(C.shape, dtype=bool)
    
    for n in range(max_iter):
        z[mascara] = (z[mascara]**2 +C[mascara])         
        
        abs_z = np.array([mp.fabs(val) for val in z.flatten()]).reshape(z.shape)
        mascara = np.logical_and(mascara, abs_z <= 2)
        
        M[mascara] = n
        print(n)
    return M

#hace un zoom en un centro determinado centro(x,y), con una determinada relacion y con una iteracion n 
def hacer_conjunto_n(xmin, xmax,ymin,ymax,centro_x,centro_y,width, height, max_iter,precision,relacion,ruta_guardado2,n,dpi1):
    for i in range(n):
        xmin = centro_x - (centro_x - xmin) * relacion
        xmax = centro_x + (xmax - centro_x) * relacion  
        ymin = centro_y - (centro_y - ymin) * relacion  
        ymax = centro_y + (ymax - centro_y) * relacion 
    
    mandelbrot = mandelbrot_high_precision_np(xmin, xmax,ymin,ymax, width, height, max_iter,precision)
    guardar_mandelbrot(mandelbrot, ruta_guardado2, width, height, dpi1)
    #Reducir las dimensiones alrededor del punto de zoom
    print(xmin,xmax,ymin,ymax)
    print("ITERACION", n)
    return 123233

#guarda la foto en la direccion indicada
def guardar_mandelbrot(M,xmin,xmax,ymin,ymax,filepath, width, height, dpi):
    figsize = ((width) / dpi, height / dpi)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis('off')
    
    ax.imshow(M, extent=(xmin, xmax, ymin, ymax), cmap="twilight_shifted")
    
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0, format="png")
    plt.close()

#Solo muestra el fractal
def mostrar_mandelbrot(M,xmin,xmax,ymin,ymax):
    plt.figure(figsize=(16, 9))
    plt.imshow(M, extent=(xmin, xmax, ymin, ymax), cmap="twilight_shifted")
    plt.colorbar()
    plt.title('Conjunto de Mandelbrot')
    plt.show()

def mandelbrot_distance_estimator_cp(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    Z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=cp.float64)  # Para guardar la distancia

    for i in range(max_iter):
        mascara = cp.abs(Z) < 2
        Z[mascara] = Z[mascara]**2 + C[mascara]
        M[mascara] = cp.log(cp.abs(Z[mascara]))
        print(i)
    return M.get()

def mandelbrot_lyapunov_exponent_cp(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    Z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=cp.float64)

    for i in range(max_iter):
        Z = Z**2 + C
        lyapunov = cp.log(cp.abs(2 * Z + C))
        M += lyapunov
        print(i)
    return M.get()

def mandelbrot_histogram_coloring_cp(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    Z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=cp.int32)  # Para contar las iteraciones

    for i in range(max_iter):
        mascara = cp.abs(Z) < 2
        Z[mascara] = Z[mascara]**2 + C[mascara]
        M[mascara] += 1
        print(i)
    # Histograma para la coloración
    hist, bins = cp.histogram(M, bins=256)
    normalized_M = cp.interp(M, bins[:-1], cp.cumsum(hist) / cp.sum(hist))
    return normalized_M.get()

def mandelbrot_newton_raphson_cp(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    Z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=cp.float64)

    for i in range(max_iter):
        Z = Z**2 + C
        M += cp.abs(Z - C)  # Raíces atractoras
        print(i)    
    return M.get()

def mandelbrot_logarithmic_potential_cp(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    Z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=cp.float64)

    for i in range(max_iter):
        Z = Z**2 + C
        M += cp.log(cp.abs(Z))
        print(i)
    return M.get()

def mandelbrot_growth_exponent_cp(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    Z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=cp.float64)

    for i in range(max_iter):
        Z = Z**2 + C
        M += cp.log(cp.abs(Z + 1))  # Medir el crecimiento
        print(i)   
    return M.get()

def mandelbrot_periodicity_check_cp(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    Z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=cp.int32)

    for i in range(max_iter):
        mascara = cp.abs(Z) < 2
        Z[mascara] = Z[mascara]**2 + C[mascara]
        M[mascara] = i

        # Verificar periodicidad
        if i % 10 == 0:
            if cp.allclose(Z, Z**2 + C, atol=1e-8):
                break
        print(i)
    return M.get()

def mandelbrot_harmonized_interval_cp(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    Z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=cp.float64)

    for i in range(max_iter):
        Z = Z**2 + C
        M += cp.abs(Z)
        print(i)    
    M = cp.log(M + 1)  # Suavizado
    return M.get()

def mandelbrot_angle_coloring_cp(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    Z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=cp.float64)

    for i in range(max_iter):
        Z = Z**2 + C
        M += cp.angle(Z)  # Usar el ángulo para colorear
        print(i)
    return M.get()

import cupy as np

def mandelbrot_distance_estimator_np(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=np.float64)  # Para guardar la distancia

    for i in range(max_iter):
        mascara = np.abs(Z) < 2
        Z[mascara] = Z[mascara]**2 + C[mascara]
        M[mascara] = np.log(np.abs(Z[mascara]))
        print(i)
    return M

def mandelbrot_lyapunov_exponent_np(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=np.float64)

    for i in range(max_iter):
        Z = Z**2 + C
        lyapunov = np.log(np.abs(2 * Z + C))
        M += lyapunov
        print(i)
    return M

def mandelbrot_histogram_coloring_np(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=np.int32)  # Para contar las iteraciones

    for i in range(max_iter):
        mascara = np.abs(Z) < 2
        Z[mascara] = Z[mascara]**2 + C[mascara]
        M[mascara] += 1
        print(i)
    # Histograma para la coloración
    hist, bins = np.histogram(M, bins=256)
    normalized_M = np.interp(M, bins[:-1], np.cumsum(hist) / np.sum(hist))
    return normalized_M

def mandelbrot_newton_raphson_np(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=np.float64)

    for i in range(max_iter):
        Z = Z**2 + C
        M += np.abs(Z - C)  # Raíces atractoras
        print(i)    
    return M

def mandelbrot_logarithmic_potential_np(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=np.float64)

    for i in range(max_iter):
        Z = Z**2 + C
        M += np.log(np.abs(Z))
        print(i)
    return M

def mandelbrot_growth_exponent_np(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=np.float64)

    for i in range(max_iter):
        Z = Z**2 + C
        M += np.log(np.abs(Z + 1))  # Medir el crecimiento
        print(i)   
    return M

def mandelbrot_periodicity_check_np(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=np.int32)

    for i in range(max_iter):
        mascara = np.abs(Z) < 2
        Z[mascara] = Z[mascara]**2 + C[mascara]
        M[mascara] = i

        # Verificar periodicidad
        if i % 10 == 0:
            if np.allclose(Z, Z**2 + C, atol=1e-8):
                break
        print(i)
    return M

def mandelbrot_harmonized_interval_np(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=np.float64)

    for i in range(max_iter):
        Z = Z**2 + C
        M += np.abs(Z)
        print(i)    
    M = np.log(M + 1)  # Suavizado
    return M

def mandelbrot_angle_coloring_np(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros(C.shape, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=np.float64)

    for i in range(max_iter):
        Z = Z**2 + C
        M += np.angle(Z)  # Usar el ángulo para colorear
        print(i)
    return M

def transformarexpresion(expression, variables, mask_name="mask"):
    for var in variables:
        expression = expression.replace(var, f"{var}[{mask_name}]")
    return expression

def mandelbrot_cp_transformado(xmin, xmax, ymin, ymax, width, height, max_iter,expresion):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    z = cp.zeros(C.shape, dtype=cp.complex128)
    M = cp.zeros(C.shape, dtype=int)
    mask = cp.ones(C.shape, dtype=bool)
    
    for n in range(max_iter):
        z[mask] = eval(expresion)
        mask = cp.logical_and(mask, cp.abs(z) <= 2)
        M[mask] = n
##        print(f"\rMANDELBROT {n}", end="", flush=True)
    return M.get()