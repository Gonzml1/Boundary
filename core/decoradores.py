import numpy as np
import time
from functools import wraps
import matplotlib.pyplot as plt

# 1. Función lógica: cálculo del fractal
def mandelbrot(C, max_iter=100):
    z = np.zeros_like(C, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)
    mask = np.ones(C.shape, dtype=bool)
    for n in range(max_iter):
        z[mask] = z[mask] * z[mask] + C[mask]
        mask = np.logical_and(mask, np.abs(z) <= 2)
        M[mask] = n
        print(f"\rMandelbrot {n}", end="", flush=True)
    return M

# 2. Decorador que conecta una grilla con una función de cálculo
def aplicar_logica_fractal(funcion_de_calculo, nombre="Fractal", max_iter=100):
    def decorador(funcion_grilla):
        @wraps(funcion_grilla)
        def wrapper(*args, **kwargs):
            print(f"🔧 Generando grilla para {nombre}...")
            inicio = time.time()
            C = funcion_grilla(*args, **kwargs)
            print(f"⚙ Calculando {nombre}...")
            resultado = funcion_de_calculo(C, max_iter=max_iter)
            fin = time.time()
            print(f"✅ {nombre} completado en {fin - inicio:.3f} segundos")
            return resultado
        return wrapper
    return decorador

# 3. Grilla decorada con la lógica fractal
@aplicar_logica_fractal(mandelbrot, nombre="Mandelbrot (Numpy)", max_iter=200)
def generar_grilla_compleja(xmin, xmax, ymin, ymax, width, height):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    return X + 1j * Y

# 4. Uso de la grilla decorada
if __name__ == "__main__":
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 800, 600
    resultado = generar_grilla_compleja(xmin, xmax, ymin, ymax, width, height)
    plt.imshow(resultado, extent=(xmin, xmax, ymin, ymax), cmap='hot')
    plt.colorbar()
    plt.title("Fractal de Mandelbrot")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.show()
    