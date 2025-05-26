import numpy as np
import ctypes
import os
from time import time

# 1) Cargar la librería
_here = os.path.dirname(__file__)
lib = ctypes.CDLL(os.path.join(_here, "libmandelbrot.so"))

# 2) Definir la firma de la función en Python
#    void mandelbrot_simd(const float *cx, const float *cy, int *iters,
#                         int N, int max_iter);
lib.mandelbrot_simd.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # cx
    ctypes.POINTER(ctypes.c_float),  # cy
    ctypes.POINTER(ctypes.c_int),    # iters
    ctypes.c_int,                    # N
    ctypes.c_int,                    # max_iter
]
lib.mandelbrot_simd.restype = None

# 3) Preparar datos con NumPy
N = 800 * 600
max_iter = 500

# Ejemplo: generar un grid lineal
xs = np.linspace(-2.0, 1.0, 800, dtype=np.float32)
ys = np.linspace(-1.5, 1.5, 600, dtype=np.float32)
cx = np.repeat(xs[np.newaxis, :], 600, axis=0).ravel()
cy = np.repeat(ys[:, np.newaxis], 800, axis=1).ravel()

# Asegúrate de que los arrays sean contiguos en C y del tipo correcto
cx = np.ascontiguousarray(cx, dtype=np.float32)
cy = np.ascontiguousarray(cy, dtype=np.float32)
iters = np.empty(N, dtype=np.int32)

# 4) Llamar a la función
lib.mandelbrot_simd(
    cx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    cy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    iters.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    ctypes.c_int(N),
    ctypes.c_int(max_iter)
)
star_time = time()


# 5) Volver a moldear y usar el resultado
iters = iters.reshape((600, 800))
end_time = time()
print(f"Tiempo de ejecución: {end_time - star_time:.2f} segundos")
# Ahora puedes mostrarlo con matplotlib, por ejemplo:
import matplotlib.pyplot as plt
plt.imshow(iters, cmap="inferno", extent=(-2,1,-1.5,1.5))
plt.show()



