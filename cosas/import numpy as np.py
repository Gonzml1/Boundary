import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros
width, height = 1000, 1000
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
max_iter = 1000

x = np.linspace(xmin, xmax, width)
y = np.linspace(ymin, ymax, height)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y
Z = np.zeros_like(C, dtype=np.complex128)
M = np.zeros(C.shape, dtype=int)
mask = np.ones(C.shape, dtype=bool)

for n in range(max_iter):
    Z[mask] = Z[mask]**2 + C[mask]
    mask_new = np.abs(Z) <= 2
    M[mask & ~mask_new] = n
    mask = mask_new
    print( f"Iteración {n+1}/{max_iter} completada", end='\r')

# Normalización a [0, 1]


# (Opcional) Si querés usar logaritmo, después de normalizar:
# M_norm = np.log1p(M) / np.log1p(M.max())

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

step = 3  # Muestra 1 de cada 3 para que sea liviano
ax.plot_surface(X[::step, ::step], Y[::step, ::step], M[::step, ::step],
                cmap='inferno', edgecolor='none', antialiased=True)

ax.set_xlabel("Re")
ax.set_ylabel("Im")
ax.set_zlabel("Altura normalizada")
ax.set_title("Mandelbrot 3D (altura normalizada)")
plt.tight_layout()
plt.show()
