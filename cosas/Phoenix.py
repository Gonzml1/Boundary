import numpy as np
import matplotlib.pyplot as plt

# 1) Parámetros globales
#   c puede ser un escalar complejo o un array 2D del mismo tamaño que z
c = -0.5 + 0.0j           # puedes cambiarlo por array 2D si quieres Julia‐Phoenix
p = 0.566              # coeficiente real (puede ser complejo)

# 2) Creamos la malla de puntos (ejemplo para Mandelbrot‐Phoenix)
nx, ny = 800, 600
x = np.linspace(-2.0, 2.0, nx)
y = np.linspace(-1.5, 1.5, ny)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y               # array 2D de puntos c
Z  = np.zeros_like(C)        # z_n, comienza en 0
Zp = np.zeros_like(C)        # z_{n-1}, inicialmente también 0

# 3) Array para llevar cuenta de iteraciones
max_iters = 200
iters = np.zeros(C.shape, dtype=int)
mask = np.ones(C.shape, dtype=bool)   # True = sigue iterando (no ha escapado)

for i in range(max_iters):
    # a) cálculo de z_{n+1} para los puntos que aún no han escapado
    #    = Z[mask]**2 + p * Zp[mask] + C[mask]
    Z_next = Z[mask] * Z[mask] + p * Zp[mask] + C[mask]

    # b) actualizar Zp ← Z, Z ← Z_next (solo en los que están en mask)
    Zp[mask] = Z[mask]
    Z[mask]  = Z_next

    # c) comprobamos escape (|z| > 2)
    escaped = np.abs(Z) > 2.0

    # d) para quienes escaparon en esta iteración, guardamos i y sacamos de mask
    just_escaped = mask & escaped
    iters[just_escaped] = i
    mask[just_escaped] = False

    # e) si todos escaparon, podemos salir
    if not mask.any():
        break

plt.imshow(iters, cmap='hot', extent=(-2.0, 2.0, -1.5, 1.5))
plt.colorbar()
plt.title("Fractal de Phoenix")
plt.show()

import numpy as np

# 1) Definimos el parámetro c (puede ser escalar o un array 2D fijo)
c = -0.4 + 0.6j    # cambia a tu gusto

# 2) Malla de puntos z0
nx, ny = 800, 600
x = np.linspace(-1.5, 1.5, nx)
y = np.linspace(-1.5, 1.5, ny)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y        # Array 2D de valores iniciales

# 3) Arrays auxiliares
max_iters = 200
iters = np.zeros(Z.shape, dtype=int)
mask = np.ones(Z.shape, dtype=bool)

for i in range(max_iters):
    # a) Tomamos sólo los que no han escapado
    z = Z[mask]

    # b) Aplicamos |Re(z)| + i |Im(z)|  (Burning Ship)
    re = np.abs(z.real)
    im = np.abs(z.imag)
    z_abs = re + 1j * im

    # c) iteramos: z_{n+1} = (z_abs)**2 + c
    Z_next = z_abs * z_abs + c

    # d) actualizamos
    Z[mask] = Z_next

    # e) comprobamos escape
    escaped = np.abs(Z) > 2.0

    # f) registramos iteraciones de quienes escapen ahora
    just_escaped = mask & escaped
    iters[just_escaped] = i
    mask[just_escaped] = False

    if not mask.any():
        break

# 4) Ahora “iters” contiene el número de iteración de escape. Pintalo con matplotlib.
#    p.ej.: plt.imshow(iters, cmap='inferno'), etc.
