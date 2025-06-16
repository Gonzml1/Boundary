import numpy as np
import matplotlib.pyplot as plt

def domain_coloring_iteradd(xmin, xmax, ymin, ymax, width, height, iterations):
    """
    Domain‐coloring de la iteración z -> z^2 + z0,
    donde z0 es la coordenada compleja inicial de cada píxel.
    """
    # 1) rejilla de puntos
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    Z0 = X + 1j * Y    # valor “particular” de cada punto

    # 2) iteramos
    Z = Z0.copy()
    for _ in range(iterations):
        Z = Z**2 + Z0

    # 3) domain‐coloring: tono según argumento, brillo según módulo
    H = (np.angle(Z) + np.pi) / (2 * np.pi)        # normalizo a [0,1]
    L = 1 - 1/(1 + np.log1p(np.abs(Z)))            # más brillo cuanto mayor |Z|

    # 4) construyo la imagen y la muestro
    img = plt.cm.hsv(H)[:, :, :3] * L[:, :, None]
    plt.figure(figsize=(6,6))
    plt.imshow(img, extent=(xmin, xmax, ymin, ymax))
    plt.axis('off')
    plt.show()


# Ejemplo de uso:
domain_coloring_iteradd(
    xmin=-2, xmax=2,
    ymin=-2, ymax=2,
    width=2000, height=2000,
    iterations=256
)