import numpy as np
import matplotlib.pyplot as plt

def generar_muestra(nx, ny, muestras):
    """
    Genera 'muestras' puntos complejos uniformes en el plano [-2,1] x [-1.5,1.5].
    """
    xs = np.random.uniform(-2, 1, size=muestras)
    ys = np.random.uniform(-1.5, 1.5, size=muestras)
    return xs + 1j * ys

import numpy as np

def buddhabrot_np(muestras, max_iter, width, height):
    """
    Buddhabrot acelerado: usa numpy para mapear y acumular la órbita completa
    de cada punto escapado de una sola vez.
    """
    hist = np.zeros((height, width), dtype=np.uint32)
    
    # 1) generar puntos c
    xs = np.random.uniform(-2, 1, size=muestras)
    ys = np.random.uniform(-1.5, 1.5, size=muestras)
    cs = xs + 1j * ys
    
    for c in cs:
        z = 0 + 0j
        orbit = []
        for _ in range(max_iter):
            z = z*z + c
            orbit.append(z)
            if abs(z) > 2:
                # 2) convertir orbit (lista de complejos) en array
                orbit_arr = np.array(orbit)
                
                # 3) vectorizar el mapeo real→i, imag→j
                i = ((orbit_arr.real + 2) / 3 * (width - 1)).astype(int)
                j = ((orbit_arr.imag + 1.5) / 3 * (height - 1)).astype(int)
                
                # 4) quedarnos sólo con los índices válidos
                valid = (i >= 0) & (i < width) & (j >= 0) & (j < height)
                i, j = i[valid], j[valid]
                
                # 5) acumular en el histograma de una sola llamada
                np.add.at(hist, (j, i), 1)
                break

    return hist


# Parámetros de ejemplo (ajustables según tus necesidades)
width, height = 400, 300
muestras = 1000000
max_iter = 200

# Cálculo y plot
H = buddhabrot_np(muestras, max_iter, width, height)

plt.figure(figsize=(8, 6))
plt.imshow(H, extent=(-2, 1, -1.5, 1.5), origin='lower')
plt.title("Buddhabrot")
plt.xlabel("Re")
plt.ylabel("Im")
plt.tight_layout()
plt.show()
