import matplotlib.pyplot as plt
import numpy as np
import core.modulo_de_calculo_fractales as tf

def generar_arbol(x, y, angulo, longitud, nivel, factor=0.7, segmentos=None):
    if segmentos is None:
        segmentos = []

    if nivel == 0 or longitud < 1:
        return segmentos

    x2 = x + longitud * np.cos(angulo)
    y2 = y + longitud * np.sin(angulo)

    segmentos.append(((x, y), (x2, y2)))

    # Ramas izquierda y derecha
    generar_arbol(x2, y2, angulo + np.pi/6, longitud * factor, nivel - 1, factor, segmentos)
    generar_arbol(x2, y2, angulo - np.pi/6, longitud * factor, nivel - 1, factor, segmentos)

    return segmentos

def guardar_mandelbrot(M,xmin,xmax,ymin,ymax,filepath, width, height, cmap1,dpi ):
    figsize = ((width) / dpi, height / dpi)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis('off')
    
    ax.imshow(M, extent=(xmin, xmax, ymin, ymax), cmap=cmap1)
    
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0, format="png")
    plt.close()
 


segmentos =generar_arbol(x=0, y=0, angulo=np.pi/2, longitud=100, nivel=10)
tf.guardar_mandelbrot(segmentos,)

print(generar_arbol(x=0, y=0, angulo=np.pi/2, longitud=100, nivel=10))
