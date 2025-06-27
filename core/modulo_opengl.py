import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from core.modulo_de_calculo_fractales import calculos_mandelbrot
from math import sin, cos, radians
from gui.MandelbrotGUI import Ui_Boundary
from matplotlib import cm
from typing import Callable
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog
import matplotlib.pyplot as plt
PALETTE_REGISTRY: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = []

def register_palette(palette_name: str) -> Callable[[Callable[[np.ndarray], np.ndarray]], Callable[[np.ndarray], np.ndarray]]:
    """
    Decorador que registra (nombre, función) en PALETTE_REGISTRY.
    La función _aún no_ está ligada a ninguna instancia.
    """
    def deco(fn: Callable[[np.ndarray], np.ndarray]):
        PALETTE_REGISTRY.append((palette_name, fn))
        return fn
    return deco

class MandelbrotWidget(QOpenGLWidget):
    def __init__(self,cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, zoom_in, zoom_out, boundary=Ui_Boundary):
        super().__init__()
        self.cmap           =       cmap    
        self.xmin           =       xmin
        self.xmax           =       xmax
        self.ymin           =       ymin
        self.ymax           =       ymax
        self.width          =       width
        self.height         =       height
        self.max_iter       =       max_iter
        self.formula        =       formula
        self.tipo_calculo   =       tipo_calculo
        self.tipo_fractal   =       tipo_fractal
        self.real           =       real
        self.imag           =       imag
        self.ui             =       boundary
        self.zoom_in        =       zoom_in
        self.zoom_out       =       zoom_out
        self.zoom_factor    =       1.0
        self.dragging       =       False
        self.last_pos       =       None
        self.mandelbrot     =       calculos_mandelbrot(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter,self.formula, self.tipo_calculo, self.tipo_fractal, self.real, self.imag, self.ui)                               
        self.lsystem        =       None  
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.ui.boton_hacer_fractal.clicked.connect(lambda : self.update())
        self.ui.slider_iteraciones.valueChanged.connect(self.update)
        self.actualizar_parametros()
        self.palettes = []
        self.clase_equiv = self.ui.clase_equiv_entrada.text()
        for name, fn in PALETTE_REGISTRY:
            bound_fn = fn.__get__(self, type(self))
            self.palettes.append((name, bound_fn))
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.palette_index = 0
        self.ui.boton_guardar.clicked.connect(lambda: self.guardar_imagen())
        self.linkeo_botones()
    
    ######################
    # Paletas de colores #
    ######################

    # se queda, para proximamente implementar clase de equiv
    
    @register_palette("Iteraciones variables (Bandas RGB variable)")
    def _paleta_bandas_rgb_equiv(self, norm: np.ndarray) -> np.ndarray:
        """
        Paleta RGB cíclica tipo bandas con clase_equiv franjas.
        - Divide [0,1] en `self.clase_equiv` franjas.
        - Dentro de cada franja: transición lineal R→G→B→R...
        """
        iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
        mod = iters % self.clase_equiv
        pos = mod / self.clase_equiv * 3  # Escalamos a [0,3)

        r = np.where(pos < 1, pos, np.where(pos < 2, 2 - pos, 0))
        g = np.where(pos < 1, 0, np.where(pos < 2, pos - 1, 3 - pos))
        b = np.where(pos < 2, 0, pos - 2)

        rgb = np.stack([r.clip(0,1), g.clip(0,1), b.clip(0,1)], axis=-1)
        return (rgb * 255).astype(np.uint8)
    
    # se queda, para proximamente implementar clase de equiv
    @register_palette("Cian→Magenta→Amarillo")
    def _paleta_inferno(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'inferno' de Matplotlib (una paleta predefinida).
        """
        iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
        cycle = self.clase_equiv
        mod = iters % cycle
        cmap = cm.get_cmap('inferno', cycle)
        lut = (cmap(np.arange(cycle))[:, :3] * 255).astype(np.uint8)
        return lut[mod]
    
    # se queda, para proximamente implementar clase de equiv
    #@register_palette("Viridis")
    def _paleta_viridis(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'viridis' de Matplotlib (verde-amarillo-azul oscuro).
        """
        cmap = cm.get_cmap('viridis', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)  # (256,3)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]  # (H, W, 3)
    # se queda, para proximamente implementar clase de equiv
    #@register_palette("Plasma")
    def _paleta_plasma(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'plasma' de Matplotlib (magenta-naranja-amarillo).
        """
        cmap = cm.get_cmap('plasma', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    # se queda, para proximamente implementar clase de equiv
    #@register_palette("Magma")
    def _paleta_magma(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'magma' de Matplotlib (negros-rosas-rojos).
        """
        cmap = cm.get_cmap('magma', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    # se queda, para proximamente implementar clase de equiv
    #@register_palette("Cividis")
    def _paleta_cividis(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'cividis' de Matplotlib (amarillo-azul verdoso, enfoque en perceptibilidad).
        """
        cmap = cm.get_cmap('cividis', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    # se queda, para proximamente implementar clase de equiv
    #@register_palette("Coolwarm")
    def _paleta_coolwarm(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'coolwarm' de Matplotlib (azul frío a rojo cálido).
        """
        cmap = cm.get_cmap('coolwarm', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    # se queda, para proximamente implementar clase de equiv
    #@register_palette("Spring")
    def _paleta_spring(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'spring' de Matplotlib (magenta a amarillo).
        """
        cmap = cm.get_cmap('spring', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    # se queda, para proximamente implementar clase de equiv
    #@register_palette("Summer")
    def _paleta_summer(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'summer' de Matplotlib (verde claro a amarillo).
        """
        cmap = cm.get_cmap('summer', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    # se queda, para proximamente implementar clase de equiv
    #@register_palette("Autumn")
    def _paleta_autumn(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'autumn' de Matplotlib (rojo a amarillo).
        """
        cmap = cm.get_cmap('autumn', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    # se queda, para proximamente implementar clase de equiv
    #@register_palette("Winter")
    def _paleta_winter(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'winter' de Matplotlib (verde azulado a azul).
        """
        cmap = cm.get_cmap('winter', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    #@register_palette("Jet")
    def _paleta_jet(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'jet' clásico (azul-cian-verde-amarillo-rojo).
        """
        cmap = cm.get_cmap('jet', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

   #@register_palette("Twilight Shifted")
    def _paleta_twilight_shifted(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'twilight_shifted' de Matplotlib (cambia de púrpura a amarillo).
        """
        cmap = cm.get_cmap('twilight_shifted', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    #@register_palette("Turbo")
    def _paleta_turbo(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'turbo' de Matplotlib (espectro de colores vibrantes).
        """
        cmap = cm.get_cmap('turbo', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    #@register_palette("Rainbow")
    def _paleta_rainbow(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'rainbow' de Matplotlib (arcoíris).
        """
        cmap = cm.get_cmap('rainbow', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    #@register_palette("Ocean")
    def _paleta_ocean(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'ocean' de Matplotlib (azul marino a verde).
        """
        cmap = cm.get_cmap('ocean', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    #@register_palette("Pink")
    def _paleta_pink(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'pink' de Matplotlib (rosa claro).
        """
        cmap = cm.get_cmap('pink', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
#    @register_palette("Escape Speed")
    def _paleta_escape_speed_from_norm(self, norm: np.ndarray) -> np.ndarray:
        """
        Paleta según rapidez de escape, aplicable a `norm` (valores en [0,1]):
        - Escape rápido (norm bajo) → tonos cálidos
        - Escape lento (norm alto) → tonos fríos

        Parámetros:
        - norm: array H×W con M/max_iter, valores en [0,1]

        Devuelve:
        - Array uint8 (H, W, 3) con los valores RGB.
        """
        # 1) Invertir para que norm bajo = rápido = índice alto
        inv = np.clip(1 - norm, 0, 1)
        # 2) Construir LUT de 256 colores (p.ej. 'inferno')
        cmap = cm.get_cmap('inferno', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        # 3) Mapear inv [0,1] → índices 0–255
        indices = (inv * 255).astype(np.uint8)
        # 4) Devolver RGB
        return lut[indices]

    # se queda, para proximamente implementar clase de equiv
    #@register_palette("Cubehelix")
    def _paleta_cubehelix(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'cubehelix' de Matplotlib (perceptualmente uniforme,
        ideal para resaltar detalle). Muestreamos 1024 colores.
        """
        # 1) Generamos un LUT de 1024 entradas
        cmap = cm.get_cmap('cubehelix', 1024)
        lut = (cmap(np.linspace(0, 1, 1024))[:, :3] * 255).astype(np.uint8)  # (1024, 3)

        # 2) Mapear norm ∈ [0,1] a índices 0..1023
        indices = np.uint16((norm * 1023).clip(0, 1023))  # shape=(H,W), valores 0–1023

        # 3) Indexar la LUT
        return lut[indices]  # shape=(H, W, 3), dtype=uint8
    
    #@register_palette("Spectral")
    def _paleta_spectral(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'Spectral' de Matplotlib (divergente, multicolor).
        """
        cmap = cm.get_cmap('Spectral', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    #@register_palette("RdYlBu")
    def _paleta_rdyalbu(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'RdYlBu' de Matplotlib (divergente rojo→amarillo→azul).
        """
        cmap = cm.get_cmap('RdYlBu', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    @register_palette("Iteracion varibales (YlGnBu variable)")
    def _paleta_ylgnbu(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'YlGnBu' de Matplotlib (secuencial amarillo→verde→azul).
        """
        iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
        cycle = self.clase_equiv
        mod = iters % cycle
        cmap = cm.get_cmap('YlGnBu', cycle)
        lut = (cmap(np.arange(cycle))[:, :3] * 255).astype(np.uint8)
        return lut[mod]
    
    @register_palette("Iteraciones variables (Viridis variable)")
    def _pallete_iters_variable_viridis(self, norm: np.ndarray) -> np.ndarray:
        """
        - Reconstruye iter ∈ [0..max_iter] desde norm
        - Usa iter % 64 para indexar un LUT de viridis de self.clase_equiv colores
        """
        iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
        cycle = self.clase_equiv
        mod = iters % cycle
        cmap = cm.get_cmap('viridis', cycle)
        lut = (cmap(np.arange(cycle))[:, :3] * 255).astype(np.uint8)
        return lut[mod]

    @register_palette("Iteraciones variables (Plasma variable)")
    def _pallete_iters_variable_plasma(self, norm: np.ndarray) -> np.ndarray:
        """
        - Reconstruye iter ∈ [0..max_iter] desde norm
        - Usa iter % 64 para indexar un LUT de plasma de self.clase_equiv colores
        """
        iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
        cycle = self.clase_equiv
        cmap= cm.get_cmap('plasma', cycle)
        lut = (cmap(np.arange(cycle))[:, :3] * 255).astype(np.uint8)
        return lut[iters % cycle]

    @register_palette("Iteraciones variables (Grises)")
    def _pallete_iters_variable_grises(self, norm: np.ndarray) -> np.ndarray:
        """
        - Reconstruye iter ∈ [0..max_iter] desde norm
        - Usa iter % 64 para indexar un LUT de grises cíclico de self.clase_equiv colores
        """
        iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
        cycle = self.clase_equiv
        mod = iters % cycle
        gray = np.uint8(((mod.astype(float) / (cycle - 1)) * 255).clip(0, 255))
        return np.dstack([gray, gray, gray])
    
    @register_palette("Iteraciones variables (Twilight Shifted)")
    def _paleta_iters_variable_twilight(self, norm: np.ndarray) -> np.ndarray:
        """
        - Reconstruye iter ∈ [0..max_iter] desde norm
        - Usa iter % 512 para indexar un LUT de twilight_shifted de 512 colores
        """
        iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
        cycle = self.clase_equiv
        cmap = cm.get_cmap('twilight_shifted', cycle)
        lut = (cmap(np.arange(cycle))[:, :3] * 255).astype(np.uint8)
        return lut[iters % cycle]
        
    # ——— Método para pasar a la siguiente paleta ———
    
    def next_palette(self):
        """
        Incrementa palette_index y actualiza el widget.
        """
        self.palette_index = (self.palette_index + 1) % len(self.palettes)
        self.update()  # Fuerza un repaint (llamará de nuevo a paintGL)

    def previous_palette(self):
        """
        Decrementa palette_index y actualiza el widget.
        """
        self.palette_index = (self.palette_index - 1) % len(self.palettes)
        self.update()
        
    # Opcional: atajar una tecla para cambiar


    # ——— paintGL revisitado, usando el índice de paleta ———
    def guardar_imagen(self) -> None:
        ruta, _ = QFileDialog.getSaveFileName(
            None,
            "Guardar imagen",
            f"fractal_{self.xmin:.16f}_{self.xmax:.16f}_{self.ymin:.16f}_{self.ymax:.16f}_{self.tipo_fractal}_{self.max_iter}.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;Todos los archivos (*)"
        )
        if not ruta:
            return

        # 1) Generar los datos del fractal
        self.mandelbrot.actualizar_fractal(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            4*self.width, 4*self.height,
            self.max_iter,
            self.formula, self.tipo_calculo,
            self.tipo_fractal,
            self.real, self.imag
        )
        data = self.mandelbrot.calcular_fractal()  # np.ndarray de enteros [0..max_iter]

        # 2) Normalizar a [0,1]
        norm = data.astype(float) / self.max_iter

        # 3) Elegir la paleta actual
        name, func = self.palettes[self.palette_index]

        # 4) Aplicar la paleta → rgb uint8 (H,W,3)
        rgb = func(norm)

        # 5) Invertir verticalmente si lo hacés en paintGL

        # 6) Guardar la imagen directamente como RGB (sin pasar cmap)
        # plt.imsave admite uint8 RGB si no le pasás cmap
        plt.imsave(ruta, rgb)
        print(f"Imagen guardada en: {ruta}")

    def linkeo_botones(self):
        self.ui.boton_dividir.clicked.connect(lambda : self.dividir())
        self.ui.boton_duplicar.clicked.connect(lambda : self.duplicar())
        self.ui.boton_dividir_clase_equiv.clicked.connect(lambda : self.dividir_clase_equiv())
        self.ui.boton_duplicar_clase_equiv.clicked.connect(lambda : self.duplicar_clase_equiv())


        self.ui.slider_iteraciones.valueChanged.connect(lambda value: self.ui.max_iter_entrada.setText(str(value)))
    
    def duplicar(self):
        self.ui.max_iter_entrada.setText(str(int(int(self.ui.max_iter_entrada.text())*2)))
        self.update()
        
    def dividir(self):
        self.ui.max_iter_entrada.setText(str(int(int(self.ui.max_iter_entrada.text())/2)))
        self.update()
    
    def dividir_clase_equiv(self=Ui_Boundary()):
        self.ui.clase_equiv_entrada.setText(str(int(int(self.ui.clase_equiv_entrada.text())/2)))
        self.update()
    
    def duplicar_clase_equiv(self=Ui_Boundary()):
        self.ui.clase_equiv_entrada.setText(str(int(int(self.ui.clase_equiv_entrada.text())*2)))
        self.update()
    
    def reset_view(self):
        """
        Resetea la vista a los valores iniciales.
        """
        self.xmin = -2.0
        self.xmax = 1.2
        self.ymin = -0.9
        self.ymax = 0.9
        self.max_iter = 256
        self.width = 1000
        self.height = 600
        self.zoom_factor = 1.0
        self.clase_equiv = 128
        self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter, self.clase_equiv)
        self.update()

    def mostrar_parametros(self, xmin, xmax, ymin, ymax, width, height, max_iter, clase_equiv):
        """
        Muestra los parámetros del fractal en la UI.
        """
        self.ui.xmin_entrada.setText(f"{xmin}")
        self.ui.xmax_entrada.setText(f"{xmax}")
        self.ui.ymin_entrada.setText(f"{ymin}")
        self.ui.ymax_entrada.setText(f"{ymax}")
        self.ui.width_entrada.setText(f"{width}")
        self.ui.high_entrada.setText(f"{height}")
        self.ui.max_iter_entrada.setText(f"{max_iter}")
        self.ui.clase_equiv_entrada.setText(f"{clase_equiv}")


    def actualizar_parametros(self) -> None:
        """
        Actualiza los parámetros del fractal según los valores de la UI.
        """
        
        self.cmap           =   str(self.ui.cmap_comboBox.currentText())
        self.zoom_in        =   float(self.ui.zoom_in_factor_entrada.text())
        self.zoom_out       =   float(self.ui.zoom_out_factor_entrada.text()) 
        self.width          =   int(self.ui.width_entrada.text())
        self.height         =   int(self.ui.high_entrada.text())
        self.max_iter       =   int(self.ui.max_iter_entrada.text())
        self.tipo_calculo   =   str(self.ui.tipo_calculo_comboBox.currentText())
        self.tipo_fractal   =   str(self.ui.tipo_fractal_comboBox.currentText())
        self.formula        =   str(self.ui.formula_entrada.text())
        self.real           =   float(self.ui.real_julia_entrada.text())
        self.imag           =   float(self.ui.im_julia_entrada.text())
        self.clase_equiv    =   int(self.ui.clase_equiv_entrada.text())
        self.mandelbrot.actualizar_fractal(
                self.xmin, self.xmax,
                self.ymin, self.ymax,
                self.width, self.height,
                self.max_iter,
                self.formula, self.tipo_calculo,
                self.tipo_fractal,
                self.real, self.imag
            )


    def initializeGL(self):
        glClearColor(0, 0, 0, 1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()
        real, imag = self.pixel_a_complejo(x, y)
        self.ui.label_coordenadas.setText(f"Re: {real:.16f}, Im: {imag:.16f}")

        if self.dragging:
            x0, y0 = self.last_pos.x(), self.last_pos.y()
            c0_real, c0_imag = self.pixel_a_complejo(x0, y0)
            dx = c0_real - real
            dy = c0_imag - imag
            self.xmin += dx
            self.xmax += dx
            self.ymin += dy
            self.ymax += dy
            self.last_pos = event.pos()
            self.actualizar_parametros()
            self.mostrar_parametros(
                self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter, self.clase_equiv
            )
            self.update()
    
    def pixel_a_complejo(self, x, y):
        real = self.xmin + (x / self.width) * (self.xmax - self.xmin)
        imag = self.ymin + (y / self.height) * (self.ymax - self.ymin)
        return real, imag
    
    def draw_branch(self, x, y, angle, length, depth):
        if depth == 0:
            return

        # Aplicar el factor de zoom a la longitud
        length *= self.zoom_factor

        x2 = x + length * cos(radians(angle))
        y2 = y + length * sin(radians(angle))

        glBegin(GL_LINES)
        glVertex2f(x, y)
        glVertex2f(x2, y2)
        glEnd()

        # Llamadas recursivas para las ramas izquierda y derecha
        self.draw_branch(x2, y2, angle - 30, length * 0.7, depth - 1)
        self.draw_branch(x2, y2, angle + 30, length * 0.7, depth - 1)
    
    
    def paintGL(self):
        if str(self.ui.generador_comboBox.currentText()) == "Sucesion":
            glClear(GL_COLOR_BUFFER_BIT)
            glLoadIdentity()

            # 1) Actualizar parámetros y generar fractal
            self.actualizar_parametros()
            try:
                data = self.mandelbrot.calcular_fractal()  # np.ndarray float64
            except Exception as e:
                print(f"Error al calcular el fractal: {e}")
                return
            
            # 2) Normalizar a [0,1]
            norm = data/self.max_iter

            # 3) Elegir nombre y función de paleta actual
            name, func = self.palettes[self.palette_index]

            # 4) Llamar a la función de paleta para obtener rgb (uint8, shape=(H,W,3))
            rgb = func(norm)

            # 5) Invertir verticalmente
            rgb = rgb[::-1, :, :]

            # 6) Dibujar
            glDrawPixels(self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, rgb)
            
        if str(self.ui.generador_comboBox.currentText()) == "Lsystem":
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            # Centrar el árbol en el medio inferior de la pantalla
            self.draw_branch(0.0, -0.9, 90, 0.3, 15)

    def resizeGL(self, w, h):
        self.width = w
        self.height = h
        glViewport(0, 0, w, h)
        self.update()
  
    def wheelEvent(self, event):
        zoom = 0.9 if event.angleDelta().y() > 0 else 1.1
        cx = (self.xmin + self.xmax) / 2
        cy = (self.ymin + self.ymax) / 2
        dx = (self.xmax - self.xmin) * zoom / 2
        dy = (self.ymax - self.ymin) * zoom / 2
        self.xmin, self.xmax = cx - dx, cx + dx
        self.ymin, self.ymax = cy - dy, cy + dy
        
        self.actualizar_parametros()
        self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter, self.clase_equiv)
        self.update()

    def mousePressEvent(self, event):
        
        if event.button() == Qt.LeftButton:

            x_pixel = event.x()
            y_pixel = event.y()  

            c_x = self.xmin + (x_pixel / self.width) * (self.xmax - self.xmin)
            c_y = self.ymin + (y_pixel / self.height) * (self.ymax - self.ymin)

            self.xmin = c_x - (c_x - self.xmin) * self.zoom_in
            self.xmax = c_x + (self.xmax - c_x) * self.zoom_in
            self.ymin = c_y - (c_y - self.ymin) * self.zoom_in
            self.ymax = c_y + (self.ymax - c_y) * self.zoom_in
            self.actualizar_parametros()
            self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter, self.clase_equiv)
            self.update()
            
        elif event.button() == Qt.RightButton:

            x_pixel = event.x()
            y_pixel = event.y()  

            c_x = self.xmin + (x_pixel / self.width) * (self.xmax - self.xmin)
            c_y = self.ymin + (y_pixel / self.height) * (self.ymax - self.ymin)

            self.xmin = c_x - (c_x - self.xmin) * self.zoom_out
            self.xmax = c_x + (self.xmax - c_x) * self.zoom_out
            self.ymin = c_y - (c_y - self.ymin) * self.zoom_out
            self.ymax = c_y + (self.ymax - c_y) * self.zoom_out
            
            self.actualizar_parametros()
            self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter, self.clase_equiv)
            self.update()

        elif event.button() == Qt.MiddleButton:
            self.dragging = True
            self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.dragging = False
    
    def keyPressEvent(self, event):
        
        if str(self.ui.generador_comboBox.currentText()) == "Sucesion":
            move = 0.05
            dx = (self.xmax - self.xmin) * move
            dy = (self.ymax - self.ymin) * move
            
            if event.key() in (Qt.Key_Left, Qt.Key_A):
                self.xmin -= dx
                self.xmax -= dx
                self.update()
                
            elif event.key() in (Qt.Key_Right, Qt.Key_D):
                self.xmin += dx
                self.xmax += dx
                self.update()
                
            elif event.key() in (Qt.Key_Up, Qt.Key_W):
                self.ymin -= dy
                self.ymax -= dy
                self.update()
                
            elif event.key() in (Qt.Key_Down, Qt.Key_S):
                self.ymin += dy
                self.ymax += dy
                self.update()

            elif event.key() == Qt.Key_Plus:
                # Calcular el punto central actual
                c_x = (self.xmin + self.xmax) / 2
                c_y = (self.ymin + self.ymax) / 2

                # Ajustar los límites en torno al centro con el factor de zoom
                dx = (self.xmax - self.xmin) * self.zoom_in / 2
                dy = (self.ymax - self.ymin) * self.zoom_in / 2
                self.xmin, self.xmax = c_x - dx, c_x + dx
                self.ymin, self.ymax = c_y - dy, c_y + dy

                # Refrescar parámetros y repintar
                self.actualizar_parametros()
                self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax,
                                        self.width, self.height, self.max_iter, self.clase_equiv)
                self.update()
                
            elif event.key() == Qt.Key_Minus:
                c_x = (self.xmin + self.xmax) / 2
                c_y = (self.ymin + self.ymax) / 2

                # Ajustar los límites en torno al centro con el factor de zoom
                dx = (self.xmax - self.xmin) * self.zoom_out / 2
                dy = (self.ymax - self.ymin) * self.zoom_out / 2
                self.xmin, self.xmax = c_x - dx, c_x + dx
                self.ymin, self.ymax = c_y - dy, c_y + dy

                # Refrescar parámetros y repintar
                self.actualizar_parametros()
                self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax,
                                        self.width, self.height, self.max_iter, self.clase_equiv)
                self.update()
                
            elif event.key() == Qt.Key_P:    
                self.next_palette()
            
            elif event.key() == Qt.Key_O:
                self.previous_palette()
                
            elif event.key() == Qt.Key_R:
                self.reset_view()

            elif event.key() == Qt.Key_G:
                self.duplicar()
            
            elif event.key() == Qt.Key_H:
                self.dividir()

            elif event.key() == Qt.Key_B:
                self.duplicar_clase_equiv()

            elif event.key() == Qt.Key_N:
                self.dividir_clase_equiv()

            
        if str(self.ui.generador_comboBox.currentText()) == "Lsystem":
            if event.key() == Qt.Key_Plus:
                self.zoom_factor *= 1.1  # Acercar
            elif event.key() == Qt.Key_Minus:
                self.zoom_factor /= 1.1  # Alejar
            self.update()  # Redibujar la escena