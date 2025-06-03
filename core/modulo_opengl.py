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
    
PALETTE_REGISTRY = []

def register_palette(palette_name: str):
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
        self.mandelbrot     =       calculos_mandelbrot(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter,self.formula, self.tipo_calculo, self.tipo_fractal, self.real, self.imag, self.ui)                               
        self.lsystem        =       None  
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.ui.boton_hacer_fractal.clicked.connect(lambda : self.update())
        self.ui.slider_iteraciones.valueChanged.connect(self.update)
        self.actualizar_parametros()
        self.palettes = []
        for name, fn in PALETTE_REGISTRY:
            bound_fn = fn.__get__(self, type(self))
            self.palettes.append((name, bound_fn))
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.palette_index = 0


    @register_palette("Grises")
    def _paleta_grises(self, norm: np.ndarray) -> np.ndarray:
        """
        Escala de grises: norm en [0,1] → (gray,gray,gray) en [0..255]
        """
        gray = np.uint8((norm * 255).clip(0, 255))
        return np.dstack([gray, gray, gray])  # shape=(H,W,3)

    @register_palette("Rojo→Amarillo→Blanco")
    def _paleta_rojo_amarillo(self, norm: np.ndarray) -> np.ndarray:
        """
        Rojo→Amarillo→Blanco:
        - R siempre 255
        - G crece linealmente de 0→255 para norm en [0,0.5], luego se mantiene 255
        - B se activa solo para norm≥0.5, crece de 0→255 en [0.5,1]
        """
        r = np.uint8(255 * np.ones_like(norm))
        g = np.uint8((np.clip(norm * 2, 0, 1) * 255).clip(0, 255))
        b = np.uint8((np.clip((norm - 0.5) * 2, 0, 1) * 255).clip(0, 255))
        return np.dstack([r, g, b])

    @register_palette("HSV")
    def _paleta_hsv(self, norm: np.ndarray) -> np.ndarray:
        """
        Usa el colormap 'hsv' de Matplotlib:
        - Crea una LUT de 256 colores HSV→RGB y para cada valor de norm indexa a la LUT.
        """
        # 1) Obtenemos la LUT (solo la generamos una vez si quieres optimizar)
        cmap = cm.get_cmap('hsv', 256)                     # Colormap de 256 entradas
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)  # (256,3)

        # 2) Mapeamos norm ∈ [0,1] a 0..255
        indices = np.uint8((norm * 255).clip(0, 255))      # shape=(H,W), valores 0..255

        # 3) Indexamos
        return lut[indices]                                # shape=(H,W,3), dtype=uint8

    @register_palette("Púrpura Psicodélica")
    def _paleta_psicodelica(self, norm: np.ndarray) -> np.ndarray:
        """
        Púrpura psicodélica usando funciones sinusoidales:
        - Tres ciclos de color, fases desplazadas en R,G,B
        """
        # norm ∈ [0,1]
        r = np.uint8((0.5 + 0.5 * np.sin(2 * np.pi * norm * 3 + 0)) * 255)
        g = np.uint8((0.5 + 0.5 * np.sin(2 * np.pi * norm * 3 + 2)) * 255)
        b = np.uint8((0.5 + 0.5 * np.sin(2 * np.pi * norm * 3 + 4)) * 255)
        return np.dstack([r, g, b])

    @register_palette("Bandas RGB")
    def _paleta_bandas_rgb(self, norm: np.ndarray) -> np.ndarray:
        """
        Bandas semilineales: divide norm en 3 franjas, con degradado lineal dentro de cada franja:
        - franja0: rojo crece de 0→1
        - franja1: verde crece de 0→1
        - franja2: azul crece de 0→1
        """
        pos = norm * 3  # pos ∈ [0,3)
        # En la primera porción (pos < 1): r=pos, g=0,b=0
        # Segunda (1 ≤ pos < 2): r=2-pos, g=pos-1, b=0
        # Tercera (2 ≤ pos < 3): r=0, g=3-pos, b=pos-2
        r = np.where(pos < 1, pos, np.where(pos < 2, 2 - pos, 0))
        g = np.where(pos < 1, 0, np.where(pos < 2, pos - 1, 3 - pos))
        b = np.where(pos < 2, 0, pos - 2)
        rgb = np.dstack([r.clip(0,1), g.clip(0,1), b.clip(0,1)])
        return np.uint8(rgb * 255)

    @register_palette("Cian→Magenta→Amarillo")
    def _paleta_inferno(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'inferno' de Matplotlib (una paleta predefinida).
        """
        cmap = cm.get_cmap('inferno', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    @register_palette("Viridis")
    def _paleta_viridis(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'viridis' de Matplotlib (verde-amarillo-azul oscuro).
        """
        cmap = cm.get_cmap('viridis', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)  # (256,3)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]  # (H, W, 3)

    @register_palette("Plasma")
    def _paleta_plasma(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'plasma' de Matplotlib (magenta-naranja-amarillo).
        """
        cmap = cm.get_cmap('plasma', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    @register_palette("Magma")
    def _paleta_magma(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'magma' de Matplotlib (negros-rosas-rojos).
        """
        cmap = cm.get_cmap('magma', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    @register_palette("Cividis")
    def _paleta_cividis(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'cividis' de Matplotlib (amarillo-azul verdoso, enfoque en perceptibilidad).
        """
        cmap = cm.get_cmap('cividis', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    @register_palette("Coolwarm")
    def _paleta_coolwarm(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'coolwarm' de Matplotlib (azul frío a rojo cálido).
        """
        cmap = cm.get_cmap('coolwarm', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    @register_palette("Spring")
    def _paleta_spring(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'spring' de Matplotlib (magenta a amarillo).
        """
        cmap = cm.get_cmap('spring', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    @register_palette("Summer")
    def _paleta_summer(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'summer' de Matplotlib (verde claro a amarillo).
        """
        cmap = cm.get_cmap('summer', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    @register_palette("Autumn")
    def _paleta_autumn(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'autumn' de Matplotlib (rojo a amarillo).
        """
        cmap = cm.get_cmap('autumn', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    @register_palette("Winter")
    def _paleta_winter(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'winter' de Matplotlib (verde azulado a azul).
        """
        cmap = cm.get_cmap('winter', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]

    @register_palette("Jet")
    def _paleta_jet(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'jet' clásico (azul-cian-verde-amarillo-rojo).
        """
        cmap = cm.get_cmap('jet', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    @register_palette("Twilight Shifted")
    def _paleta_twilight_shifted(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'twilight_shifted' de Matplotlib (cambia de púrpura a amarillo).
        """
        cmap = cm.get_cmap('twilight_shifted', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    @register_palette("Turbo")
    def _paleta_turbo(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'turbo' de Matplotlib (espectro de colores vibrantes).
        """
        cmap = cm.get_cmap('turbo', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    @register_palette("Rainbow")
    def _paleta_rainbow(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'rainbow' de Matplotlib (arcoíris).
        """
        cmap = cm.get_cmap('rainbow', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    @register_palette("Ocean")
    def _paleta_ocean(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'ocean' de Matplotlib (azul marino a verde).
        """
        cmap = cm.get_cmap('ocean', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    @register_palette("Pink")
    def _paleta_pink(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'pink' de Matplotlib (rosa claro).
        """
        cmap = cm.get_cmap('pink', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    @register_palette("Accent")
    def _paleta_accent(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'Accent' de Matplotlib (colores brillantes).
        """
        cmap = cm.get_cmap('Accent', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    @register_palette("Dark2")
    def _paleta_dark2(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'Dark2' de Matplotlib (colores oscuros y saturados).
        """
        cmap = cm.get_cmap('Dark2', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    @register_palette("Set1")
    def _paleta_set1(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'Set1' de Matplotlib (colores brillantes y saturados).
        """
        cmap = cm.get_cmap('Set1', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    @register_palette("Set2")
    def _paleta_set2(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'Set2' de Matplotlib (colores suaves y agradables).
        """
        cmap = cm.get_cmap('Set2', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    @register_palette("Set3")
    def _paleta_set3(self, norm: np.ndarray) -> np.ndarray:
        """
        Colormap 'Set3' de Matplotlib (colores variados y agradables).
        """
        cmap = cm.get_cmap('Set3', 256)
        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
        indices = np.uint8((norm * 255).clip(0, 255))
        return lut[indices]
    
    
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
        self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
        self.update()

    def mostrar_parametros(self, xmin, xmax, ymin, ymax, width, height, max_iter):
        self.ui.xmin_entrada.setText(f"{xmin}")
        self.ui.xmax_entrada.setText(f"{xmax}")
        self.ui.ymin_entrada.setText(f"{ymin}")
        self.ui.ymax_entrada.setText(f"{ymax}")
        self.ui.width_entrada.setText(f"{width}")
        self.ui.high_entrada.setText(f"{height}")
        self.ui.max_iter_entrada.setText(f"{max_iter}")


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

            data = self.mandelbrot.calcular_fractal()  # np.ndarray float64

            # 2) Normalizar a [0,1]
            norm = data / self.max_iter

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
        self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
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
            self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
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
            self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
            self.update()
    
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
                
            elif event.key() == Qt.Key_P:    
                self.next_palette()
            
            elif event.key() == Qt.Key_O:
                self.previous_palette()
                
            elif event.key() == Qt.Key_R:
                self.reset_view()

        if str(self.ui.generador_comboBox.currentText()) == "Lsystem":
            if event.key() == Qt.Key_Plus:
                self.zoom_factor *= 1.1  # Acercar
            elif event.key() == Qt.Key_Minus:
                self.zoom_factor /= 1.1  # Alejar
            self.update()  # Redibujar la escena