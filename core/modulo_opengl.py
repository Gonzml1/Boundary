import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from core.modulo_de_calculo_fractales import calculos_mandelbrot
from math import sin, cos, radians

    
class MandelbrotWidget(QOpenGLWidget):
    def __init__(self,cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, boundary, zoom_in, zoom_out ):
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
        self.mandelbrot     =       calculos_mandelbrot(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter,self.formula, self.tipo_calculo, self.tipo_fractal, self.real, self.imag)                               
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.ui.boton_hacer_fractal.clicked.connect(lambda : self.update())
        self.actualizar_parametros()
    
    def mostrar_parametros(self, xmin, xmax, ymin, ymax):
        self.ui.xmin_entrada.setText(f"{xmin}")
        self.ui.xmax_entrada.setText(f"{xmax}")
        self.ui.ymin_entrada.setText(f"{ymin}")
        self.ui.ymax_entrada.setText(f"{ymax}")
        
    def actualizar_parametros(self):
        self.cmap           =   str(self.ui.cmap_comboBox.currentText())
        self.zoom_in        =   float(self.ui.zoom_in_factor_entrada.text())
        self.zoom_out       =   float(self.ui.zoom_out_factor_entrada.text()) 
        self.width          =   int(self.ui.width_entrada.text())
        self.height         =   int(self.ui.high_entrada.text())
        self.max_iter       =   int(self.ui.max_iter_entrada.text())
        self.tipo_calculo   =   str(self.ui.tipo_calculo_comboBox.currentText())
        self.tipo_fractal   =   str(self.ui.tipo_fractal_comboBox.currentText())
        self.formula        =   str(self.ui.formula_entrada.text())

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

            self.actualizar_parametros()
            self.mandelbrot.actualizar_fractal(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter,self.formula, self.tipo_calculo, self.tipo_fractal, self.real, self.imag)
            data = self.mandelbrot.calcular_fractal()

            # Normalizamos para color RGB
            norm = data / self.max_iter
            rgb = np.uint8(np.dstack([norm*255, norm**0.5*255, norm**0.3*255]))
            rgb = rgb[::-1, :]
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
        
        self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax)
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
            self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax)
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
            
            self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax)
            self.update()
    
    def keyPressEvent(self, event):
        
        if str(self.ui.generador_comboBox.currentText()) == "Sucesion":
            move = 0.05
            dx = (self.xmax - self.xmin) * move
            dy = (self.ymax - self.ymin) * move

            if event.key() in (Qt.Key_Left, Qt.Key_A):
                self.xmin -= dx
                self.xmax -= dx
            elif event.key() in (Qt.Key_Right, Qt.Key_D):
                self.xmin += dx
                self.xmax += dx
            elif event.key() in (Qt.Key_Up, Qt.Key_W):
                self.ymin -= dy
                self.ymax -= dy
            elif event.key() in (Qt.Key_Down, Qt.Key_S):
                self.ymin += dy
                self.ymax += dy
            self.mostrar_parametros(self.xmin, self.xmax, self.ymin, self.ymax)    
            self.update()
    
        if str(self.ui.generador_comboBox.currentText()) == "Lsystem":
            if event.key() == Qt.Key_Plus:
                self.zoom_factor *= 1.1  # Acercar
            elif event.key() == Qt.Key_Minus:
                self.zoom_factor /= 1.1  # Alejar
            self.update()  # Redibujar la escena