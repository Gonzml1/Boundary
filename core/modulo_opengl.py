import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
import core.modulo_de_calculo_fractales as tf


    
class MandelbrotWidget(QOpenGLWidget):
    def __init__(self,cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, boundary):
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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def actualizar_parametros(self):
        self.cmap           =   str(self.ui.cmap_comboBox.currentText())
        self.width          =   int(self.ui.width_entrada.text())
        self.height         =   int(self.ui.high_entrada.text())
        self.max_iter       =   int(self.ui.max_iter_entrada.text())
        self.tipo_calculo   =   str(self.ui.tipo_calculo_comboBox.currentText())
        self.tipo_fractal   =   str(self.ui.tipo_fractal_comboBox.currentText())
        self.formula        =   str(self.ui.formula_entrada.text())

    def initializeGL(self):
        glClearColor(0, 0, 0, 1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        
        self.actualizar_parametros()
        data = tf.calcular_fractal(
            self.xmin, self.xmax,
            self.ymin, self.ymax,
            self.width, self.height,
            self.max_iter, self.formula, 
            self.tipo_calculo,self.tipo_fractal,
            self.real, self.imag
        )

        # Normalizamos para color RGB
        norm = data / self.max_iter
        rgb = np.uint8(np.dstack([norm*255, norm**0.5*255, norm**0.3*255]))
        rgb = rgb[::-1, :]
        glDrawPixels(self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, rgb)

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
        self.update()

    def keyPressEvent(self, event):
        move = 0.05
        dx = (self.xmax - self.xmin) * move
        dy = (self.ymax - self.ymin) * move

        if event.key() == Qt.Key_Left:
            self.xmin -= dx
            self.xmax -= dx
        elif event.key() == Qt.Key_Right:
            self.xmin += dx
            self.xmax += dx
        elif event.key() == Qt.Key_Up:
            self.ymin -= dy
            self.ymax -= dy
        elif event.key() == Qt.Key_Down:
            self.ymin += dy
            self.ymax += dy

        self.update()