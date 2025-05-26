import sys
from math import radians, sin, cos

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt
from OpenGL.GL import *

class TreeGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Parámetros fractal
        self.depth        = 12    # niveles de recursión
        self.length       = 0.4   # longitud inicial (coordenadas normalizadas)
        self.shrink       = 0.7   # factor de reducción por nivel
        # Ángulos manipulables
        self.left_angle   = 25.0  # ángulo para la rama izquierda
        self.right_angle  = 25.0  # ángulo para la rama derecha
        # Transformaciones globales
        self.zoom_factor  = 1.0
        self.pan_y        = -0.9  # desplaza la raíz hacia abajo

    def initializeGL(self):
        glClearColor(1, 1, 1, 1)       # fondo blanco
        glColor3f(0.2, 0.6, 0.2)       # color ramas (verde)
        glLineWidth(2.0)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h if h else 1
        glOrtho(-aspect, aspect, -1.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glPushMatrix()
        glTranslatef(0.0, self.pan_y, 0.0)
        glScalef(self.zoom_factor, self.zoom_factor, 1.0)
        # arrancamos desde el origen local (0,0) apuntando hacia arriba (90°)
        self.draw_branch(0.0, 0.0, 90.0, self.length, self.depth)
        glPopMatrix()

    def draw_branch(self, x, y, angle_deg, length, depth):
        if depth == 0:
            return

        # calcula extremo de la rama
        rad = radians(angle_deg)
        x2 = x + length * cos(rad)
        y2 = y + length * sin(rad)

        # dibuja línea
        glBegin(GL_LINES)
        glVertex2f(x, y)
        glVertex2f(x2, y2)
        glEnd()

        # bifurcación izquierda usando self.left_angle
        self.draw_branch(
            x2, y2,
            angle_deg - self.left_angle,
            length * self.shrink,
            depth - 1
        )
        # bifurcación derecha usando self.right_angle
        self.draw_branch(
            x2, y2,
            angle_deg + self.right_angle,
            length * self.shrink,
            depth - 1
        )

    def wheelEvent(self, event):
        """Zoom con rueda del ratón."""
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 1/1.1
        self.zoom_factor = max(0.1, min(self.zoom_factor * factor, 10.0))
        self.update()

    def keyPressEvent(self, event):
        """Ajusta ángulos con teclado:
           A/Z: incrementar/decrementar izquierda
           S/X: incrementar/decrementar derecha
        """
        step = 1.0
        key = event.key()
        if key == Qt.Key_A:
            self.left_angle += step
        elif key == Qt.Key_Z:
            self.left_angle -= step
        elif key == Qt.Key_S:
            self.right_angle += step
        elif key == Qt.Key_X:
            self.right_angle -= step
        else:
            super().keyPressEvent(event)
            return
        # repinta con los nuevos ángulos
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Árbol Fractal Interactivo (OpenGL)")
        self.glWidget = TreeGLWidget(self)
        self.setCentralWidget(self.glWidget)
        self.resize(800, 600)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
