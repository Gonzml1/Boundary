from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow, QGraphicsEllipseItem, QGraphicsScene, QGraphicsView
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import QPointF
import core.funciones_ui as md
import gui.tema_oscuro as ts
from gui.MandelbrotGUI import Ui_Boundary
from OpenGL.GL import *

class Punto(QGraphicsEllipseItem):
    def __init__(self, callback):
        super().__init__(-10, -10, 20, 20)
        self.setBrush(QBrush(QColor("white")))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.callback = callback

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange:
            new_x = max(0, min(200, value.x()))
            new_y = max(0, min(200, value.y()))
            self.callback(new_x, new_y)
            return QPointF(new_x, new_y)
        return super().itemChange(change, value) 

class GraphicsViewFlechas(QGraphicsView):
    def __init__(self, *args, punto=None, ui=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.punto = punto
        self.ui = ui

    def keyPressEvent(self, event):
        try:
            paso = float(self.ui.paso_entrada.text())
        except Exception:
            paso = 1
        pos = self.punto.pos()
        if event.key() == QtCore.Qt.Key_Left:
            self.punto.setPos(max(0, pos.x() - paso), pos.y())
        elif event.key() == QtCore.Qt.Key_Right:
            self.punto.setPos(min(200, pos.x() + paso), pos.y())
        elif event.key() == QtCore.Qt.Key_Up:
            self.punto.setPos(pos.x(), max(0, pos.y() - paso))
        elif event.key() == QtCore.Qt.Key_Down:
            self.punto.setPos(pos.x(), min(200, pos.y() + paso))
        super().keyPressEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Boundary()
        self.ui.setupUi(self)

        # Tema oscuro
        ts.tema_oscuro(QtWidgets.QApplication.instance())

        # Mostrar fractal
        mandelbrot = md.mostrar_fractal_opengl(self.ui)

        # Crear escena de tamaño fijo
        self.scene = QGraphicsScene(0, 0, 200, 200)

        # Punto movible
        self.punto = Punto(self.actualizar_coordenadas)
        self.scene.addItem(self.punto)
        self.punto.setPos(100, 100)

        # Cambiás la clase del graphicsView original por la tuya
        self.ui.graphicsView.__class__ = GraphicsViewFlechas
        self.ui.graphicsView.punto = self.punto
        self.ui.graphicsView.ui = self.ui    # <<---- ESTA LINEA ES CLAVE NASHE
        self.ui.graphicsView.setScene(self.scene)
        self.ui.graphicsView.setSceneRect(0, 0, 200, 200)
        self.ui.graphicsView.setFixedSize(200, 200)
        self.ui.graphicsView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.ui.graphicsView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.ui.graphicsView.setInteractive(True)

        # ComboBox por defecto
        self.ui.tipo_calculo_comboBox.setCurrentIndex(2)
        self.ui.formula_entrada.setText("z**2 + c")
        self.scene.changed.connect(mandelbrot.update)

        # Foco directo al graphicsView original
        self.ui.graphicsView.setFocus()

    def actualizar_coordenadas(self, x, y):
        x_real = (x / 100) * 2 - 2
        y_real = -((y / 100) * 2 - 2)
        self.ui.real_julia_entrada.setText(f"{x_real:.5f}")
        self.ui.im_julia_entrada.setText(f"{y_real:.5f}")
        self.ui.label_coordenadas2.setText(f"Re: {x_real:.5f}, Im: {y_real:.5f}")
