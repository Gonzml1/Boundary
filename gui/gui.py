from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow, QGraphicsEllipseItem, QGraphicsScene
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Boundary()
        self.ui.setupUi(self)
        
        # Tema oscuro
        ts.tema_oscuro(QtWidgets.QApplication.instance())

        # Mostrar fractal
        mandelbrot=md.mostrar_fractal_opengl(self.ui)

        # Linkear botones
        md.linkeo_botones(self.ui)

        # Crear escena de tamaño fijo
        self.scene = QGraphicsScene(0, 0, 200, 200)
        self.ui.graphicsView.setScene(self.scene)

        # Desactivar scrollbars
        self.ui.graphicsView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.ui.graphicsView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # Ajustar el view para que no agregue márgenes extras
        self.ui.graphicsView.setSceneRect(0, 0, 200, 200)
        self.ui.graphicsView.setFixedSize(200, 200)

        # Evitar que se pueda hacer zoom o mover con el mouse
        self.ui.graphicsView.setInteractive(True)
        
        # Punto movible
        self.punto = Punto(self.actualizar_coordenadas)
        self.scene.addItem(self.punto)
        self.punto.setPos(100, 100)

        # ComboBox por defecto
        self.ui.tipo_calculo_comboBox.setCurrentIndex(4)
        self.ui.graphicsView.scene().changed.connect(mandelbrot.update)
        

    def actualizar_coordenadas(self, x, y):
        x_real = (x / 100) * 2 -2
        y_real = -((y / 100) * (2)-2)
        self.ui.real_julia_entrada.setText(f"{x_real:.5f}")
        self.ui.im_julia_entrada.setText(f"{y_real:.5f}")
        self.ui.label_coordenadas2.setText(f"Re: {x_real:.5f}, Im: {y_real:.5f}")
