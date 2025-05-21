# pyuic6 -x "V:\ABoundary\gui\MandelbrotGUI.ui"  -o "V:\ABoundary\gui\MandelbrotGUI.py"
import core.funciones_ui as md
from PyQt5 import QtWidgets
from gui.MandelbrotGUI import Ui_Boundary
import sys
import gui.tema_oscuro as ts
from OpenGL.GL import *
from PyQt5.QtWidgets import QMainWindow, QGraphicsEllipseItem, QGraphicsScene
from PyQt5.QtGui import QBrush, QColor


# fractales a añadir en un futuro (muy cercano)
# Newton-Raphson  
# Mejorar interfaz grafica

class Punto(QGraphicsEllipseItem):
    def __init__(self, callback):
        super().__init__(-5, -5, 10, 10)
        self.setBrush(QBrush(QColor("red")))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.callback = callback

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange:
            self.callback(value.x(), value.y())
        return super().itemChange(change, value)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Boundary()  
        self.ui.setupUi(self)

        # Tema oscuro
        ts.tema_oscuro(QtWidgets.QApplication.instance())

        # Mostrar fractal
        md.mostrar_fractal_opengl(self.ui)

        # Linkear botones
        md.linkeo_botones(self.ui)

        # Agregar escena 2D si hay un graphicsView en el .ui
        self.scene = QGraphicsScene(0, 0, 200, 200)
        self.ui.graphicsView.setScene(self.scene)

        self.punto = Punto(self.actualizar_coordenadas)
        self.scene.addItem(self.punto)
        self.punto.setPos(100, 100)

    def actualizar_coordenadas(self, x, y):
        self.ui.label.setText(f"X: {x:.1f}, Y: {y:.1f}")
        
#########################
#      TEMA OSCURO      #
#########################

app = QtWidgets.QApplication(sys.argv)
ts.tema_oscuro(app)

#########################
#     Creacion Qmain    #
#########################

MainWindow = QtWidgets.QMainWindow()
ui = Ui_Boundary()
ui.setupUi(MainWindow)
md.mostrar_fractal_opengl(ui)
ui.tipo_calculo_comboBox.setCurrentIndex(3)


#########################
#   LINKEO DE BOTONES   #
#########################

md.linkeo_botones(ui)

if __name__ == "__main__":

    MainWindow.show()
    sys.exit(app.exec())

