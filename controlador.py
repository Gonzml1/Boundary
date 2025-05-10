# pyuic6 -x "V:\ABoundary\gui\MandelbrotGUI.ui"  -o "V:\ABoundary\gui\MandelbrotGUI.py"
import core.funciones_ui as md
from PyQt5 import QtWidgets
from gui.MandelbrotGUI import Ui_Boundary
import sys
import gui.tema_oscuro as ts
from core.modulo_opengl import MandelbrotWidget
from OpenGL.GL import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from PyQt5.QtCore import Qt
# fractales a añadir en un futuro (muy cercano)
# Newton-Raphson 

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
cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag = md.obtener_datos(ui)
mandelbrot_widget = MandelbrotWidget(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, ui)
md.mostrar_fractal_opengl(ui)


#########################
#   LINKEO DE BOTONES   #
#########################

md.linkeo_botones(ui)


if __name__ == "__main__":

    MainWindow.show()
    # md.generar_mandelbrot(ui)
    sys.exit(app.exec())

