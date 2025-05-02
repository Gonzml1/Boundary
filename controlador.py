# pyuic6 -x "V:\ABoundary\MandelbrotGUI.ui"  -o "V:\ABoundary\MandelbrotGUI.py"

import TodasFunciones2 as tf
import modelo as md
from PyQt6 import QtCore, QtGui, QtWidgets
from MandelbrotGUI import Ui_Boundary
import cupy as cp
import numpy as np
from PyQt6.QtGui import QImage, QPixmap


    
import sys
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_Boundary()
ui.setupUi(MainWindow)

#########################
#   LINKEO DE BOTONES   #
#########################

ui.boton_hacer_fractal.clicked.connect(lambda : md.generar_mandelbrot(ui))
ui.boton_hacer_zoom_in.clicked.connect(lambda : md.zoom_in(ui))
ui.boton_hacer_zoom_out.clicked.connect(lambda : md.zoom_out(ui))
ui.boton_bajar.clicked.connect(lambda : md.bajar(ui))
ui.boton_subir.clicked.connect(lambda : md.subir(ui))
ui.boton_derecha.clicked.connect(lambda : md.derecha(ui))
ui.boton_izquierda.clicked.connect(lambda : md.izquierda(ui))



MainWindow.show()
md.generar_mandelbrot(ui)
sys.exit(app.exec())

    