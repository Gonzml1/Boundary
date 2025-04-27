import TodasFunciones2 as tf
from PyQt6 import QtCore, QtGui, QtWidgets
from MandelbrotGUI import Ui_MainWindow
import cupy as cp
import numpy as np
from PyQt6.QtGui import QImage, QPixmap

def generar_mandelbrot():
    xmin      =   float(ui.xmin_entrada.text())
    xmax      =   float(ui.xmax_entrada.text())
    ymin      =   float(ui.ymin_entrada.text())
    ymax      =   float(ui.ymax_entrada.text())
    width     =   int(ui.width_entrada.text())
    height    =   int(ui.high_entrada.text())
    max_iter  =   int(ui.max_iter_entrada.text())
    M = tf.hacer_mandelbrot(xmin, xmax, ymin, ymax,width,height,max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png")
    return "Mandelbrot generado"

def mostrar_fractal(ruta_imagen):
    # Cargar la imagen del fractal
    pixmap = QtGui.QPixmap(ruta_imagen)
    
    # Crear una escena y añadir la imagen
    scene = QtWidgets.QGraphicsScene()
    scene.addPixmap(pixmap)
    # Asignar la escena al QGraphicsView
    ui.Grafica_mostrar.setScene(scene)
    
def mostrar_imagen(ruta_imagen):
    pixmap = QPixmap(ruta_imagen)
    ui.imagen_label.setPixmap(pixmap)
    ui.imagen_label.setScaledContents(True)



import sys
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)

#########################
#   LINKEO DE BOTONES   #
#########################

ui.boton_hacer_fractal.clicked.connect(lambda : generar_mandelbrot())



MainWindow.show()
sys.exit(app.exec())

    