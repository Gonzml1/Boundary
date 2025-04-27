import TodasFunciones2 as tf
from PyQt6 import QtCore, QtGui, QtWidgets
from MandelbrotGUI import Ui_MainWindow
import cupy as cp
import numpy as np
from PyQt6.QtGui import QImage, QPixmap

def mostrar_fractal(ruta_imagen):
    pixmap = QtGui.QPixmap(ruta_imagen)
    scene = QtWidgets.QGraphicsScene()
    scene.addPixmap(pixmap)
    ui.Grafica_mostrar.setScene(scene)
    

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






"""
relacion = 0.5
dimensiones=np.array([xmin,xmax,ymin,ymax])
for i in range(cantidad_ite):
    mandelbrot = generate_mandelbrot_gpu(dimensiones[0], dimensiones[1], dimensiones[2], dimensiones[3], width, height, max_iter)
    ruta_guardado2 = f"V:\\archivosvisual\\zoom2\\mandelbrotvideo{i}.png"
    plot_mandelbrot(mandelbrot, ruta_guardado2, width, height, dpi1)
    
    # Reducir las dimensiones alrededor del punto de zoom
    dimensiones[0] = centro_x - (centro_x - dimensiones[0]) *relacion
    dimensiones[1] = centro_x + (dimensiones[1] - centro_x) *relacion  
    dimensiones[2] = centro_y - (centro_y - dimensiones[2]) *relacion  
    dimensiones[3] = centro_y + (dimensiones[3] - centro_y) *relacion 
    print(dimensiones[0],dimensiones[1],dimensiones[2],dimensiones[3])
    print("ITERACION", i)
"""

def zoom_in():
    ui.xmin_entrada.setText(str(float(ui.xmin_entrada.text())*0.5))
    ui.xmax_entrada.setText(str(float(ui.xmax_entrada.text())*0.5))
    ui.ymin_entrada.setText(str(float(ui.ymin_entrada.text())*0.5))
    ui.ymax_entrada.setText(str(float(ui.ymax_entrada.text())*0.5))
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



import sys
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)

#########################
#   LINKEO DE BOTONES   #
#########################

ui.boton_hacer_fractal.clicked.connect(lambda : generar_mandelbrot())
ui.boton_hacer_zoom_in.clicked.connect(lambda : zoom_in())
ui.boton_hacer_zoom_out.clicked.connect(lambda : tf.zoom_out())


MainWindow.show()
sys.exit(app.exec())

    