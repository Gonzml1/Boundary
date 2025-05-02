# pyuic6 -x "V:\ABoundary\MandelbrotGUI.ui"  -o "V:\ABoundary\MandelbrotGUI.py"

import TodasFunciones2 as tf
from PyQt6 import QtCore, QtGui, QtWidgets
from MandelbrotGUI import Ui_Boundary
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
    formula=   ui.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png")
    return "Mandelbrot generado"

def zoom_in():
    zoom_in_factor= float(ui.zoom_in_factor_entrada.text())
    ui.xmin_entrada.setText(str(float(ui.xmin_entrada.text())*zoom_in_factor))
    ui.xmax_entrada.setText(str(float(ui.xmax_entrada.text())*zoom_in_factor))
    ui.ymin_entrada.setText(str(float(ui.ymin_entrada.text())*zoom_in_factor))
    ui.ymax_entrada.setText(str(float(ui.ymax_entrada.text())*zoom_in_factor))
    xmin      =   float(ui.xmin_entrada.text())
    xmax      =   float(ui.xmax_entrada.text())
    ymin      =   float(ui.ymin_entrada.text())
    ymax      =   float(ui.ymax_entrada.text())
    width     =   int(ui.width_entrada.text())
    height    =   int(ui.high_entrada.text())
    max_iter  =   int(ui.max_iter_entrada.text())
    formula=   ui.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png")
    return "Mandelbrot generado"

def zoom_out():
    zoom_out_factor= float(ui.zoom_out_factor_entrada.text())
    ui.xmin_entrada.setText(str(float(ui.xmin_entrada.text())*zoom_out_factor))
    ui.xmax_entrada.setText(str(float(ui.xmax_entrada.text())*zoom_out_factor))
    ui.ymin_entrada.setText(str(float(ui.ymin_entrada.text())*zoom_out_factor))
    ui.ymax_entrada.setText(str(float(ui.ymax_entrada.text())*zoom_out_factor))
    xmin      =   float(ui.xmin_entrada.text())
    xmax      =   float(ui.xmax_entrada.text())
    ymin      =   float(ui.ymin_entrada.text())
    ymax      =   float(ui.ymax_entrada.text())
    width     =   int(ui.width_entrada.text())
    height    =   int(ui.high_entrada.text())
    max_iter  =   int(ui.max_iter_entrada.text())
    formula=   ui.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png")
    return "Mandelbrot generado"

def izquierda():
    mover= float(ui.mover_entrada.text())
    ui.xmin_entrada.setText(str(float(ui.xmin_entrada.text())-mover))
    ui.xmax_entrada.setText(str(float(ui.xmax_entrada.text())-mover))
    ui.ymin_entrada.setText(str(float(ui.ymin_entrada.text())))
    ui.ymax_entrada.setText(str(float(ui.ymax_entrada.text())))
    xmin      =   float(ui.xmin_entrada.text())
    xmax      =   float(ui.xmax_entrada.text())
    ymin      =   float(ui.ymin_entrada.text())
    ymax      =   float(ui.ymax_entrada.text())
    width     =   int(ui.width_entrada.text())
    height    =   int(ui.high_entrada.text())
    max_iter  =   int(ui.max_iter_entrada.text())
    formula=   ui.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png")
    return "Mandelbrot generado"

def derecha():
    mover= float(ui.mover_entrada.text())
    ui.xmin_entrada.setText(str(float(ui.xmin_entrada.text())+mover))
    ui.xmax_entrada.setText(str(float(ui.xmax_entrada.text())+mover))
    ui.ymin_entrada.setText(str(float(ui.ymin_entrada.text())))
    ui.ymax_entrada.setText(str(float(ui.ymax_entrada.text())))
    xmin      =   float(ui.xmin_entrada.text())
    xmax      =   float(ui.xmax_entrada.text())
    ymin      =   float(ui.ymin_entrada.text())
    ymax      =   float(ui.ymax_entrada.text())
    width     =   int(ui.width_entrada.text())
    height    =   int(ui.high_entrada.text())
    max_iter  =   int(ui.max_iter_entrada.text())
    formula=   ui.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png")
    return "Mandelbrot generado"

def subir():
    ui.xmin_entrada.setText(str(float(ui.xmin_entrada.text())))
    ui.xmax_entrada.setText(str(float(ui.xmax_entrada.text())))
    ui.ymin_entrada.setText(str(float(ui.ymin_entrada.text())-0.25))
    ui.ymax_entrada.setText(str(float(ui.ymax_entrada.text())-0.25))
    xmin      =   float(ui.xmin_entrada.text())
    xmax      =   float(ui.xmax_entrada.text())
    ymin      =   float(ui.ymin_entrada.text())
    ymax      =   float(ui.ymax_entrada.text())
    width     =   int(ui.width_entrada.text())
    height    =   int(ui.high_entrada.text())
    max_iter  =   int(ui.max_iter_entrada.text())
    formula=   ui.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png")
    return "Mandelbrot generado"

def bajar():
    ui.xmin_entrada.setText(str(float(ui.xmin_entrada.text())))
    ui.xmax_entrada.setText(str(float(ui.xmax_entrada.text())))
    ui.ymin_entrada.setText(str(float(ui.ymin_entrada.text())+0.25))
    ui.ymax_entrada.setText(str(float(ui.ymax_entrada.text())+0.25))
    xmin      =   float(ui.xmin_entrada.text())
    xmax      =   float(ui.xmax_entrada.text())
    ymin      =   float(ui.ymin_entrada.text())
    ymax      =   float(ui.ymax_entrada.text())
    width     =   int(ui.width_entrada.text())
    height    =   int(ui.high_entrada.text())
    max_iter  =   int(ui.max_iter_entrada.text())
    formula=   ui.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png")
    return "Mandelbrot generado"


import sys
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_Boundary()
ui.setupUi(MainWindow)

#########################
#   LINKEO DE BOTONES   #
#########################

ui.boton_hacer_fractal.clicked.connect(lambda : generar_mandelbrot())
ui.boton_hacer_zoom_in.clicked.connect(lambda : zoom_in())
ui.boton_hacer_zoom_out.clicked.connect(lambda : zoom_out())
ui.boton_bajar.clicked.connect(lambda : bajar())
ui.boton_subir.clicked.connect(lambda : subir())
ui.boton_derecha.clicked.connect(lambda : derecha())
ui.boton_izquierda.clicked.connect(lambda : izquierda())



MainWindow.show()
generar_mandelbrot()
sys.exit(app.exec())

    