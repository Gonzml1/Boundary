import TodasFunciones2 as tf
from MandelbrotGUI import Ui_Boundary
from PyQt6 import QtCore, QtGui, QtWidgets

def mostrar_fractal(ruta_imagen,self):
    pixmap = QtGui.QPixmap(ruta_imagen)
    scene = QtWidgets.QGraphicsScene()
    scene.addPixmap(pixmap)
    self.Grafica_mostrar.setScene(scene)


def generar_mandelbrot(self):
    xmin      =   float(self.xmin_entrada.text())
    xmax      =   float(self.xmax_entrada.text())
    ymin      =   float(self.ymin_entrada.text())
    ymax      =   float(self.ymax_entrada.text())
    width     =   int(self.width_entrada.text())
    height    =   int(self.high_entrada.text())
    max_iter  =   int(self.max_iter_entrada.text())
    formula=   self.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"

def zoom_in(self):
    zoom_in_factor= float(self.zoom_in_factor_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())*zoom_in_factor))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())*zoom_in_factor))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())*zoom_in_factor))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())*zoom_in_factor))
    xmin      =   float(self.xmin_entrada.text())
    xmax      =   float(self.xmax_entrada.text())
    ymin      =   float(self.ymin_entrada.text())
    ymax      =   float(self.ymax_entrada.text())
    width     =   int(self.width_entrada.text())
    height    =   int(self.high_entrada.text())
    max_iter  =   int(self.max_iter_entrada.text())
    formula=   self.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"

def zoom_out(self):
    zoom_out_factor= float(self.zoom_out_factor_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())*zoom_out_factor))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())*zoom_out_factor))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())*zoom_out_factor))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())*zoom_out_factor))
    xmin      =   float(self.xmin_entrada.text())
    xmax      =   float(self.xmax_entrada.text())
    ymin      =   float(self.ymin_entrada.text())
    ymax      =   float(self.ymax_entrada.text())
    width     =   int(self.width_entrada.text())
    height    =   int(self.high_entrada.text())
    max_iter  =   int(self.max_iter_entrada.text())
    formula=   self.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"

def izquierda(self):
    mover= float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())-mover))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())-mover))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())))
    xmin      =   float(self.xmin_entrada.text())
    xmax      =   float(self.xmax_entrada.text())
    ymin      =   float(self.ymin_entrada.text())
    ymax      =   float(self.ymax_entrada.text())
    width     =   int(self.width_entrada.text())
    height    =   int(self.high_entrada.text())
    max_iter  =   int(self.max_iter_entrada.text())
    formula=   self.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"

def derecha(self):
    mover= float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())+mover))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())+mover))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())))
    xmin      =   float(self.xmin_entrada.text())
    xmax      =   float(self.xmax_entrada.text())
    ymin      =   float(self.ymin_entrada.text())
    ymax      =   float(self.ymax_entrada.text())
    width     =   int(self.width_entrada.text())
    height    =   int(self.high_entrada.text())
    max_iter  =   int(self.max_iter_entrada.text())
    formula=   self.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"

def subir(self):
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())-0.25))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())-0.25))
    xmin      =   float(self.xmin_entrada.text())
    xmax      =   float(self.xmax_entrada.text())
    ymin      =   float(self.ymin_entrada.text())
    ymax      =   float(self.ymax_entrada.text())
    width     =   int(self.width_entrada.text())
    height    =   int(self.high_entrada.text())
    max_iter  =   int(self.max_iter_entrada.text())
    formula=   self.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"

def bajar(self):
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())+0.25))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())+0.25))
    xmin      =   float(self.xmin_entrada.text())
    xmax      =   float(self.xmax_entrada.text())
    ymin      =   float(self.ymin_entrada.text())
    ymax      =   float(self.ymax_entrada.text())
    width     =   int(self.width_entrada.text())
    height    =   int(self.high_entrada.text())
    max_iter  =   int(self.max_iter_entrada.text())
    formula=   self.formula_entrada.text()
    M = tf.hacer_mandelbrot_con_entrada(xmin, xmax, ymin, ymax,width,height,max_iter,formula)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"