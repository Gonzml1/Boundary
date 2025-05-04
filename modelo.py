import modulo_de_calculo as tf
from MandelbrotGUI import Ui_Boundary
from PyQt6 import QtCore, QtGui, QtWidgets

#hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)

def mostrar_fractal(ruta_imagen,self=Ui_Boundary()):
    pixmap = QtGui.QPixmap(ruta_imagen)
    scene = QtWidgets.QGraphicsScene()
    scene.addPixmap(pixmap)
    self.Grafica_mostrar.setScene(scene)


def generar_mandelbrot(self=Ui_Boundary()):
    xmin      =   float(self.xmin_entrada.text())
    xmax      =   float(self.xmax_entrada.text())
    ymin      =   float(self.ymin_entrada.text())
    ymax      =   float(self.ymax_entrada.text())
    width     =   int(self.width_entrada.text())
    height    =   int(self.high_entrada.text())
    max_iter  =   int(self.max_iter_entrada.text())
    formula=   self.formula_entrada.text()
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"

def zoom_in(self=Ui_Boundary()):
    dimensiones= [0,0,0,0]
    dimensiones[0]      =   float(self.xmin_entrada.text())
    dimensiones[1]      =   float(self.xmax_entrada.text())
    dimensiones[2]      =   float(self.ymin_entrada.text())
    dimensiones[3]      =   float(self.ymax_entrada.text())
    
    centro_x= (dimensiones[0]+dimensiones[1])/2
    centro_y= (dimensiones[2]+dimensiones[3])/2
    zoom_in_factor = float(self.zoom_in_factor_entrada.text())
    
    dimensiones[0] = centro_x - (centro_x - dimensiones[0]) *zoom_in_factor
    dimensiones[1] = centro_x + (dimensiones[1] - centro_x) *zoom_in_factor  
    dimensiones[2] = centro_y - (centro_y - dimensiones[2]) *zoom_in_factor  
    dimensiones[3] = centro_y + (dimensiones[3] - centro_y) *zoom_in_factor
    
    self.xmin_entrada.setText(str(dimensiones[0]))
    self.xmax_entrada.setText(str(dimensiones[1]))
    self.ymin_entrada.setText(str(dimensiones[2]))
    self.ymax_entrada.setText(str(dimensiones[3]))

    xmin      =   float(self.xmin_entrada.text())
    xmax      =   float(self.xmax_entrada.text())
    ymin      =   float(self.ymin_entrada.text())
    ymax      =   float(self.ymax_entrada.text())
    width     =   int(self.width_entrada.text())
    height    =   int(self.high_entrada.text())
    max_iter  =   int(self.max_iter_entrada.text())
    formula=   self.formula_entrada.text()
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"

def zoom_out(self=Ui_Boundary()):
    dimensiones= [0,0,0,0]
    dimensiones[0]      =   float(self.xmin_entrada.text())
    dimensiones[1]      =   float(self.xmax_entrada.text())
    dimensiones[2]      =   float(self.ymin_entrada.text())
    dimensiones[3]      =   float(self.ymax_entrada.text())
    
    centro_x= (dimensiones[0]+dimensiones[1])/2
    centro_y= (dimensiones[2]+dimensiones[3])/2
    zoom_out_factor = float(self.zoom_out_factor_entrada.text())
    
    dimensiones[0] = centro_x - (centro_x - dimensiones[0]) *zoom_out_factor
    dimensiones[1] = centro_x + (dimensiones[1] - centro_x) *zoom_out_factor
    dimensiones[2] = centro_y - (centro_y - dimensiones[2]) *zoom_out_factor
    dimensiones[3] = centro_y + (dimensiones[3] - centro_y) *zoom_out_factor

    self.xmin_entrada.setText(str(dimensiones[0]))
    self.xmax_entrada.setText(str(dimensiones[1]))
    self.ymin_entrada.setText(str(dimensiones[2]))
    self.ymax_entrada.setText(str(dimensiones[3]))
    
    xmin      =   float(self.xmin_entrada.text())
    xmax      =   float(self.xmax_entrada.text())
    ymin      =   float(self.ymin_entrada.text())
    ymax      =   float(self.ymax_entrada.text())
    width     =   int(self.width_entrada.text())
    height    =   int(self.high_entrada.text())
    max_iter  =   int(self.max_iter_entrada.text())
    formula=   self.formula_entrada.text()
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"

def izquierda(self=Ui_Boundary()):
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
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"

def derecha(self=Ui_Boundary()):
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
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"

def subir(self=Ui_Boundary()):
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
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"

def bajar(self=Ui_Boundary()):
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
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\ABoundary\mandelbrot.png", width, height, dpi=100)
    mostrar_fractal("V:\ABoundary\mandelbrot.png",self)
    return "Mandelbrot generado"
