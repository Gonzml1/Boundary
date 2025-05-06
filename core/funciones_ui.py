import core.modulo_de_calculo as tf
from gui.MandelbrotGUI import Ui_Boundary
from PyQt6 import QtCore, QtGui, QtWidgets

#hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)

def linkeo_botones(ui=Ui_Boundary()):
    ui.boton_hacer_fractal.clicked.connect(lambda : generar_mandelbrot(ui))
    ui.boton_hacer_zoom_in.clicked.connect(lambda : zoom_in(ui))
    ui.boton_hacer_zoom_out.clicked.connect(lambda : zoom_out(ui))
    ui.boton_bajar.clicked.connect(lambda : bajar(ui))
    ui.boton_subir.clicked.connect(lambda : subir(ui))
    ui.boton_derecha.clicked.connect(lambda : derecha(ui))
    ui.boton_izquierda.clicked.connect(lambda : izquierda(ui))
    #ui.boton_guardar.clicked.connect(lambda : guardar(ui))


def mostrar_fractal(ruta_imagen,self=Ui_Boundary()):
    pixmap = QtGui.QPixmap(ruta_imagen)
    scene = QtWidgets.QGraphicsScene()
    scene.addPixmap(pixmap)
    self.Grafica_mostrar.setScene(scene)
    
def obtener_datos(self=Ui_Boundary()):
    cmap      =   str(self.cmap_comboBox.currentText())
    xmin      =   float(self.xmin_entrada.text())
    xmax      =   float(self.xmax_entrada.text())
    ymin      =   float(self.ymin_entrada.text())
    ymax      =   float(self.ymax_entrada.text())
    width     =   int(self.width_entrada.text())
    height    =   int(self.high_entrada.text())
    max_iter  =   int(self.max_iter_entrada.text())
    formula=   self.formula_entrada.text()
    
    return cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula

def hacer_dimensiones(zoom_factor,self=Ui_Boundary()):
    dimensiones= [0,0,0,0]
    dimensiones[0]      =   float(self.xmin_entrada.text())
    dimensiones[1]      =   float(self.xmax_entrada.text())
    dimensiones[2]      =   float(self.ymin_entrada.text())
    dimensiones[3]      =   float(self.ymax_entrada.text())
    
    centro_x= (dimensiones[0]+dimensiones[1])/2
    centro_y= (dimensiones[2]+dimensiones[3])/2
    
    dimensiones[0] = centro_x - (centro_x - dimensiones[0]) *zoom_factor
    dimensiones[1] = centro_x + (dimensiones[1] - centro_x) *zoom_factor
    dimensiones[2] = centro_y - (centro_y - dimensiones[2]) *zoom_factor
    dimensiones[3] = centro_y + (dimensiones[3] - centro_y) *zoom_factor
    
    return dimensiones

def generar_mandelbrot(self=Ui_Boundary()):
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula = obtener_datos(self)
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\\ABoundary\\recursos\\mandelbrot.png", width, height,cmap,dpi=100)
    mostrar_fractal("V:\\ABoundary\\recursos\\mandelbrot.png",self)
    return "Mandelbrot generado"

def zoom_in(self=Ui_Boundary()):
    zoom_in_factor = float(self.zoom_in_factor_entrada.text())
    dimensiones= hacer_dimensiones(zoom_in_factor,self)
    
    self.xmin_entrada.setText(str(dimensiones[0]))
    self.xmax_entrada.setText(str(dimensiones[1]))
    self.ymin_entrada.setText(str(dimensiones[2]))
    self.ymax_entrada.setText(str(dimensiones[3]))

    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula = obtener_datos(self)
    
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\\ABoundary\\recursos\\mandelbrot.png", width, height,cmap,dpi=100)
    mostrar_fractal("V:\\ABoundary\\recursos\\mandelbrot.png",self)
    return "Mandelbrot generado"

def zoom_out(self=Ui_Boundary()):
    zoom_out_factor = float(self.zoom_out_factor_entrada.text())
    dimensiones= hacer_dimensiones(zoom_out_factor, self)

    self.xmin_entrada.setText(str(dimensiones[0]))
    self.xmax_entrada.setText(str(dimensiones[1]))
    self.ymin_entrada.setText(str(dimensiones[2]))
    self.ymax_entrada.setText(str(dimensiones[3]))
    
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula = obtener_datos(self)
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\\ABoundary\\recursos\\mandelbrot.png", width, height,cmap,dpi=100)
    mostrar_fractal("V:\\ABoundary\\recursos\\mandelbrot.png",self)
    return "Mandelbrot generado"

def izquierda(self=Ui_Boundary()):
    mover= float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())-mover))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())-mover))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())))
    
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula = obtener_datos(self)
    
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\\ABoundary\\recursos\\mandelbrot.png", width, height,cmap,dpi=100)
    mostrar_fractal("V:\\ABoundary\\recursos\\mandelbrot.png",self)
    return "Mandelbrot generado"

def derecha(self=Ui_Boundary()):
    mover = float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())+mover))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())+mover))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())))
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula = obtener_datos(self)
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\\ABoundary\\recursos\\mandelbrot.png", width, height,cmap,dpi=100)
    mostrar_fractal("V:\\ABoundary\\recursos\\mandelbrot.png",self)
    return "Mandelbrot generado"

def subir(self=Ui_Boundary()):
    mover= float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())-mover))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())-mover))
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula = obtener_datos(self)
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\\ABoundary\\recursos\\mandelbrot.png", width, height, cmap, dpi=100)
    mostrar_fractal("V:\\ABoundary\\recursos\\mandelbrot.png",self)
    return "Mandelbrot generado"

def bajar(self=Ui_Boundary()):
    mover= float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())+mover))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())+mover))
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula = obtener_datos(self)
    M = tf.hacer_mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"V:\\ABoundary\\recursos\\mandelbrot.png", width, height, cmap, dpi=100)
    mostrar_fractal("V:\\ABoundary\\recursos\\mandelbrot.png",self)
    return "Mandelbrot generado"
