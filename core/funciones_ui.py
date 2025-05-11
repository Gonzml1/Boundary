import core.modulo_de_calculo_fractales as tf
from gui.MandelbrotGUI import Ui_Boundary
from PyQt5 import QtGui, QtWidgets
from core.modulo_opengl import MandelbrotWidget
#calcular_fractal(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag)

def mostrar_fractal_opengl(self=Ui_Boundary):
    try:
        # Obtener valores desde los campos de entrada
        cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag = obtener_datos(self)


        mandelbrot_widget = MandelbrotWidget(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag,self)


        if self.grafico_openGLWidget.layout() is None:
            layout = QtWidgets.QVBoxLayout(self.grafico_openGLWidget)
            layout.setContentsMargins(0, 0, 0, 0)
            self.grafico_openGLWidget.setLayout(layout)
        else:
            layout = self.grafico_openGLWidget.layout()

        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        layout.addWidget(mandelbrot_widget)

    except ValueError:
        print("Error: Asegurate de que los campos tengan valores numéricos válidos.")
    
def linkeo_botones(ui=Ui_Boundary()):
    ui.boton_hacer_fractal.clicked.connect(lambda : generar_mandelbrot(ui))
    ui.boton_hacer_zoom_in.clicked.connect(lambda : zoom_in(ui))
    ui.boton_hacer_zoom_out.clicked.connect(lambda : zoom_out(ui))
    ui.boton_bajar.clicked.connect(lambda : bajar(ui))
    ui.boton_subir.clicked.connect(lambda : subir(ui))
    ui.boton_derecha.clicked.connect(lambda : derecha(ui))
    ui.boton_izquierda.clicked.connect(lambda : izquierda(ui))
    ui.boton_bajar_derecha.clicked.connect(lambda : bajar_derecha(ui))
    ui.boton_bajar_izquierda.clicked.connect(lambda : bajar_izquierda(ui))
    ui.boton_subir_derecha.clicked.connect(lambda : subir_derecha(ui))
    ui.boton_subir_izquierda.clicked.connect(lambda : subir_izquierda(ui))
    ui.boton_no_hace_nada.clicked.connect(lambda : no_hace_nada(ui))
    ui.boton_resetear.clicked.connect(lambda : resetear_entrada(ui))
    ui.boton_dividir.clicked.connect(lambda : dividir(ui))
    ui.boton_duplicar.clicked.connect(lambda : duplicar(ui))
    #ui.boton_guardar.clicked.connect(lambda : guardar(ui))

def resetear_entrada(self=Ui_Boundary()):
    self.xmin_entrada.setText("-2.0")
    self.xmax_entrada.setText("2.0")
    self.ymin_entrada.setText("-0.9")
    self.ymax_entrada.setText("0.9")
    self.width_entrada.setText("800")
    self.high_entrada.setText("600")
    self.max_iter_entrada.setText("256")
    self.real_julia_entrada.setText("0.0")
    self.im_julia_entrada.setText("0.0")
    self.formula_entrada.setText("z**2 + c")
    self.mover_entrada.setText("0.25")
    self.zoom_in_factor_entrada.setText("0.5")
    self.zoom_out_factor_entrada.setText("2.0")
    return print("Entradas reseteadas")

def duplicar(self=Ui_Boundary()):
    self.max_iter_entrada.setText(str(int(int(self.max_iter_entrada.text())*2)))
    
def dividir(self=Ui_Boundary()):
    self.max_iter_entrada.setText(str(int(int(self.max_iter_entrada.text())/2)))

def no_hace_nada(self=Ui_Boundary()):
    return print("Te dije que no hace nada")

def mostrar_fractal(ruta_imagen,self=Ui_Boundary()):
    pixmap = QtGui.QPixmap(ruta_imagen)
    scene = QtWidgets.QGraphicsScene()
    scene.addPixmap(pixmap)
    self.Grafica_mostrar.setScene(scene)
    
def obtener_datos(self=Ui_Boundary()):
    cmap          =   str(self.cmap_comboBox.currentText())
    xmin          =   float(self.xmin_entrada.text())
    xmax          =   float(self.xmax_entrada.text())
    ymin          =   float(self.ymin_entrada.text())
    ymax          =   float(self.ymax_entrada.text())
    width         =   int(self.width_entrada.text())
    height        =   int(self.high_entrada.text())
    max_iter      =   int(self.max_iter_entrada.text())
    tipo_calculo  =   str(self.tipo_calculo_comboBox.currentText())
    tipo_fractal  =   str(self.tipo_fractal_comboBox.currentText())
    real          =   float(self.real_julia_entrada.text())
    imag          =   float(self.im_julia_entrada.text())
    formula       =   str(self.formula_entrada.text())
    
    return cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag

def hacer_dimensiones(zoom_factor,self=Ui_Boundary()):
    dimensiones= [0,0,0,0]
    dimensiones[0]    =   float(self.xmin_entrada.text())
    dimensiones[1]    =   float(self.xmax_entrada.text())
    dimensiones[2]    =   float(self.ymin_entrada.text())
    dimensiones[3]    =   float(self.ymax_entrada.text())
    
    centro_x= (dimensiones[0]+dimensiones[1])/2
    centro_y= (dimensiones[2]+dimensiones[3])/2
    
    dimensiones[0] = centro_x - (centro_x - dimensiones[0]) *zoom_factor
    dimensiones[1] = centro_x + (dimensiones[1] - centro_x) *zoom_factor
    dimensiones[2] = centro_y - (centro_y - dimensiones[2]) *zoom_factor
    dimensiones[3] = centro_y + (dimensiones[3] - centro_y) *zoom_factor
    
    return dimensiones

def calcular_guardar_mostrar_fractal(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag,self = Ui_Boundary()):
    M = tf.calcular_fractal(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag)
    tf.guardar_mandelbrot(M,xmin,xmax,ymin,ymax,"recursos\mandelbrot.png", width, height,cmap,dpi=100)
    mostrar_fractal("recursos\mandelbrot.png",self)
    return

def generar_mandelbrot(self=Ui_Boundary()):
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag = obtener_datos(self)
    calcular_guardar_mostrar_fractal(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, self)
    return "Mandelbrot generado"

#########################
#         ZOOM          #
#########################

def zoom_in(self=Ui_Boundary()):
    zoom_in_factor = float(self.zoom_in_factor_entrada.text())
    dimensiones= hacer_dimensiones(zoom_in_factor,self)
    
    self.xmin_entrada.setText(str(dimensiones[0]))
    self.xmax_entrada.setText(str(dimensiones[1]))
    self.ymin_entrada.setText(str(dimensiones[2]))
    self.ymax_entrada.setText(str(dimensiones[3]))

    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag = obtener_datos(self)
    calcular_guardar_mostrar_fractal(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, self)
    return "Mandelbrot generado"

def zoom_out(self=Ui_Boundary()):
    zoom_out_factor = float(self.zoom_out_factor_entrada.text())
    dimensiones= hacer_dimensiones(zoom_out_factor, self)

    self.xmin_entrada.setText(str(dimensiones[0]))
    self.xmax_entrada.setText(str(dimensiones[1]))
    self.ymin_entrada.setText(str(dimensiones[2]))
    self.ymax_entrada.setText(str(dimensiones[3]))
    
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag = obtener_datos(self)
    calcular_guardar_mostrar_fractal(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, self)
    return "Mandelbrot generado"

#########################
#      MOVIMIENTO       #
#########################

def izquierda(self=Ui_Boundary()):
    mover= float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())-mover))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())-mover))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())))
    
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag = obtener_datos(self)
    
    calcular_guardar_mostrar_fractal(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, self)
    return "Mandelbrot generado"

def derecha(self=Ui_Boundary()):
    mover = float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())+mover))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())+mover))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())))
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag = obtener_datos(self)
    calcular_guardar_mostrar_fractal(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, self)
    return "Mandelbrot generado"

def subir(self=Ui_Boundary()):
    mover= float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())-mover))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())-mover))
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag = obtener_datos(self)
    calcular_guardar_mostrar_fractal(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, self)
    return "Mandelbrot generado"

def bajar(self=Ui_Boundary()):
    mover= float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())+mover))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())+mover))
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag = obtener_datos(self)
    calcular_guardar_mostrar_fractal(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, self)
    return "Mandelbrot generado"

def bajar_izquierda(self=Ui_Boundary()):
    mover= float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())-mover/2**(1/2)))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())-mover/2**(1/2)))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())+mover/2**(1/2)))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())+mover/2**(1/2)))
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag = obtener_datos(self)
    calcular_guardar_mostrar_fractal(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, self)
    return "Mandelbrot generado"

def bajar_derecha(self=Ui_Boundary()):
    mover = float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())+mover/2**(1/2)))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())+mover/2**(1/2)))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())+mover/2**(1/2)))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())+mover/2**(1/2)))    
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag = obtener_datos(self)
    calcular_guardar_mostrar_fractal(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, self)
    return "Mandelbrot generado"

def subir_izquierda(self=Ui_Boundary()):
    mover= float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())-mover/2**(1/2)))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())-mover/2**(1/2)))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())-mover/2**(1/2)))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())-mover/2**(1/2)))
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag = obtener_datos(self)
    calcular_guardar_mostrar_fractal(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, self)
    return "Mandelbrot generado"

def subir_derecha(self=Ui_Boundary()):
    mover= float(self.mover_entrada.text())
    self.xmin_entrada.setText(str(float(self.xmin_entrada.text())+mover/2**(1/2)))
    self.xmax_entrada.setText(str(float(self.xmax_entrada.text())+mover/2**(1/2)))
    self.ymin_entrada.setText(str(float(self.ymin_entrada.text())-mover/2**(1/2)))
    self.ymax_entrada.setText(str(float(self.ymax_entrada.text())-mover/2**(1/2)))
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag = obtener_datos(self)
    calcular_guardar_mostrar_fractal(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, self)
    return "Mandelbrot generado"