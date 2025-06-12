import core.modulo_de_calculo_fractales as tf
from gui.MandelbrotGUI import Ui_Boundary
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from core.modulo_opengl import MandelbrotWidget
import matplotlib.pyplot as plt
from core.modulo_de_calculo_fractales import calculos_mandelbrot

#calcular_fractal(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag)

def guardar_imagen(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, ui):
    ruta, _ = QFileDialog.getSaveFileName(
        None,
        "Guardar imagen",
        "fractal.png",
        "Imágenes PNG (*.png);;JPEG (*.jpg *.jpeg);;Todos los archivos (*)"
    )
    calculos = calculos_mandelbrot(
        xmin, xmax, ymin, ymax, width, height, max_iter,
        formula, tipo_calculo, tipo_fractal, real, imag, ui
    )
    if ruta:
        # Reemplazá esto por tu lógica de fractal real
        calculos.actualizar_fractal()
        imagen_array = calculos.calcular_fractal()
        plt.imsave(ruta, imagen_array,cmap='twilight_shifted')
        print(f"Imagen guardada en: {ruta}")
            

def mostrar_fractal_opengl(self=Ui_Boundary()):
    try:
        # Obtener valores desde los campos de entrada
        cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, zoom_in, zoom_out = obtener_datos(self)
        
        mandelbrot_widget = MandelbrotWidget(cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag,zoom_in, zoom_out,self)

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
        return mandelbrot_widget
    except ValueError:
        print("Error: Asegurate de que los campos tengan valores numéricos válidos.")
    
def linkeo_botones(ui=Ui_Boundary()):
    cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, zoom_in, zoom_out = obtener_datos(ui)
    ui.boton_resetear.clicked.connect(lambda : resetear_entrada(ui))
    ui.boton_dividir.clicked.connect(lambda : dividir(ui))
    ui.boton_duplicar.clicked.connect(lambda : duplicar(ui))
    ui.boton_guardar.clicked.connect(
        lambda: guardar_imagen(
            xmin, xmax, ymin, ymax, width, height, max_iter,
            formula, tipo_calculo, tipo_fractal, real, imag, ui
        )
    )
    ui.slider_iteraciones.valueChanged.connect(lambda value: ui.max_iter_entrada.setText(str(value)))
    

def resetear_entrada(self=Ui_Boundary()):
    self.xmin_entrada.setText("-2.0")
    self.xmax_entrada.setText("2.0")
    self.ymin_entrada.setText("-0.9")
    self.ymax_entrada.setText("0.9")
    self.width_entrada.setText("1000")
    self.high_entrada.setText("600")
    self.max_iter_entrada.setText("256")
    self.real_julia_entrada.setText("0.0")
    self.im_julia_entrada.setText("0.0")
    self.formula_entrada.setText("z = z**2 + C")
    self.zoom_in_factor_entrada.setText("0.5")
    self.zoom_out_factor_entrada.setText("2.0")
    return print("Entradas reseteadas")

def duplicar(self=Ui_Boundary()):
    self.max_iter_entrada.setText(str(int(int(self.max_iter_entrada.text())*2)))
    
def dividir(self=Ui_Boundary()):
    self.max_iter_entrada.setText(str(int(int(self.max_iter_entrada.text())/2)))

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
    zoom_out      =   float(self.zoom_out_factor_entrada.text())
    zoom_in       =   float(self.zoom_in_factor_entrada.text())

    
    return cmap, xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag, zoom_in, zoom_out   




