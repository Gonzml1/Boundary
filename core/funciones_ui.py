import core.modulo_de_calculo_fractales as tf
from gui.MandelbrotGUI import Ui_Boundary
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from core.modulo_opengl import MandelbrotWidget
import matplotlib.pyplot as plt
from core.modulo_de_calculo_fractales import calculos_mandelbrot

#calcular_fractal(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag)
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




