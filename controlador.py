# pyuic6 -x "V:\ABoundary\gui\MandelbrotGUI.ui"  -o "V:\ABoundary\gui\MandelbrotGUI.py"
import core.funciones_ui as md
from PyQt5 import QtWidgets
from gui.MandelbrotGUI import Ui_Boundary
import sys
import gui.tema_oscuro as ts
from OpenGL.GL import *

# fractales a añadir en un futuro (muy cercano)
# Newton-Raphson  
# Mejorar interfaz grafica

 #########################
#      TEMA OSCURO      #
#########################

app = QtWidgets.QApplication(sys.argv)
ts.tema_oscuro(app)

#########################
#     Creacion Qmain    #
#########################

MainWindow = QtWidgets.QMainWindow()
ui = Ui_Boundary()
ui.setupUi(MainWindow)
md.mostrar_fractal_opengl(ui)

#########################
#   LINKEO DE BOTONES   #
#########################

md.linkeo_botones(ui)

if __name__ == "__main__":

    MainWindow.show()
    sys.exit(app.exec())

