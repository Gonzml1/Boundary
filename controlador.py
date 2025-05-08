# pyuic6 -x "V:\ABoundary\gui\MandelbrotGUI.ui"  -o "V:\ABoundary\gui\MandelbrotGUI.py"
import core.funciones_ui as md
from PyQt6 import QtWidgets
from gui.MandelbrotGUI import Ui_Boundary
import sys
import gui.tema_oscuro as ts

# fractales a añadir en un futuro (muy cercano)
# Newton-Raphson 

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

#########################
#   LINKEO DE BOTONES   #
#########################

md.linkeo_botones(ui)


if __name__ == "__main__":

    MainWindow.show()
    md.generar_mandelbrot(ui)
    sys.exit(app.exec())

