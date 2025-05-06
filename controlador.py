# pyuic6 -x "V:\ABoundary\gui\MandelbrotGUI.ui"  -o "V:\ABoundary\gui\MandelbrotGUI.py"
import core.funciones_ui as md
from PyQt6 import QtWidgets
from gui.MandelbrotGUI import Ui_Boundary
import sys
from gui.tema_oscuro import dark_palette

app = QtWidgets.QApplication(sys.argv)
QtWidgets.QApplication.setStyle("Fusion")

app.setPalette(dark_palette)

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

