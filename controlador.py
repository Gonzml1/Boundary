# pyuic6 -x "V:\ABoundary\gui\MandelbrotGUI.ui" -o "V:\ABoundary\gui\MandelbrotGUI.py"

import sys
from PyQt5 import QtWidgets
import gui.tema_oscuro as ts
from OpenGL.GL import *
from gui.gui import MainWindow 


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ts.tema_oscuro(app)
    
    window = MainWindow()
    window.show()

    sys.exit(app.exec())