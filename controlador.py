# pyuic6 -x "V:\ABoundary\MandelbrotGUI.ui"  -o "V:\ABoundary\MandelbrotGUI.py"
import core.funciones_ui as md
from PyQt6 import QtCore, QtGui, QtWidgets
from gui.MandelbrotGUI import Ui_Boundary
from PyQt6.QtGui import QImage, QPixmap,QPalette, QColor
import sys
from PyQt6.QtWidgets import QApplication


app = QtWidgets.QApplication(sys.argv)
QtWidgets.QApplication.setStyle("Fusion")

dark_palette = QPalette()
dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
dark_palette.setColor(QPalette.ColorRole.WindowText, QtGui.QColor("white"))
dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QtGui.QColor("white"))
dark_palette.setColor(QPalette.ColorRole.ToolTipText, QtGui.QColor("white"))
dark_palette.setColor(QPalette.ColorRole.Text, QtGui.QColor("white"))
dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
dark_palette.setColor(QPalette.ColorRole.ButtonText, QtGui.QColor("white"))
dark_palette.setColor(QPalette.ColorRole.BrightText, QtGui.QColor("red"))
dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(142, 45, 197))
dark_palette.setColor(QPalette.ColorRole.HighlightedText, QtGui.QColor("black"))

app.setPalette(dark_palette)

MainWindow = QtWidgets.QMainWindow()
ui = Ui_Boundary()
ui.setupUi(MainWindow)

#########################
#   LINKEO DE BOTONES   #
#########################

ui.boton_hacer_fractal.clicked.connect(lambda : md.generar_mandelbrot(ui))
ui.boton_hacer_zoom_in.clicked.connect(lambda : md.zoom_in(ui))
ui.boton_hacer_zoom_out.clicked.connect(lambda : md.zoom_out(ui))
ui.boton_bajar.clicked.connect(lambda : md.bajar(ui))
ui.boton_subir.clicked.connect(lambda : md.subir(ui))
ui.boton_derecha.clicked.connect(lambda : md.derecha(ui))
ui.boton_izquierda.clicked.connect(lambda : md.izquierda(ui))



MainWindow.show()
md.generar_mandelbrot(ui)
sys.exit(app.exec())

