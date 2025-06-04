from PyQt5 import QtGui
from PyQt5.QtGui import QPalette, QColor
from PyQt5 import QtWidgets
import sys


def tema_oscuro(app=None):
    """Aplica un tema oscuro a la aplicación Qt dada.

    Si ``app`` es ``None`` se intenta usar la instancia de
    :class:`~PyQt5.QtWidgets.QApplication` existente. En caso de no
    existir, se crea una nueva.
    """
    if app is None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
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

    QtWidgets.QApplication.setStyle("Fusion")
    app.setPalette(dark_palette)


