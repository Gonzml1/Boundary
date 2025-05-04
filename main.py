import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from gui.MandelbrotGUI import Ui_Boundary
from controlador import MandelbrotApp

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Boundary()
        self.ui.setupUi(self)
        self.app_logic = MandelbrotApp(self.ui)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()