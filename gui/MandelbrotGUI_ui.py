# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MandelbrotGUI.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMenuBar, QPushButton,
    QSizePolicy, QSlider, QStatusBar, QVBoxLayout,
    QWidget)

class Ui_Boundary(object):
    def setupUi(self, Boundary):
        if not Boundary.objectName():
            Boundary.setObjectName(u"Boundary")
        Boundary.resize(1300, 660)
        icon = QIcon()
        icon.addFile(u"../Iconos/assets2Ftask_01jsfxefnefwvtb960bws6yaa72Fimg_0.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Boundary.setWindowIcon(icon)
        Boundary.setAutoFillBackground(False)
        self.centralwidget = QWidget(Boundary)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayoutWidget = QWidget(self.centralwidget)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(1130, 180, 160, 391))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.xmin_label = QLabel(self.verticalLayoutWidget)
        self.xmin_label.setObjectName(u"xmin_label")

        self.verticalLayout.addWidget(self.xmin_label)

        self.xmin_entrada = QLineEdit(self.verticalLayoutWidget)
        self.xmin_entrada.setObjectName(u"xmin_entrada")

        self.verticalLayout.addWidget(self.xmin_entrada)

        self.xmax_label = QLabel(self.verticalLayoutWidget)
        self.xmax_label.setObjectName(u"xmax_label")

        self.verticalLayout.addWidget(self.xmax_label)

        self.xmax_entrada = QLineEdit(self.verticalLayoutWidget)
        self.xmax_entrada.setObjectName(u"xmax_entrada")

        self.verticalLayout.addWidget(self.xmax_entrada)

        self.ymin_label = QLabel(self.verticalLayoutWidget)
        self.ymin_label.setObjectName(u"ymin_label")

        self.verticalLayout.addWidget(self.ymin_label)

        self.ymin_entrada = QLineEdit(self.verticalLayoutWidget)
        self.ymin_entrada.setObjectName(u"ymin_entrada")

        self.verticalLayout.addWidget(self.ymin_entrada)

        self.ymax_label = QLabel(self.verticalLayoutWidget)
        self.ymax_label.setObjectName(u"ymax_label")

        self.verticalLayout.addWidget(self.ymax_label)

        self.ymax_entrada = QLineEdit(self.verticalLayoutWidget)
        self.ymax_entrada.setObjectName(u"ymax_entrada")

        self.verticalLayout.addWidget(self.ymax_entrada)

        self.width_label = QLabel(self.verticalLayoutWidget)
        self.width_label.setObjectName(u"width_label")

        self.verticalLayout.addWidget(self.width_label)

        self.width_entrada = QLineEdit(self.verticalLayoutWidget)
        self.width_entrada.setObjectName(u"width_entrada")

        self.verticalLayout.addWidget(self.width_entrada)

        self.high_label = QLabel(self.verticalLayoutWidget)
        self.high_label.setObjectName(u"high_label")

        self.verticalLayout.addWidget(self.high_label)

        self.high_entrada = QLineEdit(self.verticalLayoutWidget)
        self.high_entrada.setObjectName(u"high_entrada")

        self.verticalLayout.addWidget(self.high_entrada)

        self.max_iter_label = QLabel(self.verticalLayoutWidget)
        self.max_iter_label.setObjectName(u"max_iter_label")

        self.verticalLayout.addWidget(self.max_iter_label)

        self.max_iter_entrada = QLineEdit(self.verticalLayoutWidget)
        self.max_iter_entrada.setObjectName(u"max_iter_entrada")

        self.verticalLayout.addWidget(self.max_iter_entrada)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.boton_duplicar = QPushButton(self.verticalLayoutWidget)
        self.boton_duplicar.setObjectName(u"boton_duplicar")

        self.horizontalLayout.addWidget(self.boton_duplicar)

        self.boton_dividir = QPushButton(self.verticalLayoutWidget)
        self.boton_dividir.setObjectName(u"boton_dividir")

        self.horizontalLayout.addWidget(self.boton_dividir)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.layoutWidget = QWidget(self.centralwidget)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(980, 180, 131, 211))
        self.verticalLayout_2 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formula_label = QLabel(self.layoutWidget)
        self.formula_label.setObjectName(u"formula_label")

        self.verticalLayout_2.addWidget(self.formula_label)

        self.formula_entrada = QLineEdit(self.layoutWidget)
        self.formula_entrada.setObjectName(u"formula_entrada")

        self.verticalLayout_2.addWidget(self.formula_entrada)

        self.zoom_out_factor_label = QLabel(self.layoutWidget)
        self.zoom_out_factor_label.setObjectName(u"zoom_out_factor_label")

        self.verticalLayout_2.addWidget(self.zoom_out_factor_label)

        self.zoom_out_factor_entrada = QLineEdit(self.layoutWidget)
        self.zoom_out_factor_entrada.setObjectName(u"zoom_out_factor_entrada")

        self.verticalLayout_2.addWidget(self.zoom_out_factor_entrada)

        self.zoom_in_factor_label = QLabel(self.layoutWidget)
        self.zoom_in_factor_label.setObjectName(u"zoom_in_factor_label")

        self.verticalLayout_2.addWidget(self.zoom_in_factor_label)

        self.zoom_in_factor_entrada = QLineEdit(self.layoutWidget)
        self.zoom_in_factor_entrada.setObjectName(u"zoom_in_factor_entrada")

        self.verticalLayout_2.addWidget(self.zoom_in_factor_entrada)

        self.exponente_label = QLabel(self.layoutWidget)
        self.exponente_label.setObjectName(u"exponente_label")

        self.verticalLayout_2.addWidget(self.exponente_label)

        self.horizontalSlider = QSlider(self.layoutWidget)
        self.horizontalSlider.setObjectName(u"horizontalSlider")
        self.horizontalSlider.setMaximum(20)
        self.horizontalSlider.setSliderPosition(2)
        self.horizontalSlider.setOrientation(Qt.Horizontal)

        self.verticalLayout_2.addWidget(self.horizontalSlider)

        self.layoutWidget_2 = QWidget(self.centralwidget)
        self.layoutWidget_2.setObjectName(u"layoutWidget_2")
        self.layoutWidget_2.setGeometry(QRect(830, 180, 131, 311))
        self.verticalLayout_4 = QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.generador_label = QLabel(self.layoutWidget_2)
        self.generador_label.setObjectName(u"generador_label")

        self.verticalLayout_4.addWidget(self.generador_label)

        self.generador_comboBox = QComboBox(self.layoutWidget_2)
        self.generador_comboBox.addItem("")
        self.generador_comboBox.addItem("")
        self.generador_comboBox.setObjectName(u"generador_comboBox")

        self.verticalLayout_4.addWidget(self.generador_comboBox)

        self.tipo_fractal_label = QLabel(self.layoutWidget_2)
        self.tipo_fractal_label.setObjectName(u"tipo_fractal_label")

        self.verticalLayout_4.addWidget(self.tipo_fractal_label)

        self.tipo_fractal_comboBox = QComboBox(self.layoutWidget_2)
        self.tipo_fractal_comboBox.addItem("")
        self.tipo_fractal_comboBox.addItem("")
        self.tipo_fractal_comboBox.addItem("")
        self.tipo_fractal_comboBox.addItem("")
        self.tipo_fractal_comboBox.addItem("")
        self.tipo_fractal_comboBox.setObjectName(u"tipo_fractal_comboBox")

        self.verticalLayout_4.addWidget(self.tipo_fractal_comboBox)

        self.tipo_calculo_label = QLabel(self.layoutWidget_2)
        self.tipo_calculo_label.setObjectName(u"tipo_calculo_label")

        self.verticalLayout_4.addWidget(self.tipo_calculo_label)

        self.tipo_calculo_comboBox = QComboBox(self.layoutWidget_2)
        self.tipo_calculo_comboBox.addItem("")
        self.tipo_calculo_comboBox.addItem("")
        self.tipo_calculo_comboBox.addItem("")
        self.tipo_calculo_comboBox.setObjectName(u"tipo_calculo_comboBox")

        self.verticalLayout_4.addWidget(self.tipo_calculo_comboBox)

        self.cmap_label = QLabel(self.layoutWidget_2)
        self.cmap_label.setObjectName(u"cmap_label")

        self.verticalLayout_4.addWidget(self.cmap_label)

        self.cmap_comboBox = QComboBox(self.layoutWidget_2)
        self.cmap_comboBox.addItem("")
        self.cmap_comboBox.addItem("")
        self.cmap_comboBox.addItem("")
        self.cmap_comboBox.addItem("")
        self.cmap_comboBox.addItem("")
        self.cmap_comboBox.addItem("")
        self.cmap_comboBox.addItem("")
        self.cmap_comboBox.setObjectName(u"cmap_comboBox")

        self.verticalLayout_4.addWidget(self.cmap_comboBox)

        self.real_julia_label = QLabel(self.layoutWidget_2)
        self.real_julia_label.setObjectName(u"real_julia_label")

        self.verticalLayout_4.addWidget(self.real_julia_label)

        self.real_julia_entrada = QLineEdit(self.layoutWidget_2)
        self.real_julia_entrada.setObjectName(u"real_julia_entrada")

        self.verticalLayout_4.addWidget(self.real_julia_entrada)

        self.im_julia_label = QLabel(self.layoutWidget_2)
        self.im_julia_label.setObjectName(u"im_julia_label")

        self.verticalLayout_4.addWidget(self.im_julia_label)

        self.im_julia_entrada = QLineEdit(self.layoutWidget_2)
        self.im_julia_entrada.setObjectName(u"im_julia_entrada")

        self.verticalLayout_4.addWidget(self.im_julia_entrada)

        self.layoutWidget1 = QWidget(self.centralwidget)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.layoutWidget1.setGeometry(QRect(830, 10, 141, 161))
        self.verticalLayout_3 = QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.hacer_fractal_label = QLabel(self.layoutWidget1)
        self.hacer_fractal_label.setObjectName(u"hacer_fractal_label")

        self.verticalLayout_3.addWidget(self.hacer_fractal_label)

        self.boton_hacer_fractal = QPushButton(self.layoutWidget1)
        self.boton_hacer_fractal.setObjectName(u"boton_hacer_fractal")

        self.verticalLayout_3.addWidget(self.boton_hacer_fractal)

        self.guardar_label = QLabel(self.layoutWidget1)
        self.guardar_label.setObjectName(u"guardar_label")

        self.verticalLayout_3.addWidget(self.guardar_label)

        self.boton_guardar = QPushButton(self.layoutWidget1)
        self.boton_guardar.setObjectName(u"boton_guardar")

        self.verticalLayout_3.addWidget(self.boton_guardar)

        self.resetear_label = QLabel(self.layoutWidget1)
        self.resetear_label.setObjectName(u"resetear_label")

        self.verticalLayout_3.addWidget(self.resetear_label)

        self.boton_resetear = QPushButton(self.layoutWidget1)
        self.boton_resetear.setObjectName(u"boton_resetear")

        self.verticalLayout_3.addWidget(self.boton_resetear)

        self.grafico_openGLWidget = QOpenGLWidget(self.centralwidget)
        self.grafico_openGLWidget.setObjectName(u"grafico_openGLWidget")
        self.grafico_openGLWidget.setGeometry(QRect(10, 10, 800, 600))
        self.label_coordenadas = QLabel(self.centralwidget)
        self.label_coordenadas.setObjectName(u"label_coordenadas")
        self.label_coordenadas.setGeometry(QRect(20, 20, 441, 16))
        Boundary.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(Boundary)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1300, 22))
        Boundary.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Boundary)
        self.statusbar.setObjectName(u"statusbar")
        Boundary.setStatusBar(self.statusbar)

        self.retranslateUi(Boundary)

        QMetaObject.connectSlotsByName(Boundary)
    # setupUi

    def retranslateUi(self, Boundary):
        Boundary.setWindowTitle(QCoreApplication.translate("Boundary", u"Boundary", None))
        self.xmin_label.setText(QCoreApplication.translate("Boundary", u"xmin", None))
        self.xmin_entrada.setText(QCoreApplication.translate("Boundary", u"-2", None))
        self.xmax_label.setText(QCoreApplication.translate("Boundary", u"xmax", None))
        self.xmax_entrada.setText(QCoreApplication.translate("Boundary", u"1.2", None))
        self.ymin_label.setText(QCoreApplication.translate("Boundary", u"ymin", None))
        self.ymin_entrada.setText(QCoreApplication.translate("Boundary", u"-0.9", None))
        self.ymax_label.setText(QCoreApplication.translate("Boundary", u"ymax", None))
        self.ymax_entrada.setText(QCoreApplication.translate("Boundary", u"0.9", None))
        self.width_label.setText(QCoreApplication.translate("Boundary", u"Ancho", None))
        self.width_entrada.setText(QCoreApplication.translate("Boundary", u"1000", None))
        self.high_label.setText(QCoreApplication.translate("Boundary", u"Alto", None))
        self.high_entrada.setText(QCoreApplication.translate("Boundary", u"600", None))
        self.max_iter_label.setText(QCoreApplication.translate("Boundary", u"Iteraciones", None))
        self.max_iter_entrada.setText(QCoreApplication.translate("Boundary", u"256", None))
        self.boton_duplicar.setText(QCoreApplication.translate("Boundary", u"x2", None))
        self.boton_dividir.setText(QCoreApplication.translate("Boundary", u"/2", None))
        self.formula_label.setText(QCoreApplication.translate("Boundary", u"Formula", None))
        self.formula_entrada.setText(QCoreApplication.translate("Boundary", u"z = z**2 + C", None))
        self.zoom_out_factor_label.setText(QCoreApplication.translate("Boundary", u"Zoom out factor", None))
        self.zoom_out_factor_entrada.setText(QCoreApplication.translate("Boundary", u"2.0", None))
        self.zoom_in_factor_label.setText(QCoreApplication.translate("Boundary", u"Zoom in factor", None))
        self.zoom_in_factor_entrada.setText(QCoreApplication.translate("Boundary", u"0.5", None))
        self.exponente_label.setText(QCoreApplication.translate("Boundary", u"Exponente", None))
        self.generador_label.setText(QCoreApplication.translate("Boundary", u"Tipo de generador", None))
        self.generador_comboBox.setItemText(0, QCoreApplication.translate("Boundary", u"Sucesion", None))
        self.generador_comboBox.setItemText(1, QCoreApplication.translate("Boundary", u"Lsystem", None))

        self.tipo_fractal_label.setText(QCoreApplication.translate("Boundary", u"Tipo de fractal", None))
        self.tipo_fractal_comboBox.setItemText(0, QCoreApplication.translate("Boundary", u"Mandelbrot", None))
        self.tipo_fractal_comboBox.setItemText(1, QCoreApplication.translate("Boundary", u"Julia", None))
        self.tipo_fractal_comboBox.setItemText(2, QCoreApplication.translate("Boundary", u"Burning Ship", None))
        self.tipo_fractal_comboBox.setItemText(3, QCoreApplication.translate("Boundary", u"Tricorn", None))
        self.tipo_fractal_comboBox.setItemText(4, QCoreApplication.translate("Boundary", u"Circulo", None))

        self.tipo_calculo_label.setText(QCoreApplication.translate("Boundary", u"Tipo de Calculo", None))
        self.tipo_calculo_comboBox.setItemText(0, QCoreApplication.translate("Boundary", u"GPU_Cupy_kernel", None))
        self.tipo_calculo_comboBox.setItemText(1, QCoreApplication.translate("Boundary", u"CPU_Numpy", None))
        self.tipo_calculo_comboBox.setItemText(2, QCoreApplication.translate("Boundary", u"GPU_Cupy", None))

        self.cmap_label.setText(QCoreApplication.translate("Boundary", u"Elegir cmap", None))
        self.cmap_comboBox.setItemText(0, QCoreApplication.translate("Boundary", u"twilight_shifted", None))
        self.cmap_comboBox.setItemText(1, QCoreApplication.translate("Boundary", u"twilight", None))
        self.cmap_comboBox.setItemText(2, QCoreApplication.translate("Boundary", u"inferno", None))
        self.cmap_comboBox.setItemText(3, QCoreApplication.translate("Boundary", u"viridis", None))
        self.cmap_comboBox.setItemText(4, QCoreApplication.translate("Boundary", u"plasma", None))
        self.cmap_comboBox.setItemText(5, QCoreApplication.translate("Boundary", u"magma", None))
        self.cmap_comboBox.setItemText(6, QCoreApplication.translate("Boundary", u"cividis", None))

        self.real_julia_label.setText(QCoreApplication.translate("Boundary", u"Parte real de julia", None))
        self.real_julia_entrada.setText(QCoreApplication.translate("Boundary", u"0", None))
        self.im_julia_label.setText(QCoreApplication.translate("Boundary", u"Parte im de julia", None))
        self.im_julia_entrada.setText(QCoreApplication.translate("Boundary", u"0", None))
        self.hacer_fractal_label.setText(QCoreApplication.translate("Boundary", u"Generar Fractal", None))
        self.boton_hacer_fractal.setText(QCoreApplication.translate("Boundary", u"Generar", None))
        self.guardar_label.setText(QCoreApplication.translate("Boundary", u"Guardar mandelbrot", None))
        self.boton_guardar.setText(QCoreApplication.translate("Boundary", u"Guardar", None))
        self.resetear_label.setText(QCoreApplication.translate("Boundary", u"Resetear", None))
        self.boton_resetear.setText(QCoreApplication.translate("Boundary", u"Reset", None))
        self.label_coordenadas.setText("")
    # retranslateUi

