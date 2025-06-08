"""Base classes and utility functions for fractal calculations."""
import time
from functools import wraps
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from OpenGL.GL import *
from gui.MandelbrotGUI import Ui_Boundary

FRACTAL_REGISTRY: dict[str, dict[str, callable]] = {}

def register_fractal(fractal: str, calc: str) -> callable:
    """Register a function in FRACTAL_REGISTRY."""
    def deco(fn: callable) -> callable:
        FRACTAL_REGISTRY.setdefault(fractal, {})[calc] = fn
        return fn
    return deco

class FractalCalculator:
    """Container of fractal parameters and utilities."""
    def __init__(self, xmin: float, xmax: float , ymin: float, ymax: float,
                 width: int, height: int, max_iter: int, formula: str,
                 tipo_calculo: str, tipo_fractal: str, real: float, imag: float,
                 ui=Ui_Boundary()) -> None:
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.formula = formula
        self.tipo_calculo = tipo_calculo
        self.tipo_fractal = tipo_fractal
        self.real = real
        self.imag = imag
        self.x_np = np.linspace(self.xmin, self.xmax, self.width, dtype=np.float64)
        self.y_np = np.linspace(self.ymin, self.ymax, self.height, dtype=np.float64)
        self.ui   = ui
        self.p  = 1.0
        self.nova_m = 1.0
        self.nova_k = 1.0
        self._llenar_combo_fractales()

    def _llenar_combo_fractales(self) -> None:
        self.ui.tipo_fractal_comboBox.clear()
        self.ui.tipo_calculo_comboBox.clear()
        for fractal in FRACTAL_REGISTRY:
            self.ui.tipo_fractal_comboBox.addItem(fractal)
        if FRACTAL_REGISTRY:
            primer_fractal = next(iter(FRACTAL_REGISTRY))
            for calc in FRACTAL_REGISTRY[primer_fractal]:
                self.ui.tipo_calculo_comboBox.addItem(calc)
        self.ui.tipo_fractal_comboBox.currentTextChanged.connect(
            self._on_fractal_cambiado
        )

    def _on_fractal_cambiado(self, nombre_fractal: str) -> None:
        self.ui.tipo_calculo_comboBox.clear()
        if nombre_fractal in FRACTAL_REGISTRY:
            for calc in FRACTAL_REGISTRY[nombre_fractal]:
                self.ui.tipo_calculo_comboBox.addItem(calc)

    @staticmethod
    def medir_tiempo(nombre) -> callable:
        def decorador(func) -> callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> any:
                inicio = time.time()
                resultado = func(*args, **kwargs)
                fin = time.time()
                print(f"⏱️ Tiempo de ejecución de '{nombre}': {fin - inicio:.5f} segundos")
                return resultado
            return wrapper
        return decorador

    def actualizar_fractal(self, xmin: float,  xmax: float,
            ymin: float,  ymax: float,
            width: int,   height: int,
            max_iter: int, formula: str,
            tipo_calculo: str, tipo_fractal: str,
            real: float, imag: float) -> None:
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.formula = formula
        self.tipo_calculo = tipo_calculo
        self.tipo_fractal = tipo_fractal
        self.real = real
        self.imag = imag

    def calcular_fractal(self) -> np.ndarray:
        if self.tipo_fractal in FRACTAL_REGISTRY:
            if self.tipo_calculo in FRACTAL_REGISTRY[self.tipo_fractal]:
                fn = FRACTAL_REGISTRY[self.tipo_fractal][self.tipo_calculo]
                return fn(self)
        raise ValueError(f"Tipo de cálculo '{self.tipo_calculo}' no soportado para el fractal '{self.tipo_fractal}'.")

    @staticmethod
    def convertir_formula_compleja(formula: str) -> tuple[str, str]:
        if formula.strip() == "z**2 + C":
            real_expr = "zr**2 - zi**2 + Cr"
            imag_expr = "2 * zr * zi + Ci"
            return real_expr, imag_expr
        elif formula.strip() == "z**2 + 0":
            return "zr**2 - zi**2", "2 * zr * zi"
        raise NotImplementedError(f"Fórmula no soportada todavía: {formula}")

    @staticmethod
    def transformar_expresion(expression: str, variables: list[str], mask_name: str = "matriz") -> str:
        for var in variables:
            expression = expression.replace(var, f"{var}[{mask_name}]")
        return expression

    def guardar_mandelbrot(self, M, filepath, cmap1, dpi) -> None:
        figsize = ((self.width) / dpi, self.height / dpi)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.axis('off')
        ax.imshow(M, extent=(self.xmin, self.xmax, self.ymin, self.ymax), cmap=cmap1)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0, format="png")
        plt.close()

