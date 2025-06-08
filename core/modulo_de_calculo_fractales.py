"""Facade module aggregating fractal calculators."""
from .fractales.base import (
    FractalCalculator,
    register_fractal,
    FRACTAL_REGISTRY,
)

# Import fractal implementations so they register themselves
from .fractales import (
    mandelbrot,
    julia,
    burning_ship,
    tricorn,
    circulo,
    newton,
    phoenix,
    burning_julia,
    celtic_mandelbrot,
    nova,
    gamma_fractals,
)

# Backwards compatibility
calculos_mandelbrot = FractalCalculator
