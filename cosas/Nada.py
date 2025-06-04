FRACTAL_REGISTRY: dict[str, dict[str, callable]] = {}


def register_fractal(fractal: str, calc: str):
    def deco(fn):
        FRACTAL_REGISTRY.setdefault(fractal, {})[calc] = fn
        return fn
    return deco

@register_fractal("mandelbrot", "default")
def hola():
    return "Hola, mundo!"

print(FRACTAL_REGISTRY)

funcion= {    "mandelbrot": hola }
print(funcion)