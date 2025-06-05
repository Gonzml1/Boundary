import argparse
from core.modulo_de_calculo_fractales import calculos_mandelbrot


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fractal images")
    parser.add_argument(
        "--fractal",
        default="Mandelbrot",
        help="Tipo de fractal: Mandelbrot, Julia, Burning Ship, etc."
    )
    parser.add_argument(
        "--method",
        default="CPU_Numpy",
        help="Metodo de calculo: GPU_Cupy, GPU_Cupy_kernel, CPU_Numpy, etc."
    )
    parser.add_argument("--width"     , type=int    , default=1000)
    parser.add_argument("--height"    , type=int    , default=600)
    parser.add_argument("--max-iter"  , type=int    , default=256, dest="ma         x_iter")
    parser.add_argument("--xmin"      , type=float  , default=-2.0)
    parser.add_argument("--xmax"      , type=float  , default=1.2)
    parser.add_argument("--ymin"      , type=float  , default=-0.9)
    parser.add_argument("--ymax"      , type=float  , default=0.9)
    parser.add_argument(
        "--real",
        type=float,
        default=0.0,
        help="Parte real de la constante para conjuntos estilo Julia"
    )
    parser.add_argument(
        "--imag",
        type=float,
        default=0.0,
        help="Parte imaginaria de la constante para conjuntos estilo Julia"
    )
    parser.add_argument(
        "--formula",
        default="z = z**2 + C",
        help="Formula utilizada en la iteracion"
    )
    parser.add_argument("--output", default="fractal.png")
    parser.add_argument("--cmap", default="twilight_shifted")
    parser.add_argument("--dpi", type=int, default=100)

    args = parser.parse_args()

    calc = calculos_mandelbrot(
        args.xmin,
        args.xmax,
        args.ymin,
        args.ymax,
        args.width,
        args.height,
        args.max_iter,
        args.formula,
        args.method,
        args.fractal,
        args.real,
        args.imag,
    )

    matriz = calc.calcular_fractal()
    calc.guardar_mandelbrot(matriz, args.output, args.cmap, args.dpi)
    print(f"Imagen guardada en {args.output}")


if __name__ == "__main__":
    main()
