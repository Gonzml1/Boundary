import argparse
import numpy as np
import matplotlib.pyplot as plt
from core.modulo_de_calculo_fractales import calculos_mandelbrot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fractal images without the graphical interface"
    )
    parser.add_argument("--fractal", default="Mandelbrot", help="Type of fractal")
    parser.add_argument("--method", default="CPU_Numpy", help="Calculation method")
    parser.add_argument("--width", type=int, default=800, help="Image width")
    parser.add_argument("--height", type=int, default=600, help="Image height")
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum iterations")
    parser.add_argument("--xmin", type=float, default=-2.0)
    parser.add_argument("--xmax", type=float, default=1.0)
    parser.add_argument("--ymin", type=float, default=-1.0)
    parser.add_argument("--ymax", type=float, default=1.0)
    parser.add_argument("--formula", default="z**2 + C")
    parser.add_argument("--real", type=float, default=0.0, help="Julia constant real part")
    parser.add_argument("--imag", type=float, default=0.0, help="Julia constant imaginary part")
    parser.add_argument("--output", required=True, help="Output image path")
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
        ui=None,
    )

    data = calc.calcular_fractal()
    norm = data.astype(float) / args.max_iter
    plt.imsave(args.output, norm, cmap="twilight_shifted")
    print(f"Image saved to {args.output}")


if __name__ == "__main__":
    main()
