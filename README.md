La idea de este proyecto es hacer un visualizador de conjuntos de mandelbrot.
Las librerias utilizadas son cupy, numpy, PyQT5

## Uso por linea de comandos

Se incluye `cli.py` para generar fractales sin la interfaz grafica. Un ejemplo basico:

```bash
python cli.py --fractal Mandelbrot --method CPU_Numpy --width 800 --height 600 --max-iter 200 --output mandelbrot.png
```

El script permite elegir el fractal, metodo de calculo y parametros como las constantes de Julia.
