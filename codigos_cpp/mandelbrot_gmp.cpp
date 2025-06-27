#include <gmpxx.h>
#include <vector>
#include <cstring>

extern "C" {

struct FractalParams {
    int width;
    int height;
    int max_iter;
    int precision_bits;
    const char* xmin;
    const char* xmax;
    const char* ymin;
    const char* ymax;
};

// Exporta imagen como array plano de enteros (fila por fila)
__declspec(dllexport)
int* calcular_mandelbrot(const FractalParams* p) {
    mpf_set_default_prec(p->precision_bits);

    mpf_class x_min(p->xmin), x_max(p->xmax);
    mpf_class y_min(p->ymin), y_max(p->ymax);

    mpf_class dx = (x_max - x_min) / p->width;
    mpf_class dy = (y_max - y_min) / p->height;

    int* resultado = new int[p->width * p->height];

    for (int py = 0; py < p->height; ++py) {
        mpf_class cy = y_max - py * dy;
        for (int px = 0; px < p->width; ++px) {
            mpf_class cx = x_min + px * dx;
            mpf_class zx = 0, zy = 0, zx2, zy2;

            int iter = 0;
            while (iter < p->max_iter) {
                zx2 = zx * zx;
                zy2 = zy * zy;
                if (zx2 + zy2 > 4) break;

                mpf_class xtemp = zx2 - zy2 + cx;
                zy = 2 * zx * zy + cy;
                zx = xtemp;
                ++iter;
            }
            resultado[py * p->width + px] = iter;
        }
    }

    return resultado;
}

// Liberar memoria
__declspec(dllexport)
void liberar_memoria(int* ptr) {
    delete[] ptr;
}
}
