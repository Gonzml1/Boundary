#include <boost/multiprecision/cpp_dec_float.hpp>
#include <omp.h>
#include <windows.h>

// Precisión fija en tiempo de compilación:
typedef boost::multiprecision::number<boost::multiprecision::cpp_dec_float<100>> mpfloat;

extern "C" {
    __declspec(dllexport)
    int* mandelbrot_ap(
        double xmin, double xmax,
        double ymin, double ymax,
        int width, int height, int max_iter,
        int precision // NO USADO, solo informativo
    ) {
        // mpfloat::default_precision(precision); // <-- Esta línea ELIMINADA

        mpfloat xmin_mp = mpfloat(xmin);
        mpfloat xmax_mp = mpfloat(xmax);
        mpfloat ymin_mp = mpfloat(ymin);
        mpfloat ymax_mp = mpfloat(ymax);

        mpfloat dx = (xmax_mp - xmin_mp) / (width - 1);
        mpfloat dy = (ymax_mp - ymin_mp) / (height - 1);

        int* M = new int[width * height];
        if (!M) return nullptr;

        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < height; ++j) {
            mpfloat y = ymin_mp + j * dy;
            for (int i = 0; i < width; ++i) {
                mpfloat x = xmin_mp + i * dx;
                mpfloat zr = 0, zi = 0;
                int n = 0;
                while (n < max_iter && (zr*zr + zi*zi) <= 4) {
                    mpfloat tmp = zr*zr - zi*zi + x;
                    zi = 2 * zr * zi + y;
                    zr = tmp;
                    ++n;
                }
                M[j * width + i] = n - 1;
            }
        }
        return M;
    }

    __declspec(dllexport)
    void free_mandelbrot_ap(int* M) {
        delete[] M;
    }
}
