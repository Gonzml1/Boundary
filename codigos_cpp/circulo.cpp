#include <vector>
#include <windows.h>
#include <cstdlib>
#include <cmath>   // para std::exp, std::cos, std::sin
#include <omp.h>   // <<< incluir OpenMP

extern "C" {
    __declspec(dllexport)
    int* circulo(double xmin, double xmax, double ymin, double ymax,
                 int width, int height, int max_iter)
    {
        int* M = new(std::nothrow) int[width * height];
        if (!M) return nullptr;

        double dx = (xmax - xmin) / (width  - 1);
        double dy = (ymax - ymin) / (height - 1);

        #pragma omp parallel for collapse(2)
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                double x = xmin + i * dx;
                double y = ymin + j * dy;
                double zr = 0.0, zi = 0.0;
                int n = 0;

                while (n < max_iter && zr*zr + zi*zi <= 4.0) {
                    double zr2 = zr*zr - zi*zi;
                    double zi2 = 2.0 * zr * zi;
                    double term_r = zr2 - 1.00001 * zr;
                    double term_i = zi2 - 1.00001 * zi;

                    double c2_r = x*x - y*y;
                    double c2_i = 2.0 * x * y;
                    double c4_r = c2_r*c2_r - c2_i*c2_i;
                    double c4_i = 2.0 * c2_r * c2_i;

                    double denom = c4_r*c4_r + c4_i*c4_i;
                    if (denom < 1e-10) {
                        n = max_iter;
                        break;
                    }

                    double div_r = (term_r * c4_r + term_i * c4_i) / denom;
                    double div_i = (term_i * c4_r - term_r * c4_i) / denom;
                    double exp_r = std::exp(div_r);

                    zr = exp_r * std::cos(div_i);
                    zi = exp_r * std::sin(div_i);
                    ++n;
                }
                M[j * width + i] = n;
            }
        }
        return M;
    }

    __declspec(dllexport)
    void free_circulo(int* M) {
        delete[] M;
    }
}
