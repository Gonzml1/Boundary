#include <complex>
#include <windows.h>
#include <cstdlib>
#include <cmath>
#include <omp.h>   // <<< incluir OpenMP

extern "C" {
__declspec(dllexport)
int* celtic_mandelbrot(double xmin, double xmax, double ymin, double ymax,
                       int width, int height, int max_iter)
{
    int* M = new(std::nothrow) int[width * height];
    if (!M) return nullptr;

    double dx = (xmax - xmin) / (width - 1);
    double dy = (ymax - ymin) / (height - 1);

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double x = xmin + i * dx;
            double y = ymin + j * dy;
            std::complex<double> c(x, y);
            std::complex<double> z(0.0, 0.0);
            int n = 0;
            while (n < max_iter && std::norm(z) <= 4.0) {
                double zr_abs = std::abs(z.real());
                double zi_abs = std::abs(z.imag());
                std::complex<double> z_abs(zr_abs, zi_abs);
                std::complex<double> z_sqrt = std::sqrt(z_abs);
                z = z_sqrt + c;
                ++n;
            }
            M[j * width + i] = n;
        }
    }
    return M;
}

__declspec(dllexport)
void free_celtic_mandelbrot(int* M) {
    delete[] M;
}
} // extern "C"
