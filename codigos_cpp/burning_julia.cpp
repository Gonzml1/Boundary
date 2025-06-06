#include <complex>
#include <windows.h>
#include <cstdlib>
#include <cmath>
#include <omp.h>  // incluir OpenMP

extern "C" {
__declspec(dllexport)
int* burning_julia(double xmin, double xmax, double ymin, double ymax,
                   int width, int height, int max_iter,
                   double cr, double ci)
{
    int* M = new(std::nothrow) int[width * height];
    if (!M) return nullptr;

    double dx = (xmax - xmin) / (width - 1);
    double dy = (ymax - ymin) / (height - 1);
    std::complex<double> c(cr, ci);

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double x = xmin + i * dx;
            double y = ymin + j * dy;
            std::complex<double> z(x, y);
            int n = 0;
            while (n < max_iter && std::norm(z) <= 4.0) {
                double zr_abs = std::abs(z.real());
                double zi_abs = std::abs(z.imag());
                double zr_new = zr_abs * zr_abs - zi_abs * zi_abs + c.real();
                double zi_new = 2.0 * zr_abs * zi_abs + c.imag();
                z = std::complex<double>(zr_new, zi_new);
                ++n;
            }
            M[j * width + i] = n;
        }
    }
    return M;
}

__declspec(dllexport)
void free_burning_julia(int* M) {
    delete[] M;
}
} // extern "C"
