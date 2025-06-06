#include <complex>
#include <windows.h>
#include <cstdlib>
#include <cmath>
#include <omp.h>   // incluir OpenMP

extern "C" {
__declspec(dllexport)
int* nova(double xmin, double xmax, double ymin, double ymax,
          int width, int height, int max_iter,
          double m, double k)
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
            std::complex<double> c(x, y);
            std::complex<double> z = c;
            int n = 0;
            while (n < max_iter && std::norm(z) <= 4.0) {
                std::complex<double> z_safe = (z == std::complex<double>(0.0, 0.0))
                                              ? std::complex<double>(1e-16, 0.0)
                                              : z;
                std::complex<double> z_pow_m    = std::pow(z_safe, m);
                std::complex<double> z_pow_negm = std::pow(z_safe, -m);
                z = z_pow_m + c + k * z_pow_negm;
                ++n;
            }
            M[j * width + i] = n;
        }
    }
    return M;
}

__declspec(dllexport)
void free_nova(int* M) {
    delete[] M;
}
} // extern "C"
