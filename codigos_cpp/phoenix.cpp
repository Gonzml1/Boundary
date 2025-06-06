#include <complex>
#include <windows.h>
#include <cstdlib>
#include <omp.h>   // incluir OpenMP

extern "C" {
__declspec(dllexport)
int* phoenix(double xmin, double xmax, double ymin, double ymax,
             int width, int height, int max_iter,
             double pr, double pi)
{
    int* M = new(std::nothrow) int[width * height];
    if (!M) return nullptr;

    double dx = (xmax - xmin) / (width  - 1);
    double dy = (ymax - ymin) / (height - 1);
    std::complex<double> p(pr, pi);

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            double x = xmin + i * dx;
            double y = ymin + j * dy;
            std::complex<double> c(x, y);
            std::complex<double> z(0.0, 0.0), zp(0.0, 0.0);
            int n = 0;
            while (n < max_iter && std::norm(z) <= 4.0) {
                std::complex<double> z_next = z * z + p * zp + c;
                zp = z;
                z = z_next;
                ++n;
            }
            M[j * width + i] = n;
        }
    }
    return M;
}

__declspec(dllexport)
void free_phoenix(int* M) {
    delete[] M;
}
} // extern "C"
