#include <vector>
#include <windows.h>
#include <cstdlib>
#include <omp.h>  // <<— incluir OpenMP

extern "C" {

__declspec(dllexport)
int* julia(double xmin, double xmax,
           double ymin, double ymax,
           int width, int height,
           int max_iter,
           double cr, double ci)
{
    int *M = new(std::nothrow) int[width * height];
    if (!M) return nullptr;

    double dx = (xmax - xmin) / (width  - 1);
    double dy = (ymax - ymin) / (height - 1);

    // paralelizar ambos bucles
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < height; ++j) {
        double y0 = ymin + j * dy;
        for (int i = 0; i < width; ++i) {
            double x0 = xmin + i * dx;
            double zr = x0;
            double zi = y0;
            int n = 0;

            while (n < max_iter && (zr*zr + zi*zi) <= 4.0) {
                double zr_new = zr*zr - zi*zi + cr;
                zi = 2.0*zr*zi + ci;
                zr = zr_new;
                ++n;
            }

            M[j * width + i] = n;
        }
    }

    return M;
}

__declspec(dllexport)
void free_julia(int* M) {
    delete[] M;
}

} // extern "C"
