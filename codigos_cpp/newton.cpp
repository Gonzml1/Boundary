#include <complex>
#include <windows.h>
#include <cstdlib>
#include <cmath>
#include <omp.h>  // incluir OpenMP

extern "C" {
    __declspec(dllexport)
    int* newton(double xmin, double xmax,
                double ymin, double ymax,
                int width, int height,
                int max_iter)
    {
        int* M = new(std::nothrow) int[width * height];
        if (!M) return nullptr;

        double dx = (xmax - xmin) / (width  - 1);
        double dy = (ymax - ymin) / (height - 1);

        std::complex<double> raices[3] = {
            {1.0, 0.0},
            {-0.5, 0.8660254},
            {-0.5, -0.8660254}
        };
        const double tolerancia = 1e-6;

        #pragma omp parallel for collapse(2)
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                double x = xmin + i * dx;
                double y = ymin + j * dy;
                std::complex<double> z(x, y);
                int root_idx = 0;
                int n = 0;

                while (n < max_iter) {
                    std::complex<double> fz  = z * z * z - 1.0;
                    std::complex<double> dfz = 3.0 * z * z;
                    if (std::abs(dfz) < 1e-10) {
                        root_idx = 0;
                        break;
                    }
                    z = z - fz / dfz;
                    for (int k = 0; k < 3; ++k) {
                        if (std::abs(z - raices[k]) < tolerancia) {
                            root_idx = k + 1;
                            break;
                        }
                    }
                    if (root_idx) break;
                    ++n;
                }
                M[j * width + i] = root_idx;
            }
        }
        return M;
    }

    __declspec(dllexport)
    void free_newton(int* M) {
        delete[] M;
    }
}
