// coseno_inv.cpp
#include <complex>
#include <windows.h>
#include <cstdlib>
#include <omp.h>

extern "C" {

    __declspec(dllexport)
    int* coseno_inv(double xmin, double xmax,
                    double ymin, double ymax,
                    int width, int height,
                    int max_iter)
    {
        int* M = new int[width * height];
        if (!M) return nullptr;

        double dx = (xmax - xmin) / (width  - 1);
        double dy = (ymax - ymin) / (height - 1);

        #pragma omp parallel for collapse(2)
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                double x = xmin + i * dx;
                double y = ymin + j * dy;
                
                std::complex<double> c(x, y);
                // Evitamos división por cero en C=0
                if (c == std::complex<double>(0.0, 0.0)) {
                    M[j * width + i] = max_iter - 1;
                    continue;
                }

                std::complex<double> z(0.0, 0.0);
                int n = 0;
                while (n < max_iter && std::norm(z) <= 256.0) {
                    // z_{n+1} = cos(z_n) + 1/C
                    z = std::cos(z) + std::complex<double>(1.0, 0.0) / c;
                    ++n;
                }
                // Guardamos la iteración en que escapa (o max_iter-1 si no)
                M[j * width + i] = n > 0 ? n - 1 : 0;
            }
        }

        return M;
    }

    __declspec(dllexport)
    void free_coseno_inv(int* M) {
        delete[] M;
    }

}
