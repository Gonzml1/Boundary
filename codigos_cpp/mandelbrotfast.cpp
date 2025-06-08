#include <complex>
#include <vector>
#include <windows.h>
#include <cstdlib>
#include <omp.h>  // OpenMP

extern "C" {
    __declspec(dllexport) int* mandelbrotfast(
        double xmin, double xmax, double ymin, double ymax,
        int width, int height, int max_iter
    ) {
        int* __restrict M = new int[width * height];
        if (!M) return nullptr;

        const double dx = (xmax - xmin) / (width - 1);
        const double dy = (ymax - ymin) / (height - 1);

        // Paraleliza solo sobre filas para mejor localización de caché
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < height; ++j) {
            double y = ymin + j * dy;
            int row_base = j * width;
            for (int i = 0; i < width; ++i) {
                double x = xmin + i * dx;

                // Cardioide (main bulb)
                double q = (x - 0.25) * (x - 0.25) + y * y;
                if (q * (q + (x - 0.25)) < 0.25 * y * y) {
                    M[row_base + i] = max_iter - 1;
                    continue;
                }
                // Period-2 bulb
                if ((x + 1) * (x + 1) + y * y < 0.0625) {
                    M[row_base + i] = max_iter - 1;
                    continue;
                }

                double zr = 0.0, zi = 0.0;
                double zr2 = 0.0, zi2 = 0.0;
                int n = 0;
                while (n < max_iter && (zr2 + zi2) <= 4.0) {
                    zi = 2.0 * zr * zi + y;
                    zr = zr2 - zi2 + x;
                    zr2 = zr * zr;
                    zi2 = zi * zi;
                    ++n;
                }
                M[row_base + i] = n - 1;
            }
        }
        return M;
    }

    __declspec(dllexport) void free_mandelbrotfast(int* M) {
        delete[] M;
    }
}
