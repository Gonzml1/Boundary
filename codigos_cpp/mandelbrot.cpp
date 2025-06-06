#include <complex>
#include <vector>
#include <windows.h>
#include <cstdlib>
#include <omp.h>  // <<-- Agregar

extern "C" {
    __declspec(dllexport) int* mandelbrot(double xmin, double xmax, double ymin, double ymax, int width, int height, int max_iter) {
        int* M = new int[width * height];
        if (!M) return nullptr;

        double dx = (xmax - xmin) / (width - 1);
        double dy = (ymax - ymin) / (height - 1);

        // Paralelizar ambos bucles con collapse(2)
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                double x = xmin + i * dx;
                double y = ymin + j * dy;

                double q = (x - 0.25) * (x - 0.25) + y * y;
                if (q * (q + (x - 0.25)) < 0.25 * y * y) {
                    M[j * width + i] = max_iter - 1;
                    continue;
                }

                double zr = 0.0, zi = 0.0;
                int n = 0;
                while (n < max_iter && zr * zr + zi * zi <= 4.0) {
                    double zr_new = zr * zr - zi * zi + x;
                    zi = 2 * zr * zi + y;
                    zr = zr_new;
                    ++n;
                }
                M[j * width + i] = n - 1;
            }
        }
        return M;
    }

    __declspec(dllexport) void free_mandelbrot(int* M) {
        delete[] M;
    }
}
