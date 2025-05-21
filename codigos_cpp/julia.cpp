#include <vector>
#include <windows.h>
#include <cstdlib>

extern "C" {
    __declspec(dllexport) int* julia(double xmin, double xmax, double ymin, double ymax, int width, int height, int max_iter, double cr, double ci) {
        int* M = new int[width * height];
        if (!M) return nullptr;

        double dx = (xmax - xmin) / (width - 1);
        double dy = (ymax - ymin) / (height - 1);

        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                double zr = xmin + i * dx; // z0 = x + yi
                double zi = ymin + j * dy;

                // Verificación del cardioide
                double q = (zr - 0.25) * (zr - 0.25) + zi * zi;
                if (q * (q + (zr - 0.25)) < 0.25 * zi * zi) {
                    M[j * width + i] = max_iter - 1;
                    continue;
                }

                int n = 0;
                while (n < max_iter && zr * zr + zi * zi <= 4.0) {
                    double zr_new = zr * zr - zi * zi + cr;
                    zi = 2 * zr * zi + ci;
                    zr = zr_new;
                    ++n;
                }
                M[j * width + i] = n - 1;
            }
        }
        return M;
    }

    __declspec(dllexport) void free_julia(int* M) {
        delete[] M;
    }
}