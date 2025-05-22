#include <complex>
#include <vector>
#include <windows.h>
#include <cstdlib>

extern "C" {
    __declspec(dllexport) int* tricorn(double xmin, double xmax, double ymin, double ymax, int width, int height, int max_iter) {
        int* M = new int[width * height];
        if (!M) return nullptr;

        double dx = (xmax - xmin) / (width - 1);
        double dy = (ymax - ymin) / (height - 1);

        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                double x = xmin + i * dx;
                double y = ymin + j * dy;

                double zr = 0.0, zi = 0.0;
                int n = 0;
                while (n < max_iter && zr * zr + zi * zi <= 4.0) {
                    // Tricorn: usar el conjugado de z^2
                    double zr_new = zr * zr - zi * zi + x;
                    zi = -2 * zr * zi + y; // Negativo para el conjugado
                    zr = zr_new;
                    ++n;
                }
                M[j * width + i] = n - 1;
            }
        }
        return M;
    }

    __declspec(dllexport) void free_tricorn(int* M) {
        delete[] M;
    }
}