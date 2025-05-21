#include <complex>
#include <vector>
#include <windows.h>
#include <cstdlib>
#include <cmath> // Para std::abs

extern "C" {
    __declspec(dllexport) int* burning_ship(double xmin, double xmax, double ymin, double ymax, int width, int height, int max_iter) {
        int* M = new int[width * height];
        if (!M) return nullptr;

        double dx = (xmax - xmin) / (width - 1);
        double dy = (ymax - ymin) / (height - 1);

        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                double x = xmin + i * dx;
                double y = ymin + j * dy;

                // Verificación del cardioide principal (puede ser menos efectiva en Burning Ship)
                double q = (x - 0.25) * (x - 0.25) + y * y;
                if (q * (q + (x - 0.25)) < 0.25 * y * y) {
                    M[j * width + i] = max_iter - 1;
                    continue;
                }

                double zr = 0.0, zi = 0.0;
                int n = 0;
                while (n < max_iter && zr * zr + zi * zi <= 4.0) {
                    // Burning Ship: tomar valores absolutos de zr y zi
                    double zr_abs = std::abs(zr);
                    double zi_abs = std::abs(zi);
                    double zr_new = zr_abs * zr_abs - zi_abs * zi_abs + x;
                    zi = 2 * zr_abs * zi_abs + y;
                    zr = zr_new;
                    ++n;
                }
                M[j * width + i] = n - 1;
            }
        }
        return M;
    }

    __declspec(dllexport) void free_burning_ship(int* M) {
        delete[] M;
    }
}