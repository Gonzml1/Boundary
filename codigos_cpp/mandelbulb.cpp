#include <cmath>
#include <windows.h>
#include <omp.h>

extern "C" {

// Devuelve un puntero a una matriz 3D linealizada con los valores de iteración
__declspec(dllexport) int* mandelbulb(double xmin, double xmax,
                                      double ymin, double ymax,
                                      double zmin, double zmax,
                                      int width, int height, int depth,
                                      int max_iter, int power) {
    int* data = new int[width * height * depth];
    if (!data) return nullptr;

    double dx = (xmax - xmin) / (width - 1);
    double dy = (ymax - ymin) / (height - 1);
    double dz = (zmax - zmin) / (depth - 1);

    #pragma omp parallel for collapse(3)
    for (int k = 0; k < depth; ++k) {
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                double cx = xmin + i * dx;
                double cy = ymin + j * dy;
                double cz = zmin + k * dz;

                double x = 0.0, y = 0.0, z = 0.0;
                int n = 0;

                while (n < max_iter) {
                    double r = sqrt(x*x + y*y + z*z);
                    if (r > 2.0) break;

                    double theta = acos(z / r);
                    double phi = atan2(y, x);
                    double rn = pow(r, power);

                    double sin_theta = sin(theta * power);
                    double new_x = rn * sin_theta * cos(phi * power);
                    double new_y = rn * sin_theta * sin(phi * power);
                    double new_z = rn * cos(theta * power);

                    x = new_x + cx;
                    y = new_y + cy;
                    z = new_z + cz;

                    ++n;
                }

                data[k * width * height + j * width + i] = n;
            }
        }
    }

    return data;
}

// Función para liberar la memoria desde Python
__declspec(dllexport) void free_mandelbulb(int* data) {
    delete[] data;
}

}
