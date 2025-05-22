#include <complex>
#include <vector>
#include <windows.h>
#include <cstdlib>

extern "C" {
    __declspec(dllexport) int* newton(double xmin, double xmax, double ymin, double ymax, int width, int height, int max_iter) {
        int* M = new int[width * height];
        if (!M) return nullptr;

        double dx = (xmax - xmin) / (width - 1);
        double dy = (ymax - ymin) / (height - 1);

        // Define the three roots of z^3 - 1 = 0
        std::complex<double> raices[3] = {
            std::complex<double>(1.0, 0.0),
            std::complex<double>(-0.5, 0.8660254),
            std::complex<double>(-0.5, -0.8660254)
        };
        double tolerancia = 1e-6;

        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                double x = xmin + i * dx;
                double y = ymin + j * dy;
                std::complex<double> z(x, y);

                int n = 0;
                while (n < max_iter) {
                    // Compute f(z) = z^3 - 1 and f'(z) = 3z^2
                    std::complex<double> fz = z * z * z - 1.0;
                    std::complex<double> dfz = 3.0 * z * z;

                    // Avoid division by zero
                    if (std::abs(dfz) < 1e-10) {
                        M[j * width + i] = 0; // No convergence
                        break;
                    }

                    // Newton-Raphson iteration: z = z - f(z)/f'(z)
                    z = z - fz / dfz;

                    // Check convergence to any root
                    for (int k = 0; k < 3; ++k) {
                        if (std::abs(z - raices[k]) < tolerancia) {
                            M[j * width + i] = k + 1; // Root index (1, 2, or 3)
                            break;
                        }
                    }

                    // If we assigned a root, exit the iteration loop
                    if (M[j * width + i] != 0) {
                        break;
                    }

                    ++n;
                }

                // If no convergence after max_iter, set to 0
                if (n >= max_iter) {
                    M[j * width + i] = 0;
                }
            }
        }
        return M;
    }

    __declspec(dllexport) void free_newton(int* M) {
        delete[] M;
    }
}