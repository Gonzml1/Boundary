// mandelbrot_ap.cpp
//
// Para compilar con MSVC (Windows):
// CL /EHsc /O2 /openmp /I"C:\\boost_1_86_0" mandelbrot_ap.cpp /LD /Fe:mandelbrot_ap.dll
//
// Para compilar con MinGW:
// g++ -O3 -fopenmp -std=c++17 -I"C:/boost_1_86_0" -shared -o mandelbrot_ap.dll mandelbrot_ap.cpp
//
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <omp.h>
#include <vector>
#include <cstddef>

// Tipo de precisión dinámica en dígitos decimales
typedef boost::multiprecision::number<boost::multiprecision::cpp_dec_float<0>> mpfloat;

extern "C" {
    __declspec(dllexport)
    int* mandelbrot_precision(
        double xmin, double xmax,
        double ymin, double ymax,
        int width, int height, int max_iter,
        int precision // prec. en dígitos
    ) {
        // Ajuste de precisión en tiempo de ejecución
        mpfloat::default_precision(precision);

        mpfloat xmin_mp = xmin;
        mpfloat xmax_mp = xmax;
        mpfloat ymin_mp = ymin;
        mpfloat ymax_mp = ymax;

        mpfloat dx = (xmax_mp - xmin_mp) / static_cast<mpfloat>(width - 1);
        mpfloat dy = (ymax_mp - ymin_mp) / static_cast<mpfloat>(height - 1);

        // Precomputar valores de x
        std::vector<mpfloat> x_values(width);
        for (int i = 0; i < width; ++i) {
            x_values[i] = xmin_mp + i * dx;
        }

        auto total = static_cast<std::size_t>(width) * height;
        int* M = new int[total];
        if (!M) return nullptr;

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < height; ++j) {
            mpfloat y = ymin_mp + j * dy;
            for (int i = 0; i < width; ++i) {
                mpfloat zr = 0, zi = 0;
                mpfloat x0 = x_values[i];
                mpfloat y0 = y;
                int n = 0;
                while (n < max_iter && (zr * zr + zi * zi) <= mpfloat(4)) {
                    mpfloat zr2 = zr * zr;
                    mpfloat zi2 = zi * zi;
                    mpfloat tmp = zr2 - zi2 + x0;
                    zi = mpfloat(2) * zr * zi + y0;
                    zr = tmp;
                    ++n;
                }
                M[static_cast<std::size_t>(j) * width + i] = n;
            }
        }
        return M;
    }

    __declspec(dllexport)
    void free_mandelbrot_precision(int* M) {
        delete[] M;
    }
}
