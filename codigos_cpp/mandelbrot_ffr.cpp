// mandelbrot_ffr.cpp
// Ejemplo de integración de FFR con alta precisión, compresión adaptativa, y OpenMP SIMD

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <omp.h>
#include "double_exp.hpp"      // Motor de precisión arbitraria (dex_* API)
#include "approx_math.hpp"     // hypot_approx
#include "array_compressor.hpp"// ArrayCompressor

extern "C" {

    __declspec(dllexport)
    int* mandelbrot_ffr(
        double xmin, double xmax,
        double ymin, double ymax,
        int width, int height, int max_iter,
        int precision
    ) {
        using mpf = double_exp::float_t;

        // Ajuste de precisión en tiempo de ejecución
        mpf::set_precision(precision);

        // Incrementos en x e y
        mpf dx = (xmax - xmin) / mpf(width - 1);
        mpf dy = (ymax - ymin) / mpf(height - 1);

        // Precomputación de coordenadas
        std::vector<mpf> x_vals(width);
        std::vector<mpf> y_vals(height);
        for(int i = 0; i < width; ++i) x_vals[i] = xmin + i * dx;
        for(int j = 0; j < height; ++j) y_vals[j] = ymin + j * dy;

        // Configuración del compresor adaptativo
        ArrayCompressor compressor;
        std::vector<uint32_t> comp_vector(width * height);
        for(uint32_t k = 0; k < comp_vector.size(); ++k) comp_vector[k] = k;

        size_t total = size_t(width) * height;
        int* M = (int*) std::aligned_alloc(64, total * sizeof(int));
        if (!M) return nullptr;

        uint64_t start = 0, skip = 0;

        #pragma omp parallel for simd schedule(static)
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                uint64_t iter = start + skip++;
                uint64_t idx = compressor.compress(comp_vector, iter);

                // Carga de parámetros iniciales (C = anr + i·ani)
                mpf anr, ani;
                double_exp::dex_copy(&anr, x_vals[idx % width]);
                double_exp::dex_copy(&ani, y_vals[idx / width]);

                // Variables de iteración
                mpf zr, zi, radius;
                double_exp::dex_copy(&zr, anr);
                double_exp::dex_copy(&zi, ani);
                radius = mpf(1);

                mpf dcMax = mpf(4);
                int n = 0;
                while (n < max_iter) {
                    // Paso FFR en alta precisión
                    mpf zr2, zi2, temp;
                    double_exp::dex_mul(&zr2, zr, zr);
                    double_exp::dex_mul(&zi2, zi, zi);
                    double_exp::dex_sub(&temp, zr2, zi2);
                    double_exp::dex_add(&temp, temp, anr);

                    double_exp::dex_mul(&zi, zr, zi);
                    double_exp::dex_add(&zi, zi, zi);
                    double_exp::dex_add(&zi, zi, ani);

                    // Magnitud aproximada
                    radius = approx_math::hypot_approx(zr2, zi2);

                    // Actualizar zr
                    double_exp::dex_copy(&zr, temp);

                    // Normalización lazy
                    if (radius.normalize_required()) radius.normalize();
                    if (zr.normalize_required()) zr.normalize();
                    if (zi.normalize_required()) zi.normalize();

                    // Verificación de divergencia
                    if (radius > dcMax) break;
                    ++n;
                }

                M[j * width + i] = n;
            }
        }

        return M;
    }

    __declspec(dllexport)
    void free_mandelbrot_ffr(int* M) {
        std::free(M);
    }
}
