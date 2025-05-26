#include <immintrin.h>  // intrínsecos AVX

void mandelbrot_simd_double(const double *cx, const double *cy, int *iters,
                            int N, int max_iter) {
    const __m256d two  = _mm256_set1_pd(2.0);
    const __m256d four = _mm256_set1_pd(4.0);
    for (int i = 0; i < N; i += 4) {                // 4 doubles en 256 bits
        __m256d cr = _mm256_loadu_pd(cx + i);
        __m256d ci = _mm256_loadu_pd(cy + i);
        __m256d zr = _mm256_setzero_pd();
        __m256d zi = _mm256_setzero_pd();
        __m128i count = _mm_setzero_si128();      // 4×32-bit ints caben en 128 bits

        for (int k = 0; k < max_iter; ++k) {
            __m256d zr2 = _mm256_mul_pd(zr, zr);
            __m256d zi2 = _mm256_mul_pd(zi, zi);
            __m256d mag2 = _mm256_add_pd(zr2, zi2);
            __m256d mask = _mm256_cmp_pd(mag2, four, _CMP_LE_OQ);
            if (_mm256_movemask_pd(mask) == 0) break;

            __m256d zrzi   = _mm256_mul_pd(zr, zi);
            __m256d zr_nxt = _mm256_add_pd(_mm256_sub_pd(zr2, zi2), cr);
            __m256d zi_nxt = _mm256_add_pd(_mm256_add_pd(zrzi, zrzi), ci);
            zr = _mm256_blendv_pd(zr, zr_nxt, mask);
            zi = _mm256_blendv_pd(zi, zi_nxt, mask);

            // Para el contador hay que usar 64→32: convertimos mask a entero
            __m128i m4 = _mm256_castsi256_si128(_mm256_castpd_si256(mask));
            __m128i one = _mm_set1_epi32(1);
            count = _mm_add_epi32(count, _mm_and_si128(m4, one));
        }

        // Guardamos 4 contadores
        _mm_storeu_si128((__m128i*)(iters + i), count);
    }
}
