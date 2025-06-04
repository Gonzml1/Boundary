import cupy as cp

mandelbrot_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='int32 result',
    operation="""
        complex<double> z = 0.0;  
        for (int i = 0; i < max_iter; ++i) {
            z = z*z + c;  
            if (real(z)*real(z) + imag(z)*imag(z) > 4.0) {
                result = i;
                return;
            }
        }
        result = max_iter;  
    """,
    name='mandelbrot_kernel'
)

julia_kernel = cp.ElementwiseKernel(
    in_params='complex128 z, complex128 c, int32 max_iter',
    out_params='int32 result',
    operation="""
        complex<double> z_temp = z;  
        for (int i = 0; i < max_iter; ++i) {
            z_temp = z_temp * z_temp + c;  
            if (real(z_temp) * real(z_temp) + imag(z_temp) * imag(z_temp) > 4.0) {
                result = i;
                return;
            }
        }
        result = max_iter;  
    """,
    name='julia_kernel'
)

burning_kernel = cp.ElementwiseKernel(
    in_params='complex128 z, complex128 c, int32 max_iter',
    out_params='int32 result',
    operation="""
        complex<double> z_temp = z;
        for (int i = 0; i < max_iter; ++i) {
            double z_real = fabs(real(z_temp));
            double z_imag = fabs(imag(z_temp));
            z_temp = complex<double>(z_real * z_real - z_imag * z_imag + real(c),
                                     2.0 * z_real * z_imag + imag(c));
            if (real(z_temp) * real(z_temp) + imag(z_temp) * imag(z_temp) > 4.0) {
                result = i;
                return;
            }
        }
        result = max_iter;
    """,
    name='burning_kernel'
)

tricorn_kernel = cp.ElementwiseKernel(
    in_params='complex128 z, complex128 c, int32 max_iter',
    out_params='int32 result',
    operation="""
        complex<double> z_temp = z;
        for (int i = 0; i < max_iter; ++i) {
            z_temp = conj(z_temp) * conj(z_temp) + c;
            if (real(z_temp) * real(z_temp) + imag(z_temp) * imag(z_temp) > 4.0) {
                result = i;
                return;
            }
        }
        result = max_iter;
    """,
    name='tricorn_kernel'
)

circulo_kernel = cp.ElementwiseKernel(
    in_params='complex128 z, complex128 c, int32 max_iter',
    out_params='int32 result',
    operation="""
        complex<double> z_temp = z;
        for (int i = 0; i < max_iter; ++i) {
            // z = exp((z^2 - 1.00001*z) / c^4)
            complex<double> z2 = z_temp * z_temp;
            complex<double> numerator = z2 - 1.00001 * z_temp;
            complex<double> c4 = c * c * c * c;
            z_temp = exp(numerator / c4);
            if (real(z_temp) * real(z_temp) + imag(z_temp) * imag(z_temp) > 4.0) {
                result = i;
                return;
            }
        }
        result = max_iter;
    """,
    name='circulo_kernel'
)

newton_kernel = cp.ElementwiseKernel(
        in_params='complex128 c, int32 max_iter',
        out_params='int32 root_index, int32 iter_count',
        operation="""
            complex<double> z = c;
            complex<double> raices[3] = {
                complex<double>(1.0, 0.0),
                complex<double>(-0.5, 0.8660254),
                complex<double>(-0.5, -0.8660254)
            };
            double tolerancia = 1e-6;
            root_index = 0;
            iter_count = 0;

            for (int i = 0; i < max_iter; ++i) {
                complex<double> fz = z*z*z - 1.0;
                complex<double> dfz = 3.0*z*z;
                if (abs(dfz) < 1e-10) {  
                    root_index = 0;
                    iter_count = i;
                    return;
                }
                z = z - fz / dfz;

                for (int j = 0; j < 3; ++j) {
                    if (abs(z - raices[j]) < tolerancia) {
                        root_index = j + 1;
                        iter_count = i + 1;
                        return;
                    }
                }
            }
            root_index = 0;  
            iter_count = max_iter;
        """,
        name='newton_kernel'
    )

mandelbrot_smooth_kernel = cp.ElementwiseKernel(
    in_params='complex128 c, int32 max_iter',
    out_params='float64 mu',
    operation=r'''
        // z = 0 + 0i en doble precisión
        complex<double> z = complex<double>(0.0, 0.0);
        double abs_z = 0.0;
        int iter;

        for (iter = 0; iter < max_iter; ++iter) {
            // Iteramos z = z*z + c
            z = z * z + c;
            // abs_z = |z|^2
            abs_z = norm(z);  // Usamos norm para calcular |z|^2 eficientemente
            if (abs_z > 4.0) {
                break;  // Escapa si |z|^2 > 4
            }
        }

        if (iter == max_iter) {
            // Nunca escapó
            mu = static_cast<double>(max_iter);
        } else {
            // Escapó: calculamos valor suavizado
            double mod_z = sqrt(abs_z);  // |z|
            // Evitamos valores inválidos en el logaritmo
            if (mod_z > 1e-6) {  // Umbral para evitar log de valores pequeños
                mu = static_cast<double>(iter) + 1.0 - log(log(mod_z)) / log(2.0);
            } else {
                mu = static_cast<double>(iter);  // Valor por defecto si mod_z es muy pequeño
            }
        }
    ''',
    name='mandelbrot_smooth_kernel'
)

burning_julia_kernel = cp.ElementwiseKernel(
    in_params='complex128 z, complex128 c, bool mask',
    out_params='complex128 z_out, bool mask_out',
    operation='''
        if (mask) {
            thrust::complex<double> w = thrust::complex<double>(abs(z.real()), abs(z.imag()));
            z_out = w * w + c;
            mask_out = thrust::abs(z_out) <= 2.0;
        } else {
            z_out = z;
            mask_out = false;
        }
    ''',
    name='burning_julia'
)

phoenix_kernel = cp.ElementwiseKernel(
    in_params='complex128 z, complex128 zp, complex128 C, bool mask, complex128 p',
    out_params='complex128 z_out, complex128 zp_out, bool mask_out',
    operation='''
        if (mask) {
            z_out = z * z + p * zp + C;
            zp_out = z;
            mask_out = (thrust::abs(z_out) <= 2.0);
        } else {
            z_out = z;
            zp_out = zp;
            mask_out = false;
        }
    ''',
    name='phoenix'
)

celtic_mandelbrot_kernel = cp.ElementwiseKernel(
    in_params='complex128 z, complex128 c, bool mask',
    out_params='complex128 z_out, bool mask_out',
    operation='''
        if (mask) {
            thrust::complex<double> w = thrust::complex<double>(abs(z.real()), abs(z.imag()));
            thrust::complex<double> w_sqrt = thrust::sqrt(w);
            z_out = w_sqrt + c;
            mask_out = (thrust::abs(z_out) <= 2.0);
        } else {
            z_out = z;
            mask_out = false;
        }
    ''',
    name='celtic_mandelbrot'
)