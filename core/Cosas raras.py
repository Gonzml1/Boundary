mpmath.mp.dps = 100

def calcular_orbita_referencia(c0_real, c0_imag, max_iter):
    """Calcula la órbita de referencia usando precisión arbitraria (mpmath)."""
    z = mpmath.mpc(0, 0)
    c = mpmath.mpc(c0_real, c0_imag)
    orbita_z_real = []
    orbita_z_imag = []
    
    for _ in range(max_iter):
        z = z**2 + c
        orbita_z_real.append(float(z.real))  # Convertimos a float64 para la GPU
        orbita_z_imag.append(float(z.imag))
        
    return cp.array(orbita_z_real), cp.array(orbita_z_imag)


perturbation_kernel = cp.ElementwiseKernel(
    in_params='''
        float64 cr, float64 ci,          
        float64 ref_cr, float64 ref_ci,   
        device float64* z_ref_real,       
        device float64* z_ref_imag,       
        int32 max_iter
    ''',
    out_params='int32 result',
    operation='''
        double delta_zr = 0.0;
        double delta_zi = 0.0;
        double delta_cr = cr - ref_cr;
        double delta_ci = ci - ref_ci;
        
        for (int i = 0; i < max_iter; ++i) {
            double zr_ref = z_ref_real[i];
            double zi_ref = z_ref_imag[i];
            
            
            double new_delta_zr = 2 * (zr_ref * delta_zr - zi_ref * delta_zi) 
                                + delta_zr * delta_zr - delta_zi * delta_zi 
                                + delta_cr;
                                
            double new_delta_zi = 2 * (zr_ref * delta_zi + zi_ref * delta_zr) 
                                + 2 * delta_zr * delta_zi 
                                + delta_ci;
                                
            
            delta_zr = new_delta_zr;
            delta_zi = new_delta_zi;
            
            
            double zr_total = zr_ref + delta_zr;
            double zi_total = zi_ref + delta_zi;
            
            
            if (zr_total*zr_total + zi_total*zi_total > 4.0) {
                result = i;
                return;
            }
        }
        result = max_iter;
    ''',
    name='perturbation_kernel'
)

def hacer_perturbacion(xmin, xmax, ymin, ymax, width, height, max_iter, formula, tipo_calculo, tipo_fractal, real, imag):
    inicio = time.time()
    
    # 1. Generar malla de puntos
    x = cp.linspace(xmin, xmax, width, dtype=cp.float64)
    y = cp.linspace(ymin, ymax, height, dtype=cp.float64)
    X, Y = cp.meshgrid(x, y)
    C_real = X.ravel()
    C_imag = Y.ravel()
    
    # 2. Seleccionar punto de referencia (centro del área visualizada)
    ref_cr = (xmin + xmax) / 2
    ref_ci = (ymin + ymax) / 2
    
    # 3. Calcular órbita de referencia con alta precisión
    z_ref_real, z_ref_imag = calcular_orbita_referencia(ref_cr, ref_ci, max_iter)
    
    # 4. Preparar parámetros para el kernel
    params = {
        'cr': C_real,
        'ci': C_imag,
        'ref_cr': ref_cr,
        'ref_ci': ref_ci,
        'z_ref_real': z_ref_real,
        'z_ref_imag': z_ref_imag,
        'max_iter': max_iter
    }
    
    # 5. Ejecutar kernel de perturbación
    resultado = cp.empty(C_real.shape, dtype=cp.int32)
    perturbation_kernel(**params, result=resultado)
    
    # 6. Dar formato al resultado
    resultado = resultado.reshape((height, width))
    resultado_cpu = resultado.get()
    
    tiempo = time.time() - inicio
    print(f"Tiempo total con perturbación: {tiempo:.2f} segundos")
    
    return resultado_cpu

# Ejemplo de uso:
'''resultado = perturbacion(
    xmin=-1.5,
    xmax=-1.4,
    ymin=-0.1,
    ymax=0.1,
    width=1000,
    height=1000,
    max_iter=5000,
    formula="mandelbrot"
)'''