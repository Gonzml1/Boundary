import numpy as np

def register_palette(name):
    """
    Decorator to register a color palette function.
    """
    def decorator(func):
        func._palette_name = name
        return func
    return decorator

@register_palette("Rojo→Amarillo→Blanco")
def _paleta_rojo_amarillo(self, norm: np.ndarray) -> np.ndarray:
    """
    Rojo→Amarillo→Blanco:
    - R siempre 255
    - G crece linealmente de 0→255 para norm en [0,0.5], luego se mantiene 255
    - B se activa solo para norm≥0.5, crece de 0→255 en [0.5,1]
    """
    r = np.uint8(255 * np.ones_like(norm))
    g = np.uint8((np.clip(norm * 2, 0, 1) * 255).clip(0, 255))
    b = np.uint8((np.clip((norm - 0.5) * 2, 0, 1) * 255).clip(0, 255))
    return np.dstack([r, g, b])

@register_palette("HSV")
def _paleta_hsv(self, norm: np.ndarray) -> np.ndarray:
    """
    Usa el colormap 'hsv' de Matplotlib:
    - Crea una LUT de 256 colores HSV→RGB y para cada valor de norm indexa a la LUT.
    """
    # 1) Obtenemos la LUT (solo la generamos una vez si quieres optimizar)
    cmap = cm.get_cmap('hsv', 256)                     # Colormap de 256 entradas
    lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)  # (256,3)
    # 2) Mapeamos norm ∈ [0,1] a 0..255
    indices = np.uint8((norm * 255).clip(0, 255))      # shape=(H,W), valores 0..255
    # 3) Indexamos
    return lut[indices]                                # shape=(H,W,3), dtype=uint8

@register_palette("Púrpura Psicodélica")
def _paleta_psicodelica(self, norm: np.ndarray) -> np.ndarray:
    """
    Púrpura psicodélica usando funciones sinusoidales:
    - Tres ciclos de color, fases desplazadas en R,G,B
    """
    # norm ∈ [0,1]
    r = np.uint8((0.5 + 0.5 * np.sin(2 * np.pi * norm * 3 + 0)) * 255)
    g = np.uint8((0.5 + 0.5 * np.sin(2 * np.pi * norm * 3 + 2)) * 255)
    b = np.uint8((0.5 + 0.5 * np.sin(2 * np.pi * norm * 3 + 4)) * 255)
    return np.dstack([r, g, b])
    
@register_palette("Accent")
def _paleta_accent(self, norm: np.ndarray) -> np.ndarray:
    """
    Colormap 'Accent' de Matplotlib (colores brillantes).
    """
    cmap = cm.get_cmap('Accent', 256)
    lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    indices = np.uint8((norm * 255).clip(0, 255))
    return lut[indices]

@register_palette("Dark2")
def _paleta_dark2(self, norm: np.ndarray) -> np.ndarray:
    """
    Colormap 'Dark2' de Matplotlib (colores oscuros y saturados).
    """
    cmap = cm.get_cmap('Dark2', 256)
    lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    indices = np.uint8((norm * 255).clip(0, 255))
    return lut[indices]

@register_palette("Set1")
def _paleta_set1(self, norm: np.ndarray) -> np.ndarray:
    """
    Colormap 'Set1' de Matplotlib (colores brillantes y saturados).
    """
    cmap = cm.get_cmap('Set1', 256)
    lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    indices = np.uint8((norm * 255).clip(0, 255))
    return lut[indices]

@register_palette("Set2")
def _paleta_set2(self, norm: np.ndarray) -> np.ndarray:
    """
    Colormap 'Set2' de Matplotlib (colores suaves y agradables).
    """
    cmap = cm.get_cmap('Set2', 256)
    lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    indices = np.uint8((norm * 255).clip(0, 255))
    return lut[indices]

@register_palette("Set3")
def _paleta_set3(self, norm: np.ndarray) -> np.ndarray:
    """
    Colormap 'Set3' de Matplotlib (colores variados y agradables).
    """
    cmap = cm.get_cmap('Set3', 256)
    lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    indices = np.uint8((norm * 255).clip(0, 255))
    return lut[indices]

@register_palette("prism")
def _paleta_prism(self, norm: np.ndarray) -> np.ndarray:
    """
    Colormap 'prism' de Matplotlib (colores suaves y claros).
    """
    cmap = cm.get_cmap('prism', 256)
    lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    indices = np.uint8((norm * 255).clip(0, 255))
    return lut[indices]

@register_palette("Prism LUT")
def _palette_prism_from_norm(self, norm: np.ndarray) -> np.ndarray:
    """
    Devuelve un array H×W×3 de uint8 con la paleta 'prism',
    usando un LUT de 256 colores, a partir de norm (valores en [0,1]).
    """
    # 1) Crear la lookup table de 256 colores
    lut = (cm.get_cmap('prism', 256)(np.arange(256))[:, :3] * 255).astype(np.uint8)
    # 2) Mapear norm [0,1] a índices 0–255
    indices = np.uint8((norm * 255).clip(0, 255))
    # 3) Devolver RGB
    return lut[indices]

@register_palette("Paired")
def _paleta_paired(self, norm: np.ndarray) -> np.ndarray:
    """
    Colormap 'Paired' de Matplotlib (cualitativo, pares de colores contrastantes).
    """
    cmap = cm.get_cmap('Paired', 256)
    lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    indices = np.uint8((norm * 255).clip(0, 255))
    return lut[indices]

@register_palette("Pastel1")
def _paleta_pastel1(self, norm: np.ndarray) -> np.ndarray:
    """
    Colormap 'Pastel1' de Matplotlib (cualitativo, colores suaves).
    """
    cmap = cm.get_cmap('Pastel1', 256)
    lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    indices = np.uint8((norm * 255).clip(0, 255))
    return lut[indices]

@register_palette("Iteraciones (HSV ciclo 64)")
def _paleta_iters_hsv(self, norm: np.ndarray) -> np.ndarray:
    """
    Paleta basada en el número de iteraciones:
    - Convertimos norm [0,1] de vuelta a iter (0..max_iter)
    - Tomamos iter % 64 para indexar 64 colores HSV
    """
    # Reconstruir el entero de iteración
    iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
    cycle = 64
    # Generar LUT HSV de 64 colores
    hsv_lut = cm.get_cmap('hsv', cycle)(np.arange(cycle))[:, :3]
    lut = (hsv_lut * 255).astype(np.uint8)          # (64,3) uint8
    # Indexar módulo ciclo
    return lut[iters % cycle]  


@register_palette("Grises cíclico")
def _paleta_grises_ciclico(self, norm: np.ndarray) -> np.ndarray:
    """
    Grises cíclico basado en iteraciones:
    - Reconstruye iters ∈ [0..max_iter] desde norm
    - Toma iters % cycle para definir la intensidad de gris
    """
    # 1) Reconstruir el conteo de iteraciones
    iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
    # 2) Definir ciclo de, por ejemplo, 64 pasos
    cycle = 64
    mod = iters % cycle
    # 3) Mapear mod ∈ [0..cycle-1] a gris ∈ [0..255]
    gray = np.uint8(((mod.astype(float) / (cycle - 1)) * 255).clip(0, 255))
    # 4) Devolver imagen H×W×3
    return np.dstack([gray, gray, gray])


@register_palette("Iteraciones (Plasma ciclo 64)")
def _paleta_iters_plasma(self, norm: np.ndarray) -> np.ndarray:
    """
    Paleta basada en el número de iteraciones:
    - Reconstruye iter ∈ [0..max_iter] desde norm
    - Usa iter % 64 para indexar un LUT de Plasma de 64 colores
    """
    # 1) Reconstruir conteo de iteraciones
    iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
    cycle = 64
    # 2) Generar LUT de Plasma con 64 entradas
    cmap   = cm.get_cmap('plasma', cycle)
    lut    = (cmap(np.arange(cycle))[:, :3] * 255).astype(np.uint8)  # (64,3)
    # 3) Indexar según iter % cycle
    return lut[iters % cycle]

       
@register_palette("Iteraciones (Viridis ciclo 64)")
def _paleta_iters_viridis(self, norm: np.ndarray) -> np.ndarray:
    """
    Paleta basada en el número de iteraciones:
    - Reconstruye iter ∈ [0..max_iter] desde norm
    - Usa iter % 64 para indexar un LUT de Viridis de 64 colores
    """
    # 1) Reconstruir conteo de iteraciones
    iters = np.uint32((norm * self.max_iter).clip(0, self.max_iter))
    cycle = 64
    # 2) Generar LUT de Viridis con 64 entradas
    cmap   = cm.get_cmap('viridis', cycle)
    lut    = (cmap(np.arange(cycle))[:, :3] * 255).astype(np.uint8)  # (64,3)
    # 3) Indexar según iter % cycle
    return lut[iters % cycle]  # shape=(H, W, 3), dtype=uint8

#@register_palette("Grises")
def _paleta_grises(self, norm: np.ndarray) -> np.ndarray:
    """
    Escala de grises: norm en [0,1] → (gray,gray,gray) en [0..255]
    """
    gray = np.uint8((norm * 255).clip(0, 255))
    return np.dstack([gray, gray, gray])  # shape=(H,W,3)

@register_palette("Bandas RGB")
def _paleta_bandas_rgb(self, norm: np.ndarray) -> np.ndarray:
    """
    Bandas semilineales: divide norm en 3 franjas, con degradado lineal dentro de cada franja:
    - franja0: rojo crece de 0→1
    - franja1: verde crece de 0→1
    - franja2: azul crece de 0→1
    """
    pos = norm * 3  # pos ∈ [0,3)
    # En la primera porción (pos < 1): r=pos, g=0,b=0
    # Segunda (1 ≤ pos < 2): r=2-pos, g=pos-1, b=0
    # Tercera (2 ≤ pos < 3): r=0, g=3-pos, b=pos-2
    r = np.where(pos < 1, pos, np.where(pos < 2, 2 - pos, 0))
    g = np.where(pos < 1, 0, np.where(pos < 2, pos - 1, 3 - pos))
    b = np.where(pos < 2, 0, pos - 2)
    rgb = np.dstack([r.clip(0,1), g.clip(0,1), b.clip(0,1)])
    return np.uint8(rgb * 255)