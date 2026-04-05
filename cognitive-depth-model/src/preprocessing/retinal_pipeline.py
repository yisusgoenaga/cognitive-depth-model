"""
============================================
Pipeline de Preprocesamiento Retiniano
Fases 1-6 del Modelo Cognitivo Artificial
============================================

Implementa la simulación computacional del procesamiento
visual desde el ingreso de luz hasta las cintillas ópticas.

Fases:
    1. Ingreso de luz a la pupila (normalización, histograma, suavizado)
    2. Refracción en córnea y cristalino (enfoque óptico)
    3. Proyección en la retina (retinotopía, conos/bastones, nasal/temporal)
    4. Transformación en impulsos nerviosos (transducción, inhibición lateral, ganglionares)
    5. Quiasma óptico (cruce de fibras nasales)
    6. Cintillas ópticas (agrupación por hemisferio)

Autor: Jesús Goenaga Peña
Tesis Doctoral - Universidad Autónoma de Manizales
"""

import numpy as np
import cv2
import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


# ============================================
# Estructuras de datos
# ============================================

@dataclass
class RetinalChannels:
    """Salida de la Fase 3: canales retinianos separados."""
    # Conos (color y detalle, alta resolución en fóvea)
    left_cones_nasal: np.ndarray = None
    left_cones_temporal: np.ndarray = None
    right_cones_nasal: np.ndarray = None
    right_cones_temporal: np.ndarray = None
    # Bastones (luminancia y movimiento)
    left_rods_nasal: np.ndarray = None
    left_rods_temporal: np.ndarray = None
    right_rods_nasal: np.ndarray = None
    right_rods_temporal: np.ndarray = None


@dataclass
class GanglionMaps:
    """Salida de la Fase 4: mapas ganglionares on/off-center."""
    cones_on: list = field(default_factory=list)
    cones_off: list = field(default_factory=list)
    rods_on: list = field(default_factory=list)
    rods_off: list = field(default_factory=list)


@dataclass
class HemisphereSignals:
    """Salida de la Fase 5: señales organizadas por hemisferio."""
    cones_on: list = field(default_factory=list)
    cones_off: list = field(default_factory=list)
    rods_on: list = field(default_factory=list)
    rods_off: list = field(default_factory=list)


@dataclass
class OpticTracts:
    """Salida de la Fase 6: cintillas ópticas."""
    left_hemisphere: HemisphereSignals = None
    right_hemisphere: HemisphereSignals = None


# ============================================
# FASE 1: Ingreso de luz a la pupila
# ============================================
# Ecuación: I_norm = (I - min(I)) / (max(I) - min(I))
# Operaciones: normalización [0,1], igualación de histograma,
#              filtro gaussiano para reducir ruido
# ============================================

def phase1_pupil_light_entry(
    image: np.ndarray,
    gaussian_kernel: int = 3,
    gaussian_sigma: float = 0.5
) -> np.ndarray:
    """
    Fase 1: Simula el ingreso de luz a la pupila.

    Ajusta el rango dinámico de la imagen para simular la
    adaptación lumínica del sistema visual humano.

    Args:
        image: Imagen BGR de entrada (uint8)
        gaussian_kernel: Tamaño del kernel gaussiano
        gaussian_sigma: Sigma del filtro gaussiano

    Returns:
        Imagen preprocesada normalizada en [0, 1] (float32)
    """
    # 1. Normalización de intensidad a [0, 1]
    img_float = image.astype(np.float32) / 255.0

    # 2. Igualación de histograma (en espacio YCrCb para preservar color)
    img_uint8 = (img_float * 255).astype(np.uint8)
    img_ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2YCrCb)
    img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])  # Solo canal Y
    img_eq = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
    img_eq = img_eq.astype(np.float32) / 255.0

    # 3. Filtro gaussiano para reducir ruido
    img_smooth = cv2.GaussianBlur(
        img_eq,
        (gaussian_kernel, gaussian_kernel),
        gaussian_sigma
    )

    return img_smooth


# ============================================
# FASE 2: Refracción en córnea y cristalino
# ============================================
# Córnea: suavizado gaussiano general
# Cristalino: filtro laplaciano para mejorar nitidez local
# ============================================

def phase2_cornea_lens_refraction(
    image: np.ndarray,
    cornea_kernel: int = 5,
    cornea_sigma: float = 1.0,
    laplacian_kernel: int = 3,
    sharpening_strength: float = 0.3
) -> np.ndarray:
    """
    Fase 2: Simula la refracción en córnea y cristalino.

    La córnea realiza el suavizado general (filtro gaussiano)
    y el cristalino ajusta el enfoque fino (filtro laplaciano).

    Args:
        image: Imagen de la Fase 1 (float32, [0,1])
        cornea_kernel: Kernel del suavizado de la córnea
        cornea_sigma: Sigma del suavizado de la córnea
        laplacian_kernel: Kernel del filtro laplaciano
        sharpening_strength: Intensidad del afinamiento

    Returns:
        Imagen con enfoque óptico simulado (float32)
    """
    # 1. Córnea: suavizado gaussiano general
    img_cornea = cv2.GaussianBlur(
        image,
        (cornea_kernel, cornea_kernel),
        cornea_sigma
    )

    # 2. Cristalino: mejora de enfoque con Laplaciano
    laplacian = cv2.Laplacian(img_cornea, cv2.CV_32F, ksize=laplacian_kernel)
    img_focused = img_cornea - sharpening_strength * laplacian

    # Clamp a [0, 1]
    img_focused = np.clip(img_focused, 0.0, 1.0)

    return img_focused


# ============================================
# FASE 3: Proyección en la retina
# ============================================
# Transformación retinotópica: fóvea alta res, periferia baja res
# Separación en conos (color/detalle) y bastones (luminancia/movimiento)
# División en regiones nasal y temporal
# ============================================

def _create_foveal_mask(height: int, width: int, fovea_ratio: float = 0.3) -> np.ndarray:
    """Crea una máscara que simula la densidad foveal."""
    center_y, center_x = height // 2, width // 2
    y_coords, x_coords = np.ogrid[:height, :width]
    max_radius = min(height, width) * fovea_ratio

    distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)

    # Máscara gaussiana: 1.0 en fóvea, decae hacia periferia
    mask = np.exp(-0.5 * (distances / max_radius) ** 2)
    return mask.astype(np.float32)


def phase3_retinal_projection(
    image: np.ndarray,
    fovea_ratio: float = 0.3,
    periphery_blur: int = 15
) -> dict:
    """
    Fase 3: Simula la proyección en la retina.

    Aplica transformación retinotópica, separa la información
    en canales de conos y bastones, y divide en regiones
    nasal y temporal.

    Args:
        image: Imagen de la Fase 2 (float32, [0,1])
        fovea_ratio: Radio de la fóvea como proporción de la imagen
        periphery_blur: Intensidad del desenfoque periférico

    Returns:
        Dict con canales 'cones' y 'rods', cada uno con
        subcanales 'nasal' y 'temporal'
    """
    h, w = image.shape[:2]
    mid_w = w // 2

    # 1. Transformación retinotópica
    foveal_mask = _create_foveal_mask(h, w, fovea_ratio)
    periphery = cv2.GaussianBlur(image, (periphery_blur, periphery_blur), 0)

    # Mezcla: fóvea nítida + periferia borrosa
    if len(image.shape) == 3:
        foveal_mask_3d = foveal_mask[:, :, np.newaxis]
        img_retina = image * foveal_mask_3d + periphery * (1 - foveal_mask_3d)
    else:
        img_retina = image * foveal_mask + periphery * (1 - foveal_mask)

    # 2. Separación en conos y bastones
    # Conos: preservan color y detalle (suavizado fino)
    cones = cv2.GaussianBlur(img_retina, (3, 3), 0.5)

    # Bastones: sensibles a luminancia y movimiento
    if len(img_retina.shape) == 3:
        gray = cv2.cvtColor((img_retina * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
    else:
        gray = img_retina
    rods = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    rods = np.abs(rods)  # Valores absolutos de cambios de intensidad
    if rods.max() > 0:
        rods = rods / rods.max()  # Normalizar a [0, 1]

    # 3. División nasal / temporal
    result = {
        'cones_nasal': cones[:, :mid_w],
        'cones_temporal': cones[:, mid_w:],
        'rods_nasal': rods[:, :mid_w],
        'rods_temporal': rods[:, mid_w:],
    }

    return result


# ============================================
# FASE 4: Transformación en impulsos nerviosos
# ============================================
# Transducción visual: función no lineal (logarítmica)
# Inhibición lateral: kernel center-surround
# Mapas ganglionares: on-center y off-center
# Ecuación transducción: R = log(1 + I)
# Ecuación inhibición: I_inh = I(x) - λ * Σ I(y) para y ∈ N(x)
# ============================================

def _transduction(signal: np.ndarray) -> np.ndarray:
    """Simula la transducción visual con respuesta logarítmica."""
    return np.log1p(signal.astype(np.float32))


def _lateral_inhibition(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Simula la inhibición lateral retiniana.
    Resalta bordes y contrastes mediante un kernel center-surround.
    """
    # Kernel de inhibición lateral (center-surround)
    kernel = -np.ones((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    kernel[center, center] = kernel_size * kernel_size - 1
    kernel = kernel / (kernel_size * kernel_size)

    if len(signal.shape) == 3:
        result = np.zeros_like(signal)
        for c in range(signal.shape[2]):
            result[:, :, c] = cv2.filter2D(signal[:, :, c], -1, kernel)
    else:
        result = cv2.filter2D(signal, -1, kernel)

    return result


def _ganglion_response(signal: np.ndarray, response_type: str = "on-center") -> np.ndarray:
    """
    Genera mapas de respuesta ganglionar.

    on-center: responde a estímulos brillantes en el centro del campo receptivo
    off-center: responde a estímulos oscuros en el centro
    """
    blurred = cv2.GaussianBlur(signal, (5, 5), 1.0)

    if response_type == "on-center":
        # Centro excitatorio, periferia inhibitoria
        return np.clip(signal - blurred, 0, None)
    else:  # off-center
        # Centro inhibitorio, periferia excitatoria
        return np.clip(blurred - signal, 0, None)


def phase4_nerve_impulse_transformation(
    retinal_channels: dict,
    inhibition_kernel: int = 5
) -> dict:
    """
    Fase 4: Transforma la información visual en impulsos nerviosos simulados.

    Aplica transducción visual, inhibición lateral y genera
    mapas ganglionares on-center y off-center.

    Args:
        retinal_channels: Salida de la Fase 3
        inhibition_kernel: Tamaño del kernel de inhibición lateral

    Returns:
        Dict con mapas ganglionares organizados por tipo
    """
    result = {}

    for channel_name in ['cones_nasal', 'cones_temporal', 'rods_nasal', 'rods_temporal']:
        signal = retinal_channels[channel_name]

        # 1. Transducción visual
        transduced = _transduction(signal)

        # 2. Inhibición lateral
        inhibited = _lateral_inhibition(transduced, inhibition_kernel)

        # 3. Mapas ganglionares
        result[f'{channel_name}_on'] = _ganglion_response(inhibited, "on-center")
        result[f'{channel_name}_off'] = _ganglion_response(inhibited, "off-center")

    return result


# ============================================
# FASE 5: Quiasma óptico
# ============================================
# Fibras nasales → hemisferio contralateral
# Fibras temporales → hemisferio ipsilateral
# Preserva distinción conos/bastones y on/off
# ============================================

def phase5_optic_chiasma(
    left_ganglion: dict,
    right_ganglion: dict
) -> dict:
    """
    Fase 5: Simula el quiasma óptico.

    Reorganiza las señales según la topología del cruce:
    - Fibras nasales se cruzan al hemisferio opuesto
    - Fibras temporales permanecen ipsilaterales

    Args:
        left_ganglion: Mapas ganglionares del ojo izquierdo (Fase 4)
        right_ganglion: Mapas ganglionares del ojo derecho (Fase 4)

    Returns:
        Dict con señales organizadas por hemisferio
    """
    hemispheres = {
        # Hemisferio izquierdo recibe:
        #   - Temporal del ojo izquierdo (ipsilateral)
        #   - Nasal del ojo derecho (contralateral)
        'left_hemisphere': {
            'cones_on': [left_ganglion['cones_temporal_on'],
                         right_ganglion['cones_nasal_on']],
            'cones_off': [left_ganglion['cones_temporal_off'],
                          right_ganglion['cones_nasal_off']],
            'rods_on': [left_ganglion['rods_temporal_on'],
                        right_ganglion['rods_nasal_on']],
            'rods_off': [left_ganglion['rods_temporal_off'],
                         right_ganglion['rods_nasal_off']],
        },
        # Hemisferio derecho recibe:
        #   - Nasal del ojo izquierdo (contralateral)
        #   - Temporal del ojo derecho (ipsilateral)
        'right_hemisphere': {
            'cones_on': [left_ganglion['cones_nasal_on'],
                         right_ganglion['cones_temporal_on']],
            'cones_off': [left_ganglion['cones_nasal_off'],
                          right_ganglion['cones_temporal_off']],
            'rods_on': [left_ganglion['rods_nasal_on'],
                        right_ganglion['rods_temporal_on']],
            'rods_off': [left_ganglion['rods_nasal_off'],
                         right_ganglion['rods_temporal_off']],
        },
    }

    return hemispheres


# ============================================
# FASE 6: Cintillas ópticas
# ============================================
# Agrupación funcional de señales por hemisferio
# Sin procesamiento adicional, preserva estructura
# ============================================

def phase6_optic_tracts(hemispheres: dict) -> dict:
    """
    Fase 6: Simula las cintillas ópticas.

    Agrupa las señales reorganizadas en el quiasma óptico
    en estructuras que representan las cintillas izquierda y derecha.

    Args:
        hemispheres: Salida de la Fase 5

    Returns:
        Dict con cintillas ópticas izquierda y derecha
    """
    optic_tracts = {
        'left_tract': {
            'cones': {
                'on': hemispheres['left_hemisphere']['cones_on'],
                'off': hemispheres['left_hemisphere']['cones_off'],
            },
            'rods': {
                'on': hemispheres['left_hemisphere']['rods_on'],
                'off': hemispheres['left_hemisphere']['rods_off'],
            },
        },
        'right_tract': {
            'cones': {
                'on': hemispheres['right_hemisphere']['cones_on'],
                'off': hemispheres['right_hemisphere']['cones_off'],
            },
            'rods': {
                'on': hemispheres['right_hemisphere']['rods_on'],
                'off': hemispheres['right_hemisphere']['rods_off'],
            },
        },
    }

    return optic_tracts


# ============================================
# PIPELINE COMPLETO: Fases 1-6
# ============================================

def run_retinal_pipeline(
    img_left: np.ndarray,
    img_right: np.ndarray,
    config: Optional[dict] = None
) -> dict:
    """
    Ejecuta el pipeline retiniano completo (Fases 1-6)
    sobre un par estereoscópico.

    Args:
        img_left: Imagen del ojo izquierdo (BGR, uint8)
        img_right: Imagen del ojo derecho (BGR, uint8)
        config: Diccionario opcional con parámetros de configuración

    Returns:
        Dict con resultados intermedios de cada fase y salida final
    """
    if config is None:
        config = {}

    results = {'config': config}

    # --- Fase 1: Ingreso de luz ---
    left_p1 = phase1_pupil_light_entry(
        img_left,
        gaussian_kernel=config.get('p1_gaussian_kernel', 3),
        gaussian_sigma=config.get('p1_gaussian_sigma', 0.5)
    )
    right_p1 = phase1_pupil_light_entry(
        img_right,
        gaussian_kernel=config.get('p1_gaussian_kernel', 3),
        gaussian_sigma=config.get('p1_gaussian_sigma', 0.5)
    )
    results['phase1'] = {'left': left_p1, 'right': right_p1}

    # --- Fase 2: Refracción ---
    left_p2 = phase2_cornea_lens_refraction(
        left_p1,
        cornea_kernel=config.get('p2_cornea_kernel', 5),
        cornea_sigma=config.get('p2_cornea_sigma', 1.0),
        sharpening_strength=config.get('p2_sharpening', 0.3)
    )
    right_p2 = phase2_cornea_lens_refraction(
        right_p1,
        cornea_kernel=config.get('p2_cornea_kernel', 5),
        cornea_sigma=config.get('p2_cornea_sigma', 1.0),
        sharpening_strength=config.get('p2_sharpening', 0.3)
    )
    results['phase2'] = {'left': left_p2, 'right': right_p2}

    # --- Fase 3: Proyección retiniana ---
    left_p3 = phase3_retinal_projection(
        left_p2,
        fovea_ratio=config.get('p3_fovea_ratio', 0.3)
    )
    right_p3 = phase3_retinal_projection(
        right_p2,
        fovea_ratio=config.get('p3_fovea_ratio', 0.3)
    )
    results['phase3'] = {'left': left_p3, 'right': right_p3}

    # --- Fase 4: Impulsos nerviosos ---
    left_p4 = phase4_nerve_impulse_transformation(left_p3)
    right_p4 = phase4_nerve_impulse_transformation(right_p3)
    results['phase4'] = {'left': left_p4, 'right': right_p4}

    # --- Fase 5: Quiasma óptico ---
    hemispheres = phase5_optic_chiasma(left_p4, right_p4)
    results['phase5'] = hemispheres

    # --- Fase 6: Cintillas ópticas ---
    optic_tracts = phase6_optic_tracts(hemispheres)
    results['phase6'] = optic_tracts

    return results
