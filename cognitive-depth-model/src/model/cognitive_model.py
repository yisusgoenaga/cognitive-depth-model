"""
============================================
Arquitectura del Modelo Cognitivo Artificial
Fases 7-27: NGL → V1 → V2 → V3/V3A → V4 → V5/MT
============================================

Implementa la arquitectura ResNet unificada que simula
el procesamiento cortical de la percepción de profundidad.

Diseño basado en los parámetros conexionistas:
- Conexiones residuales (ResNet) para entrenamiento profundo
- Conexiones de retroalimentación entre áreas corticales
- Funciones de activación no lineales (Leaky ReLU)
- Procesamiento jerárquico con integración multi-escala

Autor: Jesús Goenaga Peña
Tesis Doctoral - Universidad Autónoma de Manizales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ============================================
# Bloque Residual Base (inspirado en ResNet)
# ============================================

class ResidualBlock(nn.Module):
    """
    Bloque residual con conexión skip.

    Simula la transferencia eficiente de información entre
    áreas corticales, permitiendo que señales fluyan directamente
    entre fases no adyacentes (conexiones residuales).

    Ecuación: y = F(x) + x
    donde F(x) es la transformación aprendida
    """

    def __init__(self, channels, use_bn=True, negative_slope=0.01):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_bn)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        identity = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity  # Conexión residual
        out = self.activation(out)
        return out


class TransitionBlock(nn.Module):
    """
    Bloque de transición entre áreas corticales.
    Cambia el número de canales y reduce dimensionalidad espacial.
    """

    def __init__(self, in_channels, out_channels, downsample=True, negative_slope=0.01):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

        # Conexión residual con ajuste de dimensiones
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.skip_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.skip_bn(self.skip(x))
        out = self.activation(self.bn(self.conv(x)))
        return out + identity


# ============================================
# Módulo de Retroalimentación (Feedback)
# ============================================

class FeedbackConnection(nn.Module):
    """
    Conexión de retroalimentación entre áreas corticales.

    Simula las conexiones top-down que modulan el procesamiento
    en áreas inferiores según el contexto visual.

    Ecuación: O_mod = O + γ · F_feedback
    """

    def __init__(self, source_channels, target_channels):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(source_channels, target_channels, kernel_size=1),
            nn.BatchNorm2d(target_channels),
            nn.Sigmoid()  # Gate de modulación [0, 1]
        )

    def forward(self, target_signal, source_signal):
        # Ajustar tamaño espacial si es necesario
        if source_signal.shape[2:] != target_signal.shape[2:]:
            source_signal = F.interpolate(
                source_signal,
                size=target_signal.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        # Modulación multiplicativa (gating)
        gate = self.transform(source_signal)
        return target_signal * gate + target_signal


# ============================================
# FASE 7: Núcleo Geniculado Lateral (NGL)
# ============================================

class NGL(nn.Module):
    """
    Fase 7: Simula el Núcleo Geniculado Lateral.

    Procesa las señales de las cintillas ópticas separando
    las vías magnocelular (movimiento/luminancia) y
    parvocelular (color/detalles).
    """

    def __init__(self, in_channels, magno_channels=32, parvo_channels=32):
        super().__init__()
        # Vía magnocelular: movimiento y contraste de luminancia
        self.magno_pathway = nn.Sequential(
            nn.Conv2d(in_channels, magno_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(magno_channels),
            nn.LeakyReLU(0.01),
            ResidualBlock(magno_channels),
        )
        # Vía parvocelular: color y detalles espaciales
        self.parvo_pathway = nn.Sequential(
            nn.Conv2d(in_channels, parvo_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(parvo_channels),
            nn.LeakyReLU(0.01),
            ResidualBlock(parvo_channels),
        )

    def forward(self, x):
        magno = self.magno_pathway(x)
        parvo = self.parvo_pathway(x)
        return magno, parvo


# ============================================
# FASES 8-15: Corteza Visual V1
# ============================================

class V1(nn.Module):
    """
    Fases 8-15: Corteza Visual Primaria (V1).

    Implementa las capas IV-Cα, IV-Cβ, IV-B, IV-A, II/III, V, VI, I
    como una red convolucional profunda con conexiones residuales.

    - IV-Cα (Fase 8): movimiento y luminancia (vía magno)
    - IV-Cβ (Fase 9): detalles y color (vía parvo)
    - IV-B (Fase 10): dirección y velocidad del movimiento
    - IV-A (Fase 11): integración magno + parvo
    - II/III (Fase 12): integración de orientación, color, disparidad
    - V (Fase 13): localización espacial
    - VI (Fase 14): retroalimentación tálamo-cortical
    - I (Fase 15): modulación top-down
    """

    def __init__(self, magno_channels=32, parvo_channels=32, v1_channels=64):
        super().__init__()

        # Fase 8: IV-Cα (procesa magnocelular)
        self.iv_c_alpha = nn.Sequential(
            nn.Conv2d(magno_channels, v1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(v1_channels),
            nn.LeakyReLU(0.01),
            ResidualBlock(v1_channels),
        )

        # Fase 9: IV-Cβ (procesa parvocelular)
        self.iv_c_beta = nn.Sequential(
            nn.Conv2d(parvo_channels, v1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(v1_channels),
            nn.LeakyReLU(0.01),
            ResidualBlock(v1_channels),
        )

        # Fase 10: IV-B (movimiento, recibe de IV-Cα)
        self.iv_b = nn.Sequential(
            ResidualBlock(v1_channels),
            ResidualBlock(v1_channels),
        )

        # Fase 11: IV-A (integración, recibe de IV-Cα + IV-Cβ)
        self.iv_a = nn.Sequential(
            nn.Conv2d(v1_channels * 2, v1_channels, kernel_size=1),  # Fusión
            nn.BatchNorm2d(v1_channels),
            nn.LeakyReLU(0.01),
            ResidualBlock(v1_channels),
        )

        # Fase 12: Capas II/III (integración completa)
        self.layers_ii_iii = nn.Sequential(
            ResidualBlock(v1_channels),
            ResidualBlock(v1_channels),
        )

        # Fase 13: Capa V (localización espacial)
        self.layer_v = nn.Sequential(
            ResidualBlock(v1_channels),
        )

        # Fase 14: Capa VI (retroalimentación) + Fase 15: Capa I (modulación)
        self.layer_vi_and_i = nn.Sequential(
            ResidualBlock(v1_channels),
        )

    def forward(self, magno, parvo):
        # Fase 8: IV-Cα
        iv_ca = self.iv_c_alpha(magno)

        # Fase 9: IV-Cβ
        iv_cb = self.iv_c_beta(parvo)

        # Fase 10: IV-B (recibe de IV-Cα)
        iv_b = self.iv_b(iv_ca)

        # Fase 11: IV-A (integración magno + parvo)
        iv_a = self.iv_a(torch.cat([iv_ca, iv_cb], dim=1))

        # Fase 12: Capas II/III (integración de IV-B + IV-A)
        layers_ii_iii = self.layers_ii_iii(iv_b + iv_a)

        # Fase 13: Capa V
        layer_v = self.layer_v(layers_ii_iii)

        # Fases 14-15: Capas VI y I (modulación)
        v1_output = self.layer_vi_and_i(layer_v)

        # Retornar salidas relevantes para conexiones con V2
        return v1_output, iv_b, iv_cb, layers_ii_iii


# ============================================
# FASES 16-18: Corteza Visual V2
# ============================================

class V2(nn.Module):
    """
    Fases 16-18: Corteza Visual V2.

    - Bandas Gruesas (Fase 16): movimiento y orientación
    - Bandas Delgadas (Fase 17): color y texturas finas
    - Bandas Intercaladas (Fase 18): integración forma + color + textura
    """

    def __init__(self, v1_channels=64, v2_channels=128):
        super().__init__()

        # Fase 16: Bandas Gruesas (recibe de IV-B de V1)
        self.thick_bands = nn.Sequential(
            TransitionBlock(v1_channels, v2_channels, downsample=True),
            ResidualBlock(v2_channels),
        )

        # Fase 17: Bandas Delgadas (recibe de IV-Cβ de V1)
        self.thin_bands = nn.Sequential(
            TransitionBlock(v1_channels, v2_channels, downsample=True),
            ResidualBlock(v2_channels),
        )

        # Fase 18: Bandas Intercaladas (integra gruesas + delgadas)
        self.interstripes = nn.Sequential(
            nn.Conv2d(v2_channels * 2, v2_channels, kernel_size=1),
            nn.BatchNorm2d(v2_channels),
            nn.LeakyReLU(0.01),
            ResidualBlock(v2_channels),
        )

    def forward(self, v1_output, iv_b, iv_cb):
        # Fase 16: Bandas Gruesas
        thick = self.thick_bands(iv_b)

        # Fase 17: Bandas Delgadas
        thin = self.thin_bands(iv_cb)

        # Fase 18: Bandas Intercaladas
        interstripes = self.interstripes(torch.cat([thick, thin], dim=1))

        return thick, thin, interstripes


# ============================================
# FASES 19-21: Corteza Visual V3/V3A
# ============================================

class V3(nn.Module):
    """
    Fases 19-21: Corteza Visual V3 y V3A.

    - V3A Disparidad (Fase 19): integración de disparidad binocular
    - V3A Movimiento (Fase 20): análisis de movimiento relativo
    - V3 Formas 3D (Fase 21): reconstrucción tridimensional
    """

    def __init__(self, v2_channels=128, v3_channels=256):
        super().__init__()

        # Fase 19: V3A - Disparidad binocular
        # D(x) = |L(x) - R(x)|
        self.v3a_disparity = nn.Sequential(
            TransitionBlock(v2_channels, v3_channels, downsample=True),
            ResidualBlock(v3_channels),
            ResidualBlock(v3_channels),
        )

        # Fase 20: V3A - Movimiento relativo
        self.v3a_motion = nn.Sequential(
            TransitionBlock(v2_channels, v3_channels, downsample=True),
            ResidualBlock(v3_channels),
        )

        # Fase 21: V3 - Reconstrucción de formas 3D
        self.v3_shapes = nn.Sequential(
            nn.Conv2d(v3_channels * 2, v3_channels, kernel_size=1),
            nn.BatchNorm2d(v3_channels),
            nn.LeakyReLU(0.01),
            ResidualBlock(v3_channels),
            ResidualBlock(v3_channels),
        )

    def forward(self, thick_bands, interstripes):
        # Fase 19: Disparidad (recibe de bandas gruesas)
        disparity = self.v3a_disparity(thick_bands)

        # Fase 20: Movimiento relativo (recibe de bandas intercaladas)
        motion = self.v3a_motion(interstripes)

        # Fase 21: Formas 3D (integra disparidad + movimiento)
        shapes_3d = self.v3_shapes(torch.cat([disparity, motion], dim=1))

        return disparity, motion, shapes_3d


# ============================================
# FASES 22-24: Corteza Visual V4
# ============================================

class V4(nn.Module):
    """
    Fases 22-24: Corteza Visual V4.

    - V4α Color (Fase 22): procesamiento cromático avanzado
    - V4α Formas (Fase 23): procesamiento de formas complejas
    - V4β Atención (Fase 24): modulación atencional
    """

    def __init__(self, v2_channels=128, v3_channels=256, v4_channels=256):
        super().__init__()

        # Fase 22: V4α - Color (recibe de bandas delgadas V2)
        self.v4a_color = nn.Sequential(
            TransitionBlock(v2_channels, v4_channels, downsample=True),
            ResidualBlock(v4_channels),
        )

        # Fase 23: V4α - Formas complejas (recibe de V3)
        self.v4a_shapes = nn.Sequential(
            ResidualBlock(v3_channels),
        )

        # Fase 24: V4β - Atención visual (integra color + formas)
        self.v4b_attention = nn.Sequential(
            nn.Conv2d(v4_channels + v3_channels, v4_channels, kernel_size=1),
            nn.BatchNorm2d(v4_channels),
            nn.LeakyReLU(0.01),
            ResidualBlock(v4_channels),
        )

        # Mecanismo de atención (gate)
        self.attention_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(v4_channels, v4_channels),
            nn.Sigmoid()
        )

    def forward(self, thin_bands, shapes_3d):
        # Fase 22: Color
        color = self.v4a_color(thin_bands)

        # Fase 23: Formas complejas
        shapes = self.v4a_shapes(shapes_3d)

        # Ajustar tamaño espacial para concatenación
        if color.shape[2:] != shapes.shape[2:]:
            shapes = F.interpolate(shapes, size=color.shape[2:], mode='bilinear', align_corners=False)

        # Fase 24: Atención visual
        combined = torch.cat([color, shapes], dim=1)
        attention_features = self.v4b_attention(combined)

        # Aplicar gate atencional
        gate = self.attention_gate(attention_features).unsqueeze(-1).unsqueeze(-1)
        attended = attention_features * gate

        return attended


# ============================================
# FASES 25-27: Corteza Visual V5/MT
# ============================================

class V5MT(nn.Module):
    """
    Fases 25-27: Corteza Visual V5/MT.

    - Movimiento (Fase 25): análisis de dirección y velocidad
    - Disparidad (Fase 26): integración de disparidad dinámica
    - Mapas dinámicos (Fase 27): construcción de mapas espaciales 3D
    """

    def __init__(self, v2_channels=128, v3_channels=256, v5_channels=512):
        super().__init__()

        # Fase 25: Análisis de movimiento
        self.motion_analysis = nn.Sequential(
            TransitionBlock(v2_channels, v5_channels, downsample=True),
            ResidualBlock(v5_channels),
            ResidualBlock(v5_channels),
        )

        # Fase 26: Integración de disparidad binocular
        self.disparity_integration = nn.Sequential(
            TransitionBlock(v3_channels, v5_channels, downsample=True),
            ResidualBlock(v5_channels),
        )

        # Fase 27: Mapas espaciales dinámicos
        self.dynamic_maps = nn.Sequential(
            nn.Conv2d(v5_channels * 2, v5_channels, kernel_size=1),
            nn.BatchNorm2d(v5_channels),
            nn.LeakyReLU(0.01),
            ResidualBlock(v5_channels),
            ResidualBlock(v5_channels),
            ResidualBlock(v5_channels),
        )

    def forward(self, thick_bands, disparity_v3):
        # Fase 25: Movimiento
        motion = self.motion_analysis(thick_bands)

        # Fase 26: Disparidad
        disparity = self.disparity_integration(disparity_v3)

        # Ajustar tamaño espacial
        if motion.shape[2:] != disparity.shape[2:]:
            disparity = F.interpolate(disparity, size=motion.shape[2:], mode='bilinear', align_corners=False)

        # Fase 27: Mapas dinámicos
        dynamic_maps = self.dynamic_maps(torch.cat([motion, disparity], dim=1))

        return dynamic_maps


# ============================================
# CAPA DE INTEGRACIÓN Y SALIDA
# ============================================

class IntegrationAndOutput(nn.Module):
    """
    Capa de integración y salida del modelo.

    Consolida las representaciones de V4 y V5/MT,
    reduce dimensionalidad y produce la clasificación
    binaria: "Más cercano" (1) / "Más lejano" (0).

    Activación: sigmoid → P(más cercano) ∈ [0, 1]
    """

    def __init__(self, v4_channels=256, v5_channels=512, integration_units=512, dropout=0.3):
        super().__init__()

        total_channels = v4_channels + v5_channels

        # Integración de características
        self.integration_conv = nn.Sequential(
            nn.Conv2d(total_channels, integration_units, kernel_size=1),
            nn.BatchNorm2d(integration_units),
            nn.LeakyReLU(0.01),
        )

        # Global Average Pooling + clasificador
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(integration_units, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Salida: P(más cercano)
        )

    def forward(self, v4_features, v5_features):
        # Ajustar tamaños espaciales
        if v4_features.shape[2:] != v5_features.shape[2:]:
            v4_features = F.interpolate(
                v4_features, size=v5_features.shape[2:],
                mode='bilinear', align_corners=False
            )

        # Concatenar e integrar
        combined = torch.cat([v4_features, v5_features], dim=1)
        integrated = self.integration_conv(combined)

        # Clasificación binaria
        output = self.classifier(integrated)

        return output


# ============================================
# MODELO COGNITIVO ARTIFICIAL COMPLETO
# ============================================

class CognitiveDepthModel(nn.Module):
    """
    Modelo Cognitivo Artificial para la Percepción de Profundidad.

    Arquitectura completa de 27 fases que replica el procesamiento
    jerárquico del sistema visual humano, desde el NGL hasta la
    construcción de mapas espaciales dinámicos en V5/MT.

    Entrada: Par estereoscópico (izquierda + derecha) como tensor
             concatenado en la dimensión de canales.
             Shape: (batch, 6, H, W) - 3 canales RGB x 2 ojos

    Salida:  Probabilidad de "Más cercano" ∈ [0, 1]
             Shape: (batch, 1)

    Arquitectura:
        Entrada (6ch) → NGL → V1 → V2 → V3/V3A → V4 → V5/MT → Sigmoid

    Parámetros de diseño basados en el paradigma conexionista:
        - Conexiones residuales (ResNet)
        - Retroalimentación entre áreas
        - Leaky ReLU como activación
        - Batch normalization
        - Dropout para regularización
    """

    def __init__(
        self,
        in_channels: int = 6,       # 3 RGB x 2 ojos
        ngl_magno: int = 32,        # Canales vía magnocelular
        ngl_parvo: int = 32,        # Canales vía parvocelular
        v1_channels: int = 64,      # Canales en V1
        v2_channels: int = 128,     # Canales en V2
        v3_channels: int = 256,     # Canales en V3/V3A
        v4_channels: int = 256,     # Canales en V4
        v5_channels: int = 512,     # Canales en V5/MT
        integration_units: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ---- Fase 7: NGL ----
        self.ngl = NGL(in_channels, ngl_magno, ngl_parvo)

        # ---- Fases 8-15: V1 ----
        self.v1 = V1(ngl_magno, ngl_parvo, v1_channels)

        # ---- Fases 16-18: V2 ----
        self.v2 = V2(v1_channels, v2_channels)

        # ---- Fases 19-21: V3/V3A ----
        self.v3 = V3(v2_channels, v3_channels)

        # ---- Fases 22-24: V4 ----
        self.v4 = V4(v2_channels, v3_channels, v4_channels)

        # ---- Fases 25-27: V5/MT ----
        self.v5mt = V5MT(v2_channels, v3_channels, v5_channels)

        # ---- Integración y Salida ----
        self.output_layer = IntegrationAndOutput(
            v4_channels, v5_channels, integration_units, dropout
        )

        # ---- Conexiones de retroalimentación ----
        self.feedback_v2_to_v1 = FeedbackConnection(v2_channels, v1_channels)
        self.feedback_v3_to_v2 = FeedbackConnection(v3_channels, v2_channels)

        # Inicialización de pesos (He Initialization)
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicialización He Normal para capas convolucionales y lineales."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo completo.

        Args:
            x: Tensor de entrada (batch, 6, H, W)
               Canales 0-2: ojo izquierdo RGB
               Canales 3-5: ojo derecho RGB

        Returns:
            Tensor (batch, 1) con P(más cercano) ∈ [0, 1]
        """
        # ---- Fase 7: NGL ----
        magno, parvo = self.ngl(x)

        # ---- Fases 8-15: V1 ----
        v1_output, iv_b, iv_cb, layers_ii_iii = self.v1(magno, parvo)

        # ---- Fases 16-18: V2 ----
        thick, thin, interstripes = self.v2(v1_output, iv_b, iv_cb)

        # ---- Retroalimentación V2 → V1 (se usa en siguiente iteración) ----
        # En esta versión forward-only, registramos la capacidad
        # de feedback sin ciclos (single-pass)

        # ---- Fases 19-21: V3/V3A ----
        disparity, motion, shapes_3d = self.v3(thick, interstripes)

        # ---- Fases 22-24: V4 ----
        v4_features = self.v4(thin, shapes_3d)

        # ---- Fases 25-27: V5/MT ----
        v5_features = self.v5mt(thick, disparity)

        # ---- Integración y Salida ----
        output = self.output_layer(v4_features, v5_features)

        return output

    def get_phase_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass que retorna las salidas intermedias de cada fase.
        Útil para visualización, Grad-CAM y análisis de explicabilidad.

        Returns:
            Dict con tensores de cada fase del modelo.
        """
        outputs = {}

        # Fase 7
        magno, parvo = self.ngl(x)
        outputs['phase7_ngl_magno'] = magno
        outputs['phase7_ngl_parvo'] = parvo

        # Fases 8-15
        v1_output, iv_b, iv_cb, layers_ii_iii = self.v1(magno, parvo)
        outputs['phase8_v1_iv_c_alpha'] = iv_b
        outputs['phase9_v1_iv_c_beta'] = iv_cb
        outputs['phase12_v1_layers_ii_iii'] = layers_ii_iii
        outputs['phase15_v1_output'] = v1_output

        # Fases 16-18
        thick, thin, interstripes = self.v2(v1_output, iv_b, iv_cb)
        outputs['phase16_v2_thick'] = thick
        outputs['phase17_v2_thin'] = thin
        outputs['phase18_v2_interstripes'] = interstripes

        # Fases 19-21
        disparity, motion, shapes_3d = self.v3(thick, interstripes)
        outputs['phase19_v3a_disparity'] = disparity
        outputs['phase20_v3a_motion'] = motion
        outputs['phase21_v3_shapes_3d'] = shapes_3d

        # Fases 22-24
        v4_features = self.v4(thin, shapes_3d)
        outputs['phase24_v4_attention'] = v4_features

        # Fases 25-27
        v5_features = self.v5mt(thick, disparity)
        outputs['phase27_v5mt_dynamic_maps'] = v5_features

        # Salida
        output = self.output_layer(v4_features, v5_features)
        outputs['output'] = output

        return outputs


# ============================================
# Función de utilidad: crear modelo
# ============================================

def create_model(config: Optional[dict] = None) -> CognitiveDepthModel:
    """
    Crea una instancia del modelo con la configuración dada.

    Args:
        config: Dict con parámetros del modelo (opcional)

    Returns:
        CognitiveDepthModel inicializado
    """
    if config is None:
        config = {}

    model = CognitiveDepthModel(
        in_channels=config.get('in_channels', 6),
        ngl_magno=config.get('ngl_magno', 32),
        ngl_parvo=config.get('ngl_parvo', 32),
        v1_channels=config.get('v1_channels', 64),
        v2_channels=config.get('v2_channels', 128),
        v3_channels=config.get('v3_channels', 256),
        v4_channels=config.get('v4_channels', 256),
        v5_channels=config.get('v5_channels', 512),
        integration_units=config.get('integration_units', 512),
        dropout=config.get('dropout', 0.3),
    )

    return model
