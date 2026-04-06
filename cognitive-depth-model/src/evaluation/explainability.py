"""
============================================
Evaluación y Explicabilidad del Modelo
============================================

Incluye:
- Métricas completas (accuracy, precision, recall, F1, AUC)
- Curva ROC
- Matrices de confusión
- Grad-CAM para visualización de regiones de interés
- Análisis de activaciones por fase

Autor: Jesús Goenaga Peña
Tesis Doctoral - Universidad Autónoma de Manizales
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple


# ============================================
# Evaluación completa del modelo
# ============================================

@torch.no_grad()
def full_evaluation(model, dataloader, device):
    """
    Evaluación completa del modelo con todas las métricas.

    Returns:
        Dict con métricas, labels reales y probabilidades predichas
    """
    model.eval()
    all_labels = []
    all_probs = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = outputs.cpu().numpy().flatten()
        true_labels = labels.numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(true_labels)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Filtrar etiquetas indeterminadas
    valid = all_labels != 0.5
    labels_v = all_labels[valid]
    probs_v = all_probs[valid]
    preds_v = (probs_v >= 0.5).astype(int)

    results = {
        'labels': labels_v,
        'probabilities': probs_v,
        'predictions': preds_v,
        'accuracy': accuracy_score(labels_v, preds_v),
        'precision': precision_score(labels_v, preds_v, zero_division=0),
        'recall': recall_score(labels_v, preds_v, zero_division=0),
        'f1': f1_score(labels_v, preds_v, zero_division=0),
        'confusion_matrix': confusion_matrix(labels_v, preds_v),
        'classification_report': classification_report(
            labels_v, preds_v,
            target_names=['Más lejano', 'Más cercano'],
            zero_division=0
        ),
    }

    try:
        results['auc'] = roc_auc_score(labels_v, probs_v)
        fpr, tpr, thresholds = roc_curve(labels_v, probs_v)
        results['roc_fpr'] = fpr
        results['roc_tpr'] = tpr
        results['roc_thresholds'] = thresholds
    except ValueError:
        results['auc'] = 0.0

    return results


def plot_evaluation_summary(results, save_path=None):
    """Genera gráficas de evaluación: ROC, matriz de confusión, distribución."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Evaluación del Modelo Cognitivo Artificial',
                 fontsize=14, fontweight='bold')

    # 1. Curva ROC
    if 'roc_fpr' in results:
        axes[0].plot(results['roc_fpr'], results['roc_tpr'],
                     'b-', linewidth=2, label=f'AUC = {results["auc"]:.3f}')
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Azar (0.5)')
        axes[0].set_xlabel('Tasa de Falsos Positivos')
        axes[0].set_ylabel('Tasa de Verdaderos Positivos')
        axes[0].set_title('Curva ROC')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)

    # 2. Matriz de confusión
    cm = results['confusion_matrix']
    im = axes[1].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Más lejano', 'Más cercano'])
    axes[1].set_yticklabels(['Más lejano', 'Más cercano'])
    axes[1].set_xlabel('Predicción')
    axes[1].set_ylabel('Real')
    axes[1].set_title('Matriz de Confusión')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=16, fontweight='bold',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # 3. Distribución de probabilidades
    labels = results['labels']
    probs = results['probabilities']
    axes[2].hist(probs[labels == 0], bins=20, alpha=0.6, color='blue', label='Más lejano')
    axes[2].hist(probs[labels == 1], bins=20, alpha=0.6, color='red', label='Más cercano')
    axes[2].axvline(x=0.5, color='black', linestyle='--', label='Umbral (0.5)')
    axes[2].set_xlabel('Probabilidad predicha')
    axes[2].set_ylabel('Frecuencia')
    axes[2].set_title('Distribución de Predicciones')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================
# Grad-CAM para explicabilidad
# ============================================

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.

    Visualiza qué regiones de la imagen influyen más en la
    decisión del modelo, permitiendo establecer correspondencias
    funcionales con áreas del sistema visual.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registrar hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Genera el mapa Grad-CAM para una entrada.

        Args:
            input_tensor: Tensor de entrada (1, C, H, W)
            target_class: Clase objetivo (None = predicción del modelo)

        Returns:
            Mapa de calor normalizado (H, W)
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = (output > 0.5).float()

        self.model.zero_grad()
        output.backward(target_class)

        # Pesos: promedio global de los gradientes
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Mapa de activación ponderado
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Solo activaciones positivas
        cam = cam.squeeze().cpu().numpy()

        # Normalizar a [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def visualize_gradcam(model, input_tensor, target_layers_dict, save_path=None):
    """
    Genera visualizaciones Grad-CAM para múltiples capas del modelo.

    Args:
        model: Modelo entrenado
        input_tensor: Tensor de entrada (1, 6, H, W)
        target_layers_dict: Dict {nombre: capa} para visualizar
        save_path: Ruta para guardar la imagen
    """
    n_layers = len(target_layers_dict)
    fig, axes = plt.subplots(2, max(n_layers, 1), figsize=(5 * max(n_layers, 1), 10))
    fig.suptitle('Grad-CAM: Regiones de Interés por Fase Cortical',
                 fontsize=14, fontweight='bold')

    if n_layers == 1:
        axes = axes.reshape(2, 1)

    # Imagen original (ojo izquierdo)
    img_left = input_tensor[0, :3].permute(1, 2, 0).cpu().numpy()
    img_left = np.clip(img_left[:, :, ::-1], 0, 1)  # BGR a RGB

    for idx, (name, layer) in enumerate(target_layers_dict.items()):
        if idx >= axes.shape[1]:
            break

        try:
            grad_cam = GradCAM(model, layer)
            cam = grad_cam.generate(input_tensor)

            # Resize CAM al tamaño de la imagen
            cam_resized = cv2.resize(cam, (img_left.shape[1], img_left.shape[0]))

            # Superponer en la imagen
            heatmap = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap = heatmap.astype(np.float32) / 255.0
            overlay = 0.6 * img_left + 0.4 * heatmap[:, :, ::-1]
            overlay = np.clip(overlay, 0, 1)

            # Mostrar mapa de calor
            axes[0, idx].imshow(cam_resized, cmap='jet')
            axes[0, idx].set_title(f'{name}\n(mapa de calor)')
            axes[0, idx].axis('off')

            # Mostrar superposición
            axes[1, idx].imshow(overlay)
            axes[1, idx].set_title(f'{name}\n(superposición)')
            axes[1, idx].axis('off')

        except Exception as e:
            axes[0, idx].text(0.5, 0.5, f'Error:\n{str(e)[:50]}',
                            ha='center', va='center', transform=axes[0, idx].transAxes)
            axes[0, idx].set_title(name)
            axes[1, idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================
# Análisis de activaciones por fase
# ============================================

def visualize_phase_activations_real(model, input_tensor, save_path=None):
    """
    Visualiza las activaciones de fases clave con una imagen real.

    A diferencia del Notebook 04 (entrada aleatoria), aquí se usa
    una imagen real del dataset para ver qué patrones detecta cada fase.
    """
    model.eval()
    with torch.no_grad():
        phase_outputs = model.get_phase_outputs(input_tensor)

    key_phases = [
        ('phase7_ngl_magno', 'Fase 7: NGL\n(Magnocelular)'),
        ('phase12_v1_layers_ii_iii', 'Fase 12: V1\n(Capas II/III)'),
        ('phase18_v2_interstripes', 'Fase 18: V2\n(Intercaladas)'),
        ('phase19_v3a_disparity', 'Fase 19: V3A\n(Disparidad)'),
        ('phase24_v4_attention', 'Fase 24: V4\n(Atención)'),
        ('phase27_v5mt_dynamic_maps', 'Fase 27: V5/MT\n(Mapas 3D)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Activaciones de Fases Clave con Imagen Real (post-entrenamiento)',
                 fontsize=14, fontweight='bold')

    for idx, (key, title) in enumerate(key_phases):
        ax = axes[idx // 3, idx % 3]
        if key in phase_outputs:
            tensor = phase_outputs[key]
            # Promedio de todos los filtros para ver patrón general
            activation = tensor[0].mean(dim=0).cpu().numpy()
            im = ax.imshow(activation, cmap='viridis')
            ax.set_title(f'{title}\n{list(tensor.shape)}')
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, f'{key}\nno disponible', ha='center', va='center')
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
