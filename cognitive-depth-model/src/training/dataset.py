"""
============================================
Dataset para el Modelo Cognitivo Artificial
============================================

Carga pares estereoscópicos de KITTI Scene Flow 2015
y genera etiquetas binarias de profundidad relativa.

Autor: Jesús Goenaga Peña
Tesis Doctoral - Universidad Autónoma de Manizales
"""

import os
import json
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple


class KITTIStereoDepthDataset(Dataset):
    """
    Dataset de pares estereoscópicos KITTI con etiquetas
    de profundidad relativa (más cercano / más lejano).

    Para cada par estéreo, selecciona dos regiones de la imagen
    y usa el mapa de disparidad para determinar cuál está más cerca.

    Entrada al modelo: 6 canales (3 RGB izq + 3 RGB der)
    Etiqueta: 1 = región A más cercana, 0 = región A más lejana
    """

    def __init__(
        self,
        kitti_base_path: str,
        split_file: str,
        split_name: str = 'kitti_train',
        target_size: Tuple[int, int] = (256, 512),
        augment: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            kitti_base_path: Ruta a la carpeta KITTI en Drive
            split_file: Ruta al archivo kitti_splits.json
            split_name: 'kitti_train', 'kitti_test', o 'kitti_validation'
            target_size: (height, width) para resize
            augment: Aplicar data augmentation
            seed: Semilla para reproducibilidad
        """
        self.target_size = target_size
        self.augment = augment
        self.seed = seed

        # Cargar split
        with open(split_file, 'r') as f:
            splits = json.load(f)
        self.scene_ids = splits[split_name]

        # Encontrar las carpetas de KITTI
        self.left_dir = None
        self.right_dir = None
        self.disp_dir = None

        for root, dirs, files in os.walk(kitti_base_path):
            folder = os.path.basename(root)
            if folder == 'image_2' and 'training' in root:
                self.left_dir = root
            elif folder == 'image_3' and 'training' in root:
                self.right_dir = root
            elif folder == 'disp_occ_0' and 'training' in root:
                self.disp_dir = root

        # Verificar que encontramos las carpetas
        assert self.left_dir is not None, f"No se encontró image_2 en {kitti_base_path}"
        assert self.right_dir is not None, f"No se encontró image_3 en {kitti_base_path}"

        # Filtrar scene_ids que realmente existen en disco
        existing_ids = []
        for sid in self.scene_ids:
            left_path = os.path.join(self.left_dir, f'{sid}_10.png')
            if os.path.exists(left_path):
                existing_ids.append(sid)
        self.scene_ids = existing_ids

        print(f'Dataset {split_name}: {len(self.scene_ids)} escenas cargadas')

    def __len__(self):
        return len(self.scene_ids)

    def _load_and_preprocess(self, scene_id: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Carga y preprocesa un par estereoscópico."""
        # Cargar imágenes
        left_path = os.path.join(self.left_dir, f'{scene_id}_10.png')
        right_path = os.path.join(self.right_dir, f'{scene_id}_10.png')

        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        # Cargar disparidad si existe
        disp = None
        if self.disp_dir:
            disp_path = os.path.join(self.disp_dir, f'{scene_id}_10.png')
            if os.path.exists(disp_path):
                disp_raw = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
                disp = disp_raw.astype(np.float32) / 256.0

        # Resize
        h, w = self.target_size
        img_left = cv2.resize(img_left, (w, h), interpolation=cv2.INTER_LINEAR)
        img_right = cv2.resize(img_right, (w, h), interpolation=cv2.INTER_LINEAR)
        if disp is not None:
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR)

        return img_left, img_right, disp

    def _generate_depth_label(self, disp: np.ndarray) -> float:
        """
        Genera una etiqueta binaria de profundidad relativa.

        Selecciona dos regiones y compara su disparidad promedio.
        Mayor disparidad = más cercano.

        Returns:
            1.0 si la región A está más cerca, 0.0 si está más lejos
        """
        h, w = disp.shape[:2]

        # Seleccionar dos regiones aleatorias
        region_h, region_w = h // 4, w // 4

        # Región A (mitad izquierda)
        a_y = random.randint(0, h - region_h)
        a_x = random.randint(0, w // 2 - region_w)
        region_a = disp[a_y:a_y + region_h, a_x:a_x + region_w]

        # Región B (mitad derecha)
        b_y = random.randint(0, h - region_h)
        b_x = random.randint(w // 2, w - region_w)
        region_b = disp[b_y:b_y + region_h, b_x:b_x + region_w]

        # Filtrar píxeles válidos (disparidad > 0)
        valid_a = region_a[region_a > 0]
        valid_b = region_b[region_b > 0]

        if len(valid_a) == 0 or len(valid_b) == 0:
            return 0.5  # Indeterminado

        mean_a = valid_a.mean()
        mean_b = valid_b.mean()

        # Mayor disparidad = más cercano
        return 1.0 if mean_a > mean_b else 0.0

    def __getitem__(self, idx):
        scene_id = self.scene_ids[idx]

        # Cargar y preprocesar
        img_left, img_right, disp = self._load_and_preprocess(scene_id)

        # Generar etiqueta
        if disp is not None:
            label = self._generate_depth_label(disp)
        else:
            label = 0.5  # Sin ground truth

        # Data augmentation (solo brillo y contraste, NO flip horizontal)
        if self.augment:
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            img_left = np.clip(img_left * contrast + brightness * 10 - 5, 0, 255).astype(np.uint8)
            img_right = np.clip(img_right * contrast + brightness * 10 - 5, 0, 255).astype(np.uint8)

        # Normalizar a [0, 1]
        img_left = img_left.astype(np.float32) / 255.0
        img_right = img_right.astype(np.float32) / 255.0

        # Convertir a tensores (C, H, W)
        img_left = torch.from_numpy(img_left).permute(2, 0, 1)   # (3, H, W)
        img_right = torch.from_numpy(img_right).permute(2, 0, 1)  # (3, H, W)

        # Concatenar: 6 canales (izq + der)
        stereo_input = torch.cat([img_left, img_right], dim=0)  # (6, H, W)

        label_tensor = torch.tensor([label], dtype=torch.float32)

        return stereo_input, label_tensor


def create_dataloaders(
    kitti_base_path: str,
    split_file: str,
    batch_size: int = 16,
    target_size: Tuple[int, int] = (256, 512),
    num_workers: int = 2,
    seed: int = 42,
) -> dict:
    """
    Crea DataLoaders de train, test y validación.

    Returns:
        Dict con 'train', 'test', y 'validation' DataLoaders
    """
    # Train
    train_dataset = KITTIStereoDepthDataset(
        kitti_base_path, split_file, 'kitti_train',
        target_size=target_size, augment=True, seed=seed
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    # Test
    test_dataset = KITTIStereoDepthDataset(
        kitti_base_path, split_file, 'kitti_test',
        target_size=target_size, augment=False, seed=seed
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return {
        'train': train_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
    }
