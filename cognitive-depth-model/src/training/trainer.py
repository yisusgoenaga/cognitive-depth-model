"""
============================================
Motor de Entrenamiento del Modelo Cognitivo
============================================

Implementa el entrenamiento en dos etapas:
  Etapa 1: Congelar fases 7-15, entrenar solo fases 16-27 + salida
  Etapa 2: Fine-tuning de todo el modelo con learning rate reducido

Incluye: early stopping, métricas, logging y checkpoints.

Autor: Jesús Goenaga Peña
Tesis Doctoral - Universidad Autónoma de Manizales
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class EarlyStopping:
    """Detiene el entrenamiento si la pérdida de validación no mejora."""

    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def compute_metrics(y_true, y_pred, y_prob):
    """Calcula métricas de clasificación binaria."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['auc'] = 0.0
    return metrics


def freeze_early_phases(model):
    """
    Etapa 1: Congela NGL y V1 (Fases 7-15).
    Solo entrena V2, V3, V4, V5/MT y la capa de salida.
    """
    for param in model.ngl.parameters():
        param.requires_grad = False
    for param in model.v1.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Fases 7-15 congeladas. Parámetros entrenables: {trainable:,} / {total:,}')


def unfreeze_all(model):
    """Etapa 2: Descongela todo el modelo para fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Modelo completo descongelado. Parámetros entrenables: {trainable:,}')


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena el modelo por una época."""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Recolectar predicciones
        probs = outputs.detach().cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        true_labels = labels.cpu().numpy().flatten()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(true_labels)

    avg_loss = running_loss / len(dataloader)
    # Filtrar etiquetas indeterminadas (0.5)
    valid_mask = [l != 0.5 for l in all_labels]
    if any(valid_mask):
        valid_labels = [l for l, v in zip(all_labels, valid_mask) if v]
        valid_preds = [p for p, v in zip(all_preds, valid_mask) if v]
        valid_probs = [p for p, v in zip(all_probs, valid_mask) if v]
        metrics = compute_metrics(valid_labels, valid_preds, valid_probs)
    else:
        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0}

    metrics['loss'] = avg_loss
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evalúa el modelo en un conjunto de datos."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        probs = outputs.cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        true_labels = labels.cpu().numpy().flatten()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(true_labels)

    avg_loss = running_loss / max(len(dataloader), 1)
    valid_mask = [l != 0.5 for l in all_labels]
    if any(valid_mask):
        valid_labels = [l for l, v in zip(all_labels, valid_mask) if v]
        valid_preds = [p for p, v in zip(all_preds, valid_mask) if v]
        valid_probs = [p for p, v in zip(all_probs, valid_mask) if v]
        metrics = compute_metrics(valid_labels, valid_preds, valid_probs)
    else:
        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0}

    metrics['loss'] = avg_loss
    return metrics


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    checkpoint_dir: str,
    stage1_epochs: int = 50,
    stage2_epochs: int = 150,
    stage1_lr: float = 0.001,
    stage2_lr: float = 0.0001,
    patience: int = 20,
):
    """
    Entrenamiento completo en dos etapas.

    Etapa 1: Capas bajas congeladas, LR alto
    Etapa 2: Todo descongelado, LR bajo (fine-tuning)

    Returns:
        Dict con historial de entrenamiento
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.BCELoss()
    history = {'train': [], 'test': [], 'stage': [], 'epoch': [], 'lr': []}

    best_test_loss = float('inf')
    total_epoch = 0

    # ==========================================
    # ETAPA 1: Congelar fases tempranas
    # ==========================================
    print('\n' + '=' * 60)
    print('ETAPA 1: Entrenamiento con fases 7-15 congeladas')
    print('=' * 60)

    freeze_early_phases(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage1_lr, betas=(0.9, 0.999), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-7
    )
    early_stop = EarlyStopping(patience=patience)

    for epoch in range(stage1_epochs):
        total_epoch += 1
        start_time = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_metrics['loss'])
        elapsed = time.time() - start_time

        # Logging
        history['train'].append(train_metrics)
        history['test'].append(test_metrics)
        history['stage'].append(1)
        history['epoch'].append(total_epoch)
        history['lr'].append(current_lr)

        print(f"E1 [{epoch+1:3d}/{stage1_epochs}] "
              f"Loss: {train_metrics['loss']:.4f}/{test_metrics['loss']:.4f} "
              f"Acc: {train_metrics['accuracy']:.3f}/{test_metrics['accuracy']:.3f} "
              f"F1: {train_metrics['f1']:.3f} "
              f"LR: {current_lr:.2e} "
              f"({elapsed:.1f}s)")

        # Guardar mejor modelo
        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            torch.save({
                'epoch': total_epoch,
                'stage': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': best_test_loss,
                'metrics': test_metrics,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))

        if early_stop(test_metrics['loss']):
            print(f'Early stopping en época {epoch+1}')
            break

    # ==========================================
    # ETAPA 2: Fine-tuning completo
    # ==========================================
    print('\n' + '=' * 60)
    print('ETAPA 2: Fine-tuning del modelo completo')
    print('=' * 60)

    # Cargar mejor modelo de etapa 1
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    unfreeze_all(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=stage2_lr, betas=(0.9, 0.999), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-7
    )
    early_stop = EarlyStopping(patience=patience)

    for epoch in range(stage2_epochs):
        total_epoch += 1
        start_time = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_metrics['loss'])
        elapsed = time.time() - start_time

        history['train'].append(train_metrics)
        history['test'].append(test_metrics)
        history['stage'].append(2)
        history['epoch'].append(total_epoch)
        history['lr'].append(current_lr)

        print(f"E2 [{epoch+1:3d}/{stage2_epochs}] "
              f"Loss: {train_metrics['loss']:.4f}/{test_metrics['loss']:.4f} "
              f"Acc: {train_metrics['accuracy']:.3f}/{test_metrics['accuracy']:.3f} "
              f"F1: {train_metrics['f1']:.3f} "
              f"LR: {current_lr:.2e} "
              f"({elapsed:.1f}s)")

        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            torch.save({
                'epoch': total_epoch,
                'stage': 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': best_test_loss,
                'metrics': test_metrics,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))

        # Checkpoint periódico
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': total_epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{total_epoch}.pth'))

        if early_stop(test_metrics['loss']):
            print(f'Early stopping en época {epoch+1}')
            break

    # Guardar modelo final
    torch.save({
        'epoch': total_epoch,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, os.path.join(checkpoint_dir, 'model_final.pth'))

    print(f'\nEntrenamiento completado. Mejor test loss: {best_test_loss:.4f}')
    return history
