# Modelo Cognitivo Artificial para la Replicación de la Actividad Neurofisiológica de Percepción de Profundidad

## Descripción

Este repositorio contiene el código fuente, los notebooks de desarrollo y validación, y la documentación técnica del modelo cognitivo artificial desarrollado como parte de la tesis doctoral:

**"Modelo Cognitivo Artificial para la Replicación de la Actividad Neurofisiológica de Percepción de Profundidad"**

- **Investigador principal:** Jesús Goenaga Peña (MSc)
- **Director:** Luis Fernando Castillo Ossa (PhD)
- **Programa:** Doctorado en Ciencias Cognitivas
- **Línea de investigación:** Sistemas Cognitivos Artificiales
- **Universidad:** Universidad Autónoma de Manizales
- **Financiación:** Convocatoria 22 de Minciencias — Formación de Capital Humano de Alto Nivel (BPIN 2022000100055)

## Resumen del Proyecto

El modelo propuesto replica el procesamiento jerárquico del sistema visual humano implicado en la percepción de profundidad, mediante una arquitectura de aprendizaje profundo (CNN con ResNet y ROI Pooling comparativo) estructurada en **27 fases funcionales** inspiradas en la neurociencia cognitiva. Estas fases representan el flujo de procesamiento desde la entrada visual en la retina hasta la construcción de mapas espaciales dinámicos en la corteza visual superior (V5/MT).

La validación empírica compara el desempeño del modelo con el de participantes humanos adultos sanos en 190 tareas de percepción de profundidad presentadas mediante realidad virtual (Meta Quest 2), bajo un diseño factorial 2 × 2 × 2 (tipo de tarea × nivel de disparidad × presencia de distractores).

### Arquitectura del Modelo

| Componente | Descripción |
|---|---|
| **Entrada** | Imagen completa (6 canales: par estereoscópico izquierda/derecha) + coordenadas de regiones A y B |
| **Preprocesamiento** | Fases 1-6: Simulación retiniana, quiasma óptico, cintillas ópticas |
| **Procesamiento cortical** | Fases 7-27: NGL → V1 → V2 → V3/V3A → V4 → V5/MT |
| **ROI Pooling** | Extracción y comparación de representaciones de las regiones A y B |
| **Salida** | Clasificación binaria (sigmoid): "Más cercano" / "Más lejano" |

### Datasets

- **KITTI Scene Flow 2015** (Geiger et al., 2012): 200 escenas de entrenamiento y 200 de prueba con pares estereoscópicos reales y mapas de disparidad LiDAR.
- **3D Visual Illusion Depth Estimation** (Yao et al., NeurIPS 2025; ArXiv: 2505.13061): Imágenes perceptualmente ambiguas que aumentan la dificultad de la tarea de percepción de profundidad.

## Pipeline de Notebooks

El desarrollo se ejecutó en Google Colaboratory con GPU NVIDIA Tesla T4. Cada notebook tiene un alcance discreto y produce outputs visibles:

| Notebook | Propósito |
|---|---|
| `01_setup_and_exploration` | Configuración del entorno, GPU, dependencias y reproducibilidad |
| `02_data_download_and_exploration` | Descarga, exploración y splits de ambos datasets |
| `03_preprocessing_pipeline` | Pipeline retiniano: Fases 1-6 (pupila → cintillas ópticas) |
| `04_model_architecture` | Definición y verificación de la arquitectura (Fases 7-27) |
| `05_training` | Entrenamiento inicial con KITTI (pares estereoscópicos) |
| `05b_retraining_kitti_illusions` | Reentrenamiento v2 con KITTI + ilusiones (ROI Pooling) |
| `06_evaluation_explainability` | Evaluación cuantitativa y Grad-CAM por fase cortical |
| `07_factorial_classification` | Clasificación de imágenes en el diseño factorial 2×2×2 |
| `08_pairs_ground_truth` | Definición de pares A/B y etiquetas ground truth |
| `09_validation_set_selection` | Selección balanceada de 190 tareas + corrección GT ilusiones |
| `10_model_validation_v2` | Validación del modelo v2 sobre las 190 tareas experimentales |
| `11_human_validation_protocol` | Protocolo interactivo de recolección de datos con participantes |
| `12_statistical_analysis` | Análisis estadístico comparativo: modelo vs. humanos |

## Estructura del Repositorio

```
cognitive-depth-model/
├── README.md
├── LICENSE
├── requirements.txt
├── setup_colab.py
├── .gitignore
├── configs/
│   └── model_config.yaml
├── data/
│   ├── raw/                    # Datasets originales (no versionados)
│   │   ├── kitti/
│   │   └── illusions/
│   ├── processed/              # Datos preprocesados
│   ├── splits/                 # Splits, CSVs factoriales, validation_set_final.csv
│   ├── participants/           # Datos de sesión por participante (JSON)
│   └── private/                # Mapeo nombre↔ID (no versionado)
├── src/
│   ├── preprocessing/          # Fases 1-6: Pipeline retiniano
│   ├── phases/                 # Fases 7-27: Módulos corticales
│   ├── model/                  # Arquitectura completa del modelo
│   ├── training/               # Lógica de entrenamiento y datasets
│   ├── evaluation/             # Métricas, Grad-CAM, explicabilidad
│   └── utils/                  # Funciones auxiliares
├── notebooks/                  # 13 notebooks (NB01–NB12 + NB05b)
├── results/
│   ├── metrics/                # CSVs con métricas de entrenamiento
│   ├── visualizations/         # Gráficas generadas
│   ├── grad_cam/               # Mapas de saliencia Grad-CAM
│   ├── pairs/                  # Pares A/B y ground truth
│   ├── model_validation/       # Respuestas del modelo v2
│   └── human_responses/        # Respuestas de participantes humanos (app Quest 2)
├── app_quest2/                 # Scripts de preparación para la app VR
├── docs/                       # Documentación técnica adicional
└── tests/                      # Pruebas unitarias
```

**Nota:** La aplicación de realidad virtual (Meta Quest 2) se mantiene en un repositorio separado: [`yisusgoenaga/depth-perception-vr`](https://github.com/yisusgoenaga/depth-perception-vr).

## Configuración del Entorno

### Google Colab (Recomendado)

1. Abrir `notebooks/01_setup_and_exploration.ipynb` en Google Colab.
2. Ejecutar las celdas en orden para instalar dependencias, montar Drive y clonar el repositorio.
3. Los notebooks posteriores se ejecutan en secuencia (NB01 → NB12).

### Entorno Local

```bash
git clone https://github.com/yisusgoenaga/cognitive-depth-model.git
cd cognitive-depth-model
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Reproducibilidad

- Semillas aleatorias fijadas (`seed = 42`) en PyTorch, NumPy y Python.
- Versiones exactas documentadas: Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8, GPU Tesla T4 (15.6 GB).
- Hiperparámetros centralizados en `configs/model_config.yaml`.
- Notebooks con outputs visibles para verificación sin re-ejecución.
- Repositorio público para transparencia y trazabilidad completa.

**Nota importante:** El notebook `05_training` y `05b_retraining_kitti_illusions` NO deben re-ejecutarse — los checkpoints (`best_model.pth`, `best_model_v2.pth`) son irreemplazables sin reentrenar desde cero.

## Resultados Principales

| Métrica | KITTI | Ilusiones |
|---|---|---|
| **Modelo v2** | 100.0% | 0.0% |
| **Humanos (pilotos)** | 95.8–97.9% | 100.0% |

- El modelo replica exitosamente el procesamiento estereoscópico (KITTI) pero no los mecanismos monoculares que los humanos emplean ante estímulos ambiguos (ilusiones).
- Las activaciones Grad-CAM siguen la jerarquía V2 → V3 → V5/MT, consistente con la literatura neurocientífica (Yamins & DiCarlo, 2016).
- El modelo muestra un sesgo de proximidad coherente con la prioridad evolutiva de detección de objetos cercanos.

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Citación

Si utilizas este código o modelo en tu investigación, por favor cita:

```bibtex
@phdthesis{goenaga2026cognitive,
  title={Modelo Cognitivo Artificial para la Replicación de la Actividad Neurofisiológica de Percepción de Profundidad},
  author={Goenaga Peña, Jesús},
  director={Castillo Ossa, Luis Fernando},
  year={2026},
  school={Universidad Autónoma de Manizales},
  program={Doctorado en Ciencias Cognitivas},
  note={Financiado por Minciencias, Convocatoria 22 (BPIN 2022000100055)}
}
```

## Contacto

- **Jesús Goenaga Peña** — [GitHub](https://github.com/yisusgoenaga) — jesus.goenagap@autonoma.edu.co
- **Luis Fernando Castillo Ossa** — luis.castillo@ucaldas.edu.co
