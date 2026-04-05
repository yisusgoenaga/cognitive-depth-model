# Modelo Cognitivo Artificial para la Replicación de la Actividad Neurofisiológica de Percepción de Profundidad

## Descripción

Este repositorio contiene el código fuente, los notebooks de entrenamiento y la documentación técnica del modelo cognitivo artificial desarrollado como parte de la tesis doctoral:

**"Modelo Cognitivo Artificial para la Replicación de la Actividad Neurofisiológica de Percepción de Profundidad"**

- **Autor:** Jesús Goenaga Peña
- **Programa:** Doctorado en Ciencias Cognitivas
- **Línea de investigación:** Sistemas Cognitivos Artificiales
- **Universidad:** Universidad Autónoma de Manizales
- **Tutor:** Luis Fernando Castillo Ossa

## Resumen del Proyecto

El modelo propuesto replica el procesamiento jerárquico del sistema visual humano implicado en la percepción de profundidad, mediante una arquitectura de aprendizaje profundo (CNN con ResNet) estructurada en **27 fases funcionales** inspiradas en la neurociencia cognitiva. Estas fases representan el flujo de procesamiento desde la entrada visual en la retina hasta la construcción de mapas espaciales dinámicos en la corteza visual superior (V5/MT).

### Arquitectura del Modelo

| Componente | Descripción |
|---|---|
| **Entrada** | Dos canales binoculares (imágenes estereoscópicas izquierda/derecha) |
| **Preprocesamiento** | Fases 1-6: Simulación retiniana, quiasma óptico, cintillas ópticas |
| **Procesamiento cortical** | Fases 7-27: NGL → V1 → V2 → V3/V3A → V4 → V5/MT |
| **Salida** | Clasificación binaria (sigmoid): "Más cercano" / "Más lejano" |

### Datasets

- **KITTI Scene Flow 2015** (Geiger et al., 2012): 400 escenas estereoscópicas reales
- **3D Visual Illusion Depth Estimation** (Yao et al., 2024): ~3,000 escenas con ilusiones de profundidad

## Estructura del Repositorio

```
cognitive-depth-model/
├── README.md                  # Este archivo
├── LICENSE                    # Licencia del proyecto
├── requirements.txt           # Dependencias de Python
├── setup_colab.py             # Script de configuración para Google Colab
├── configs/
│   └── model_config.yaml      # Hiperparámetros y configuración del modelo
├── data/
│   ├── raw/                   # Datasets originales (no versionados)
│   │   ├── kitti/
│   │   └── illusions/
│   ├── processed/             # Datos preprocesados
│   └── splits/                # Archivos de división train/val/test
├── src/
│   ├── preprocessing/         # Fases 1-6: Pipeline retiniano
│   ├── phases/                # Fases 7-27: Módulos corticales
│   ├── model/                 # Arquitectura completa del modelo
│   ├── training/              # Lógica de entrenamiento
│   ├── evaluation/            # Métricas y evaluación
│   └── utils/                 # Funciones auxiliares
├── notebooks/
│   ├── 01_setup_and_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_model_architecture.ipynb
│   ├── 04_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_explainability.ipynb
├── results/
│   ├── metrics/               # CSVs con resultados
│   ├── visualizations/        # Gráficas generadas
│   └── grad_cam/              # Mapas de saliencia
├── docs/                      # Documentación técnica adicional
└── tests/                     # Pruebas unitarias
```

## Configuración del Entorno

### Opción 1: Google Colab (Recomendado)

1. Abrir el notebook `notebooks/01_setup_and_exploration.ipynb` en Google Colab
2. Ejecutar la primera celda para instalar todas las dependencias
3. Conectar con Google Drive para persistencia de datos

### Opción 2: Entorno Local

```bash
# Clonar el repositorio
git clone https://github.com/yisusgoenaga/cognitive-depth-model.git
cd cognitive-depth-model

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Reproducibilidad

- Todas las semillas aleatorias están fijadas (seed = 42)
- Las versiones exactas de las dependencias se documentan en `requirements.txt`
- Los hiperparámetros se centralizan en `configs/model_config.yaml`
- Los resultados incluyen logs de entrenamiento completos

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Citación

Si utilizas este código o modelo en tu investigación, por favor cita:

```bibtex
@phdthesis{goenaga2026cognitive,
  title={Modelo Cognitivo Artificial para la Replicación de la Actividad Neurofisiológica de Percepción de Profundidad},
  author={Goenaga Peña, Jesús},
  year={2026},
  school={Universidad Autónoma de Manizales},
  program={Doctorado en Ciencias Cognitivas}
}
```

## Contacto

- **Jesús Goenaga Peña** — [GitHub](https://github.com/yisusgoenaga)
