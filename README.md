# Predicción de incumplimiento de préstamos — Lending Club

Proyecto final del curso de Machine Learning. **Entrega final** (el informe está en `reports/informe_final.md`).

Clasificación binaria del riesgo de **incumplimiento (default)** de préstamos
personales de Lending Club, usando **únicamente** variables disponibles en el
momento de la decisión crediticia (control estricto de *leakage*).

- **Problema:** clasificación supervisada, `mal_desempeno` ∈ {0,1}.
- **Métrica principal:** ROC-AUC. **Secundarias:** PR-AUC y KS.
- **Validación:** partición temporal (out-of-time) por `issue_d`.
- **Baseline:** clase mayoritaria + regresión logística sin balanceo.

## Estructura
```
proyecto-final/
├── README.md
├── proposal.md                      # Propuesta (entrega previa)
├── requirements.txt
├── data/README.md                   # Cómo obtener el dataset
├── notebooks/
│   ├── 01_exploracion_inicial.ipynb
│   ├── 02_limpieza_features.ipynb
│   ├── 03_modelos.ipynb             # Comparación de modelos, CV temporal, test
│   └── 04_analisis_errores.ipynb    # Errores por segmento, interpretabilidad, sesgos
├── src/
│   ├── data.py                      # Carga, target y partición temporal
│   ├── features.py                  # Features, winsorización, preprocesador
│   ├── train.py                     # Pipeline principal de entrenamiento
│   ├── evaluate.py                  # Métricas, umbral, figuras, segmentos
│   └── eda.py                       # Figuras exploratorias sobre el dataset completo
├── reports/
│   ├── figures/                     # Todas las figuras del informe
│   └── informe_final.md
└── outputs/
    ├── metrics.json                 # Métricas de CV, umbral y test de todos los modelos
    ├── cv_results.csv / cv_all_trials.csv
    └── (regenerables, no versionados) predictions_test.parquet, models/
```

## Reproducir la entrega final (dataset completo)

```bash
pip install -r requirements.txt
# 1. Descargar accepted_2007_to_2018Q4.csv(.gz) en data/ (ver data/README.md)
# 2. Prueba rápida de humo (~3 min, 5% de los datos, escribe en outputs/quick/)
python -m src.train --quick
# 3. Entrenamiento completo (~15 min en un portátil de 8 GB / 8 núcleos)
python -m src.train --data data/accepted_2007_to_2018Q4.csv.gz
# 4. Figuras exploratorias sobre el dataset completo
python -m src.eda --data data/accepted_2007_to_2018Q4.csv.gz
# 5. Notebooks de análisis (leen outputs/; no reentrenan)
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modelos.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_analisis_errores.ipynb
```

`pandas` lee el `.csv.gz` directamente, no hace falta descomprimirlo. Todos los
pasos usan la semilla `RANDOM_SEED = 42`.

## Reproducir la exploración

Requiere Python 3.11.

**Opción A — Google Colab (recomendada)**

1. Abre `notebooks/01_exploracion_inicial.ipynb` en Colab.
2. Sube `accepted_2007_to_2018Q4.csv` a la misma carpeta del notebook
   (ver `data/README.md`).
3. *Entorno de ejecución → Reiniciar y ejecutar todo.*

**Opción B — Local**

```bash
python -m venv .venv && source .venv/bin/activate   # opcional
pip install -r requirements.txt
jupyter notebook notebooks/01_exploracion_inicial.ipynb
```

Coloca el CSV en `data/` y, en la celda de carga, usa
`DATA_PATH = "../data/accepted_2007_to_2018Q4.csv"`.

El notebook es autocontenido: genera las figuras en `reports/figures/` y las
métricas del baseline en `outputs/metrics.json`.

## Reproducibilidad

- Semilla global fija (`RANDOM_SEED = 42`), tanto para el muestreo como para el modelo.
- Todo el preprocesamiento (imputación, escalado, one-hot) se ajusta **solo con
  el conjunto de entrenamiento** dentro de un `Pipeline` de scikit-learn.
- Partición temporal: train = préstamos de 2007-06 a 2016-06, validación =
  2016-07 a 2018-12. El modelo nunca se evalúa sobre datos usados para entrenar.
- Dependencias listadas en `requirements.txt`.

## Control de leakage

El riesgo central de este dataset. La carga de datos usa una **lista blanca** de
variables de originación (`columnas_utiles` en el notebook), así que las columnas
generadas después de otorgar el crédito (pagos, recuperaciones, FICO actualizado,
acuerdos de liquidación/hardship) ni siquiera entran al análisis. `src/features.py`
documenta la **lista negra** (`LEAKAGE_COLUMNS`).
