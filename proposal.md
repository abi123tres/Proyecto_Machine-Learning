# 1. Título del proyecto

["Predicción de la duración de viajes en taxi amarillo de NYC con datos de la TLC" ]

# 2. Integrantes

| Nombre | Código UTEC | Rol principal |
|--------|-------------|---------------|
| Abigail Jaslin Cabanillas Ventocilla 
| | | |
| | | |

---

# 3. Dataset elegido

- **Nombre:** NYC Taxi Trip Records (Yellow Taxi)
- **Fuente:** NYC Taxi & Limousine Commission (TLC) — https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Formato:** Parquet, un archivo por mes
- **Licencia / acceso:** datos públicos, descarga libre sin registro
- **Periodo usado:** [ej. enero 2024 para la entrega previa; ampliar a 3–12 meses en la entrega final]
- **Tamaño:** [filas × columnas, aprox. X millones de filas]
- **Por qué es complejo:** [alto volumen, valores faltantes, outliers extremos, componente temporal y espacial, zonas categóricas de alta cardinalidad, etc.]
- **Diccionario de datos:** https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

---

# 4. Pregunta predictiva

<!-- Una sola oración, clara y accionable. -->

¿Se puede predecir la duración de un viaje en taxi (en minutos) usando únicamente la información disponible al momento de la recogida?

**Motivación:** [para qué serviría: estimar tiempo de llegada, planificar flota, informar al pasajero, etc.]

---

# 5. Variable objetivo

- **Nombre:** `duracion_min`
- **Definición:** `(tpep_dropoff_datetime - tpep_pickup_datetime)` en minutos
- **Tipo de problema:** regresión
- **Filtros aplicados al objetivo:** [ej. viajes entre 1 y 180 minutos; justificar el corte]

---

# 6. Unidad de predicción

Un viaje individual (una fila del dataset), predicho en el instante de la recogida.

---

# 7. Variables disponibles antes de la predicción

| Variable | Descripción | ¿Disponible al predecir? |
|----------|-------------|--------------------------|
| `tpep_pickup_datetime` (→ hora, día de semana, mes, feriado) | Momento de recogida | Sí |
| `PULocationID` | Zona de recogida | Sí |
| `DOLocationID` | Zona de destino | Sí, si el pasajero declara el destino [confirmar supuesto] |
| `passenger_count` | Número de pasajeros | Sí |
| `VendorID` | Proveedor | Sí |
| [otras] | | |

---

# 8. Riesgos de leakage

| Variable | Por qué es leakage | Medida de control |
|----------|--------------------|-------------------|
| `tpep_dropoff_datetime` | Con ella se calcula directamente el objetivo | Excluida |
| `trip_distance` | Se mide al terminar el viaje | [Excluida / reemplazada por distancia estimada entre zonas — decidir] |
| `fare_amount`, `tip_amount`, `total_amount`, `tolls_amount` | Se calculan al final del viaje | Excluidas |
| `payment_type` | Se registra al final | Excluida |
| [otros] | | |

**Leakage temporal:** [ej. la partición será por fecha, no aleatoria, para no entrenar con el futuro y evaluar con el pasado]

---

# 9. Métricas

- **Métrica principal:** MAE (minutos) — [justificar: interpretable, robusta frente a outliers comparada con RMSE]
- **Métrica secundaria:** RMSE y/o R² — [justificar]

---

# 10. Plan de validación

- **Partición:** [ej. entrenamiento: primeras 3 semanas; validación: semana 4; test final: reservado y no tocado hasta el final]
- **Tipo de validación:** [split temporal / TimeSeriesSplit / K-fold]
- **Regla:** el conjunto de test no se usa para tomar decisiones de modelado.
- **Semilla aleatoria:** 42

---

# 11. Modelo baseline

1. **Baseline trivial:** predecir la media (o mediana) de la duración en entrenamiento.
2. **Baseline lineal:** regresión lineal con las variables disponibles (hora, día, zonas codificadas, pasajeros).

Resultados esperados de la entrega previa: MAE y RMSE de ambos baselines, con una lectura corta de qué significan.

---

# 12. Riesgos técnicos

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Volumen de datos (memoria) | [ej. notebook se cae] | [muestreo, dtypes eficientes, parquet, Dask/Polars si hace falta] |
| Outliers extremos (duraciones 0 o de varias horas) | Distorsionan las métricas | [reglas de filtrado documentadas] |
| Alta cardinalidad de zonas | Muchas columnas al hacer one-hot | [target encoding, agrupar zonas, embeddings] |
| Cambios de patrón por evento/feriado | Error alto en ciertas fechas | [variables de calendario, análisis de errores por segmento] |
| [otro] | | |

**Sesgos y limitaciones iniciales:** [ej. solo taxis amarillos, no representa otros servicios; zonas con pocos viajes tienen predicciones menos confiables]

---

# 13. Plan de trabajo (semanas restantes)

| Semana | Actividad | Responsable |
|--------|-----------|-------------|
| 2 | Limpieza y feature engineering (`02_limpieza_features.ipynb`, `src/features.py`) | |
| 3–4 | Escalar a más meses, pipeline reproducible | |
| 5–7 | Modelos: lineal regularizado, árboles/Random Forest, Gradient Boosting | |
| 8–9 | Búsqueda de hiperparámetros y validación | |
| 10–11 | Evaluación final en test, interpretabilidad (importancia de variables, SHAP) | |
| 12–13 | Análisis de errores por segmento (hora, zona, distancia) | |
| 14 | Informe final (`reports/informe_final.md`) | |
| 15 | Presentación y revisión del README | |
| 16 | Entrega final | |

<!-- Ajusten las semanas según el cronograma real del curso. -->

---

## Anexo: estructura del repositorio

```
proyecto-final/
├── README.md
├── proposal.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── 01_exploracion_inicial.ipynb
├── src/
├── reports/
│   └── figures/
└── outputs/
```
