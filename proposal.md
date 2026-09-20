# 1. Título del proyecto

["Predicción de riesgo crediticio (default) en préstamos peer-to-peer con datos de Lending Club"]

# 2. Integrantes

| Nombre | Código UTEC | Rol principal |
|--------|-------------|---------------|
| Abigail Jaslin Cabanillas Ventocilla | 202510438 | |
| | | |
| | | |

# 3. Dataset elegido

- **Nombre:** Lending Club Loan Data
- **Fuente:** Kaggle / Lending Club — https://www.kaggle.com/datasets/wordsforthewise/lending-club
- **Formato:** CSV, un archivo consolidado con múltiples años de originación de préstamos
- **Licencia / acceso:** datos públicos vía Kaggle, descarga libre
- **Periodo usado:** [2007–2018, dataset completo sin muestreo]
- **Tamaño:** [filas: 2,260,701,  columnas: 24]
- **Por qué es complejo:** dataset de gran volumen con más de dos millones de registros, variables con riesgo real de leakage temporal a través de pagos y recuperaciones registrados después del desembolso, un componente 
temporal de más de una década que exige split no aleatorio, y variables con muchas categorías distintas.

---

# 4. Pregunta predictiva

¿Se puede predecir, al momento de la aprobación del préstamo, si un solicitante caerá en default o tendrá un mal desempeño de pago, usando únicamente la información disponible antes del desembolso?

**Motivación:** estimar el riesgo crediticio de un solicitante ayudaría a mejorar las decisiones de aprobación y a fijar tasas de interés más justas según el riesgo real.

---

# 5. Variable objetivo

- **Nombre:** `mal_desempeno` (binaria, derivada de `loan_status`)
- **Definición:**
  - **1 (mal desempeño):** Charged Off, Default, Late (31–120 days), Does not meet the credit policy - Charged Off
  - **0 (buen desempeño):** Fully Paid, Current, Does not meet the credit policy - Fully Paid
- **Tipo de problema:** clasificación binaria
- **Filtros aplicados al objetivo:** se excluyen préstamos con estados ambiguos o muy recientes (ej. "In Grace Period", "Issued") para evitar ruido en el label

---

# 6. Unidad de predicción

Un préstamo individual (`id` / `member_id`), predicho en el instante de su originación/aprobación.

---

# 7. Variables disponibles antes de la predicción

| Variable | Descripción | ¿Disponible al predecir? |
|----------|-------------|--------------------------|
| `loan_amnt` | Monto del préstamo solicitado/otorgado (USD) | Sí |
| `term` | Plazo del préstamo (36 o 60 meses) | Sí |
| `int_rate` | Tasa de interés anual asignada | Sí, pero se evalúa con cuidado [ver sección 8] |
| `installment` | Cuota mensual del préstamo | Sí |
| `grade` | Calificación de riesgo asignada por Lending Club (A–G) | Sí, pero se evalúa con cuidado [ver sección 8] |
| `sub_grade` | Sub-calificación de riesgo (A1–G5, 35 valores) | Sí, pero se evalúa con cuidado [ver sección 8] |
| `emp_length` | Años de antigüedad laboral del solicitante | Sí |
| `home_ownership` | Situación de vivienda (RENT, OWN, MORTGAGE, OTHER) | Sí |
| `annual_inc` | Ingreso anual autodeclarado | Sí |
| `purpose` | Motivo declarado del préstamo | Sí |
| `dti` | Ratio deuda/ingreso mensual (sin hipoteca ni este préstamo) | Sí |
| `fico_range_low` / `fico_range_high` | Rango del puntaje crediticio FICO | Sí |
| `earliest_cr_line` (→ antigüedad crediticia) | Fecha de la primera línea de crédito del solicitante | Sí, transformada a antigüedad |
| `open_acc` | Número de líneas de crédito actualmente abiertas | Sí |
| `revol_util` | % de uso del crédito revolvente respecto al límite disponible | Sí |
| `total_acc` | Número total de líneas de crédito que ha tenido (abiertas o cerradas) | Sí |
| `issue_d` | Fecha de originación del préstamo (usada para el split temporal, no como feature) | Sí, uso auxiliar |

---

# 8. Riesgos de leakage

# 8. Riesgos de leakage

| Variable | Por qué es leakage | Medida de control |
|----------|--------------------|--------------------|
| `total_pymnt` | Se calcula a partir de los pagos ya realizados por el prestatario, información que no existe al momento de aprobar el préstamo | Excluida del conjunto de features |
| `total_rec_prncp` | Es el capital ya pagado; solo se conoce después del desembolso | Excluida del conjunto de features |
| `total_rec_int` | Son los intereses ya pagados; solo se conoce después del desembolso | Excluida del conjunto de features |
| `recoveries` | Solo tiene valor distinto de cero si el préstamo ya entró en default — filtra directamente el resultado que se quiere predecir | Excluida del conjunto de features |
| `last_pymnt_d` | Fecha del último pago realizado; no existe al momento de originar el préstamo | Excluida del conjunto de features |

---

# 9. Métricas (falta cambiar plantilla)

- **Métrica principal:** AUC-ROC — apropiada para el desbalance de clases y para comparar la capacidad de ranking de riesgo entre modelos
- **Métrica secundaria:** AUC-PR (Precision-Recall) — más informativa que accuracy dado que la clase de interés (default) es minoritaria

---

# 10. Plan de validación

- **Partición:** entrenamiento con préstamos originados en los primeros años 
  disponibles (2007–2015); validación y test con los periodos más 
  recientes (2016–2018), para simular el escenario real de producción.
- **Tipo de validación:** split temporal, con validación cruzada temporal 
  (ventanas móviles) sobre el set de entrenamiento para ajustar el modelo.
- **Regla:** el conjunto de test no se usa para tomar decisiones de modelado.
- **Semilla aleatoria:** 42

---

# 11. Modelo baseline(falta cambiar plantilla)

1. **Baseline trivial:** predecir siempre la clase mayoritaria (buen desempeño).
2. **Baseline logístico:** regresión logística con variables numéricas estandarizadas y categóricas codificadas (one-hot / target encoding según cardinalidad), sin balanceo de clases ni tuning.

Resultados esperados de la entrega previa: AUC-ROC y AUC-PR de ambos baselines, con una lectura corta de qué significan.

---

# 12. Riesgos técnicos(falta cambiar plantilla)

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Alta cardinalidad en categóricas (`purpose`, `sub_grade`, códigos postales) | Explosión de columnas al hacer one-hot | [target encoding, agrupar categorías poco frecuentes] |
| Volumen de datos grande | Notebook lento o memoria insuficiente | [muestreo, dtypes eficientes, procesamiento por lotes] |
| Desbalance de clases | Métricas engañosas si solo se mira accuracy | [usar AUC-ROC/AUC-PR, class weights o resampling] |
| Valores faltantes no aleatorios (missing not at random) en variables financieras | Sesgo si se imputa sin cuidado | [documentar patrón de faltantes, imputación justificada] |
| Cambios en políticas de originación de Lending Club a lo largo del tiempo | Drift entre periodos | [análisis de errores por año/segmento] |
| [otro] | | |

**Sesgos y limitaciones iniciales:** el dataset refleja únicamente solicitantes aprobados por Lending Club (no rechazados), lo que introduce sesgo de selección; los resultados no necesariamente generalizan a otras plataformas de crédito.

---

# 13. Plan de trabajo (falta cambiar plantilla)

| Semana | Actividad | Responsable |
|--------|-----------|-------------|
| 2–3 | Limpieza de datos, tratamiento de faltantes, definición final de variables (`02_limpieza_features.ipynb`, `src/features.py`) | |
| 4–5 | Pipeline de preprocesamiento reproducible + baseline (regresión logística) | |
| 6–8 | Modelos: logística regularizada, árboles/Random Forest, Gradient Boosting | |
| 9–10 | Búsqueda de hiperparámetros y validación temporal robusta | |
| 11–12 | Análisis de errores por segmento (grade, ingresos, propósito) e interpretabilidad (SHAP) | |
| 13–14 | Informe final (`reports/informe_final.md`) | |
| 15 | Presentación y revisión del README | |
| 16 | Entrega final | |

<!-- Ajusten las semanas según el cronograma real del curso. -->

---
├── reports/
│   └── figures/
└── outputs/
```
