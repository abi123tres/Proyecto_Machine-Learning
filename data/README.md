# Datos

## Cómo obtener el dataset

**Fuente:** Lending Club Loan Data, mirror de Kaggle
`wordsforthewise/lending-club`
(<https://www.kaggle.com/datasets/wordsforthewise/lending-club>).

1. Crea una cuenta gratuita en Kaggle y acepta las reglas del dataset.
2. Descarga `accepted_2007_to_2018Q4.csv` (préstamos aprobados, ~1.6 GB).
3. Coloca el archivo donde el notebook lo pueda leer:
   - **En Google Colab:** súbelo a la misma carpeta que el notebook.
   - **En local:** colócalo en esta carpeta (`data/accepted_2007_to_2018Q4.csv`) y en
     el notebook cambia `DATA_PATH = "../data/accepted_2007_to_2018Q4.csv"`.

> Verifica que el archivo esté completo (~2.26 millones de filas). Si se abre y guarda
> en Excel, se trunca a ~1 millón de filas y se pierden años enteros.

## Muestra de trabajo

El archivo completo tiene 2 260 701 filas y 151 columnas. Para la exploración inicial
se usa una **muestra aleatoria reproducible del 8%** (semilla 42), leída saltando filas
al azar y cargando solo las 27 columnas de originación necesarias:

- Préstamos cargados: **163 818** (de 2007 a 2018).
- Préstamos con resultado observado (sin `Current`, `In Grace Period`,
  `Late (16-30 days)`): **102 159**.
