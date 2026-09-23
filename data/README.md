# Datos

## Cómo obtener el dataset

**Fuente:** Lending Club Loan Data, mirror de Kaggle
`wordsforthewise/lending-club`
(<https://www.kaggle.com/datasets/wordsforthewise/lending-club>).

1. Crea una cuenta gratuita en Kaggle y acepta las reglas del dataset.
2. Descarga `accepted_2007_to_2018Q4.csv` (préstamos aprobados).
3. Coloca el archivo en esta carpeta:

   ```
   data/accepted_2007_to_2018Q4.csv
   ```

   Si usas otro nombre o una muestra, actualiza `RAW_FILENAME` en
   `src/config.py` o pasa la ruta con `--data`.

## Muestra de trabajo

El archivo completo tiene ~2.26M filas. Para la exploración inicial usamos
una muestra ajustando de 100 000 registros 
