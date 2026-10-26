"""
data.py
-------
Carga del dataset Lending Club, construcción de la variable objetivo
(`mal_desempeno`) y partición temporal train/validación.

Reusa exactamente las decisiones ya tomadas y documentadas en
notebooks/01_exploracion_inicial.ipynb y proposal.md:
  - Muestra aleatoria reproducible del 8% (semilla 42).
  - Lista blanca de columnas de originación (sin leakage).
  - Definición de "mal desempeño" (default) a partir de loan_status.
  - Partición temporal (out-of-time) por issue_d.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd

RANDOM_SEED = 42
TARGET = "mal_desempeno"

# Columnas de originación: todo lo que está disponible ANTES de otorgar el
# préstamo. No incluye ninguna columna de LEAKAGE_COLUMNS (ver features.py).
COLUMNAS_UTILES = [
    "loan_status", "issue_d",
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "purpose", "application_type", "initial_list_status",
    "emp_length", "home_ownership", "annual_inc", "verification_status", "dti",
    "fico_range_low", "fico_range_high", "earliest_cr_line", "open_acc",
    "revol_util", "revol_bal", "total_acc", "delinq_2yrs", "inq_last_6mths",
    "pub_rec", "pub_rec_bankruptcies",
]

# Estados de loan_status con resultado observado (ver proposal.md, sección 5)
ESTADOS_MALOS = [
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
]
ESTADOS_BUENOS = [
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
]


def load_raw(data_path: str | Path, frac: float = 1.0,
             seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Carga el CSV original, usando solo las columnas de originación
    (sin leakage) para no cargar en memoria las ~125 columnas que no se usan.

    frac=1.0 (por defecto) carga TODAS las filas, como pide la entrega final.
    Para pruebas rápidas puedes pasar frac<1.0 y se aplica el mismo muestreo
    reproducible (semilla fija) que se usó en la entrega previa — pero el
    resultado final de este proyecto debe correrse con frac=1.0."""
    df = pd.read_csv(
        data_path,
        usecols=COLUMNAS_UTILES,
        low_memory=False,
        dtype={
            "term": "category", "grade": "category", "sub_grade": "category",
            "emp_length": "category", "home_ownership": "category",
            "purpose": "category", "verification_status": "category",
            "application_type": "category", "initial_list_status": "category",
        },
        skiprows=(
            None if frac >= 1.0
            else _make_sampler(frac, seed)
        ),
    )
    df = df.dropna(subset=["loan_status"])
    return df


def _make_sampler(frac: float, seed: int):
    """Genera la misma función de skiprows reproducible usada antes,
    solo para cuando frac < 1.0 (pruebas rápidas, no la entrega final)."""
    random.seed(seed)
    return lambda i: i > 0 and random.random() > frac


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra a préstamos con resultado observado y construye `mal_desempeno`.
    Descarta Current / In Grace Period / Late (16-30 days) / Issued, cuyo
    resultado aún no es observable."""
    estados_validos = ESTADOS_MALOS + ESTADOS_BUENOS
    out = df[df["loan_status"].isin(estados_validos)].copy()
    out[TARGET] = out["loan_status"].isin(ESTADOS_MALOS).astype(int)
    return out


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte issue_d y earliest_cr_line (texto 'Dec-2015') a datetime."""
    out = df.copy()
    out["issue_dt"] = pd.to_datetime(out["issue_d"], format="%b-%Y", errors="coerce")
    out["earliest_cr_line_dt"] = pd.to_datetime(
        out["earliest_cr_line"], format="%b-%Y", errors="coerce"
    )
    out = out.dropna(subset=["issue_dt"])
    return out


def temporal_split(df: pd.DataFrame, date_col: str = "issue_dt",
                    train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición out-of-time: train = préstamos más antiguos,
    validación = el (1 - train_frac) más reciente. Evita leakage temporal."""
    data = df.sort_values(date_col)
    cutoff = data[date_col].iloc[int(len(data) * train_frac)]
    train = data[data[date_col] < cutoff].copy()
    valid = data[data[date_col] >= cutoff].copy()
    assert len(train) > 0 and len(valid) > 0, (
        "La muestra no cubre suficientes meses para un split temporal"
    )
    return train, valid


def load_dataset(data_path: str | Path, frac: float = 1.0,
                  seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completo de carga: raw -> target -> fechas -> split temporal.
    Devuelve (train, valid) listos para pasar a features.py."""
    df = load_raw(data_path, frac=frac, seed=seed)
    df = build_target(df)
    df = parse_dates(df)
    train, valid = temporal_split(df)
    return train, valid
