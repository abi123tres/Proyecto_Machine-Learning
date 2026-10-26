"""
features.py
-----------
Feature engineering y preprocesamiento para el modelo de default
(Lending Club). Todas las decisiones siguen el criterio acordado:

  - fico_range_low / fico_range_high (corr. 1.00) -> se combinan en fico_avg.
  - installment (corr. 0.95 con loan_amnt) -> se descarta; loan_amnt es la
    variable "cruda" y más interpretable, installment depende de int_rate y
    term así que es casi redundante.
  - Outliers: thresholds (percentil 99) calculados SOLO con train y
    reaplicados a validación, para no filtrar información del futuro.
  - emp_length: los faltantes se tratan como categoría "Desconocido", no se
    imputan con la moda, porque no reportar antigüedad laboral puede ser
    informativo del riesgo.

LEAKAGE_COLUMNS se documenta de forma explícita (aunque nunca se cargan,
porque data.py ya usa una lista blanca) para dejar constancia, como pide la
rúbrica, de qué se excluyó y por qué.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columnas generadas DESPUÉS de otorgar el préstamo. Nunca se cargan (ver
# data.COLUMNAS_UTILES), se documentan aquí solo como referencia explícita.
LEAKAGE_COLUMNS = [
    # Pagos y saldos vivos
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "out_prncp", "out_prncp_inv",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    # Recuperaciones tras default
    "recoveries", "collection_recovery_fee",
    # Buró actualizado durante la vida del préstamo
    "last_fico_range_high", "last_fico_range_low", "last_credit_pull_d",
    # Acuerdos de liquidación / hardship
    "debt_settlement_flag", "settlement_status", "settlement_date",
    "settlement_amount", "hardship_flag", "hardship_type", "hardship_status",
    # Fondeo (posterior a la aprobación)
    "funded_amnt", "funded_amnt_inv",
]

NUM_COLS_FINAL = [
    "loan_amnt", "int_rate", "annual_inc", "dti",
    "fico_avg", "open_acc", "revol_util", "revol_bal", "total_acc",
    "credit_hist_months", "delinq_2yrs", "inq_last_6mths",
    "pub_rec", "pub_rec_bankruptcies",
]
CAT_COLS_FINAL = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "purpose", "verification_status", "application_type",
    "initial_list_status",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """fico_avg (combina fico_range_low/high) y credit_hist_months
    (antigüedad crediticia en meses, a partir de earliest_cr_line_dt)."""
    out = df.copy()
    out["fico_avg"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    out["credit_hist_months"] = (
        (out["issue_dt"] - out["earliest_cr_line_dt"]).dt.days / 30.44
    )
    out = out.drop(columns=["fico_range_low", "fico_range_high", "installment"],
                   errors="ignore")
    return out


def fit_outlier_thresholds(train: pd.DataFrame) -> dict:
    """Calcula los topes de winsorización SOLO con train (percentil 99).
    Para dti, se usa el mínimo entre el percentil 99 y 60 (tope lógico:
    un DTI > 60 suele ser un error de carga, no un caso real)."""
    thresholds = {
        "annual_inc_p99": train["annual_inc"].quantile(0.99),
        "dti_cap": min(train["dti"].quantile(0.99), 60.0),
    }
    return thresholds


def apply_outlier_caps(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Aplica los topes calculados en train (recorta, no elimina filas)."""
    out = df.copy()
    out["annual_inc"] = out["annual_inc"].clip(upper=thresholds["annual_inc_p99"])
    out["dti"] = out["dti"].clip(upper=thresholds["dti_cap"])
    return out


def handle_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    """emp_length faltante -> categoría propia 'Desconocido', en vez de
    imputar con la moda (ver justificación en el docstring del módulo).

    Si la columna llegó como dtype 'category' (data.py la carga así para
    ahorrar memoria con el dataset completo), hay que agregar "Desconocido"
    como categoría válida ANTES de rellenar: pandas no permite asignar un
    valor que no es una categoría existente en una columna Categorical."""
    out = df.copy()
    if isinstance(out["emp_length"].dtype, pd.CategoricalDtype):
        out["emp_length"] = out["emp_length"].cat.add_categories(["Desconocido"])
    out["emp_length"] = out["emp_length"].fillna("Desconocido")
    return out


def build_features(df: pd.DataFrame, thresholds: dict | None = None,
                    fit_thresholds: bool = False) -> tuple[pd.DataFrame, dict]:
    """Pipeline de features completo para un split (train o valid).

    Uso:
        train_feat, thresholds = build_features(train, fit_thresholds=True)
        valid_feat, _ = build_features(valid, thresholds=thresholds)
    """
    out = add_derived_features(df)
    out = handle_emp_length(out)

    if fit_thresholds:
        thresholds = fit_outlier_thresholds(out)
    if thresholds is None:
        raise ValueError("Debes pasar thresholds, o fit_thresholds=True si es train.")

    out = apply_outlier_caps(out, thresholds)
    return out, thresholds


def build_preprocessor(num_cols: list[str] = NUM_COLS_FINAL,
                        cat_cols: list[str] = CAT_COLS_FINAL) -> ColumnTransformer:
    """Pipeline de preprocesamiento: imputación + escalado (numéricas),
    imputación + one-hot (categóricas). Se ajusta SOLO con fit(train)."""
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])
