"""
eda.py
------
Regenera las figuras del análisis exploratorio (las mismas vistas que
notebooks/01_exploracion_inicial.ipynb) sobre el dataset COMPLETO y las guarda
en reports/figures/. Se usan en el informe final.

Uso (desde la raíz del repositorio):

    python -m src.eda --data data/accepted_2007_to_2018Q4.csv.gz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import RANDOM_SEED, TARGET, build_target, load_raw, parse_dates, temporal_split

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
sns.set_theme(style="whitegrid")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=120)
    plt.close(fig)


def make_figures(data_path: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    raw = load_raw(data_path)
    df = parse_dates(build_target(raw))
    train, test = temporal_split(df)
    # Submuestra solo para gráficos de densidad/boxplot (más rápidos y legibles);
    # las tasas y conteos usan todos los datos.
    sample = df.sample(n=min(200_000, len(df)), random_state=RANDOM_SEED)

    # 01 Faltantes
    miss = (df.drop(columns=[TARGET, "issue_dt", "earliest_cr_line_dt"])
            .isna().mean().mul(100).sort_values(ascending=False))
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=miss.values, y=miss.index, color="#4C72B0", ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f%%", fontsize=8, padding=2)
    ax.set_xlabel("% de valores faltantes")
    ax.set_title(f"Valores faltantes por variable (n={len(df):,})")
    _save(fig, "01_missing_values.png")

    # 02 Distribución del objetivo
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(["Buen desempeño (0)", "Mal desempeño (1)"], counts.values,
           color=["#55A868", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Número de préstamos")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_title("Distribución de la variable objetivo")
    _save(fig, "02_target_distribution.png")

    # 03 Default por grade
    g = df.groupby("grade", observed=True)[TARGET].mean().sort_index() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    g.plot.bar(ax=ax, color="#8e44ad")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=8)
    ax.set_title("Tasa de default por grade de Lending Club")
    ax.set_ylabel("% default")
    ax.set_xlabel("grade")
    _save(fig, "03_default_by_grade.png")

    # 04 int_rate según resultado
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, lbl, col in [(0, "Pagado", "#27ae60"), (1, "Default", "#c0392b")]:
        sns.kdeplot(sample.loc[sample[TARGET] == k, "int_rate"].dropna(), ax=ax,
                    label=lbl, fill=True, alpha=.3, color=col)
    ax.set_title("Tasa de interés (int_rate) según resultado del préstamo")
    ax.set_xlabel("int_rate (%)")
    ax.legend()
    _save(fig, "04_intrate_by_target.png")

    # 05 loan_amnt según resultado
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=sample, x=TARGET, y="loan_amnt", ax=ax)
    ax.set_title("Monto solicitado (loan_amnt) por resultado")
    ax.set_xlabel("mal_desempeno (0 = bueno, 1 = malo)")
    _save(fig, "05_loanamnt_by_target.png")

    # 06 Default por propósito
    p = df.groupby("purpose", observed=True)[TARGET].agg(["mean", "size"])
    p = p[p["size"] >= 1000]["mean"].sort_values() * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    p.plot.barh(ax=ax, color="#2980b9")
    ax.set_title("Tasa de default por propósito (purpose, n ≥ 1000)")
    ax.set_xlabel("% default")
    _save(fig, "06_default_by_purpose.png")

    # 07 Correlaciones
    num = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti",
           "fico_range_low", "fico_range_high", "open_acc", "revol_util", "total_acc"]
    corr = df[num + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax,
                cbar_kws={"shrink": .7})
    ax.set_title("Correlación de features de originación (y target)")
    _save(fig, "07_corr_heatmap.png")

    # 08 Deriva temporal + volumen
    q = df.assign(_q=df["issue_dt"].dt.to_period("Q").dt.to_timestamp())
    ts = q.groupby("_q")[TARGET].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax2 = ax.twinx()
    ax2.bar(ts.index, ts["size"], width=60, color="#bdc3c7", alpha=.6,
            label="N.º préstamos resueltos")
    ax.plot(ts.index, ts["mean"] * 100, color="#16a085", lw=2, marker="o", ms=3,
            label="% default")
    ax.axvline(test["issue_dt"].min(), color="#C44E52", ls="--", lw=1.5)
    ax.text(test["issue_dt"].min(), ax.get_ylim()[1] * 0.95, "  inicio test", color="#C44E52")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylabel("% default")
    ax2.set_ylabel("Préstamos con resultado observado")
    ax2.grid(False)
    ax.set_title("Tasa de default y volumen por trimestre de emisión")
    _save(fig, "08_default_over_time.png")

    # 09 Outliers
    cols = ["annual_inc", "dti", "revol_util", "loan_amnt"]
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 4))
    for ax, c in zip(axes, cols):
        sns.boxplot(y=sample[c], ax=ax)
        ax.set_title(c)
    fig.suptitle("Boxplots para detección de outliers (antes de winsorizar)")
    _save(fig, "09_outliers.png")

    # 10 Partición temporal
    m = df.assign(_m=df["issue_dt"].dt.to_period("M").dt.to_timestamp(),
                  split=np.where(df["issue_dt"] < test["issue_dt"].min(), "train", "test"))
    vol = m.groupby(["_m", "split"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(vol.index, vol.get("train", 0), width=25, color="#4C72B0", label="train")
    ax.bar(vol.index, vol.get("test", 0), width=25, color="#C44E52", label="test out-of-time")
    ax.set_ylabel("Préstamos por mes")
    ax.set_title(f"Partición temporal: train {len(train):,} / test {len(test):,}")
    ax.legend()
    _save(fig, "10_temporal_split.png")
    print(f"Figuras EDA guardadas en {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "accepted_2007_to_2018Q4.csv.gz"))
    make_figures(ap.parse_args().data)
