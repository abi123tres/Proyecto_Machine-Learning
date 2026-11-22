"""
evaluate.py
-----------
Métricas, selección de umbral, figuras de evaluación y análisis de errores
por segmentos para el modelo de default (Lending Club).

Métricas (ver proposal.md, sección 9):
  - Principal:   ROC-AUC (capacidad de ordenar por riesgo, robusta al desbalance).
  - Secundarias: PR-AUC (average precision) y estadístico KS.
  - Complementarias: Brier (calibración) y precision / recall / F1 a un
    umbral operativo. Nunca se reporta accuracy sola.

Todas las funciones reciben arrays ya predichos: este módulo no entrena nada.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: el script corre también en servidores / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sns.set_theme(style="whitegrid")

# Paleta fija por modelo para que todas las figuras sean consistentes
MODEL_COLORS = {
    "dummy_prior": "#9e9e9e",
    "logreg_baseline": "#4C72B0",
    "logreg_tuned": "#64B5CD",
    "random_forest": "#55A868",
    "hist_gradient_boosting": "#C44E52",
}
MODEL_LABELS = {
    "dummy_prior": "Dummy (tasa base)",
    "logreg_baseline": "Reg. logística baseline",
    "logreg_tuned": "Reg. logística ajustada",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
}


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def ks_statistic(y_true, y_score) -> float:
    """KS = máx. distancia entre las CDF de score de buenos y malos
    (equivale a max(TPR - FPR) sobre la curva ROC)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def threshold_metrics(y_true, y_score, threshold: float) -> dict:
    """Precision, recall, F1 y tasa de aprobación a un umbral dado."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "flagged_rate": float(y_pred.mean()),  # % de préstamos marcados como riesgo
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Bloque completo de métricas para un modelo sobre un conjunto."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "base_rate": float(y_true.mean()),
        "n": int(len(y_true)),
        "at_threshold": threshold_metrics(y_true, y_score, threshold),
    }


def best_f1_threshold(y_true, y_score) -> float:
    """Umbral que maximiza F1. Debe calcularse sobre un bloque de
    SELECCIÓN (nunca sobre el test final)."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    # precision_recall_curve devuelve len(thr) = len(prec) - 1
    return float(thr[np.argmax(f1[:-1])])


# ---------------------------------------------------------------------------
# Figuras de comparación de modelos
# ---------------------------------------------------------------------------
def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def plot_roc_curves(y_true, scores: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        auc = roc_auc_score(y_true, s)
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name), lw=2,
                label=f"{_label(name)} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR / recall)")
    ax.set_title("Curvas ROC — test out-of-time")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pr_curves(y_true, scores: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(y_true, s)
        ap = average_precision_score(y_true, s)
        ax.plot(rec, prec, color=MODEL_COLORS.get(name), lw=2,
                label=f"{_label(name)} (AP={ap:.3f})")
    ax.axhline(np.mean(y_true), ls="--", color="grey", lw=1,
               label=f"Tasa base ({np.mean(y_true):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall — test out-of-time")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_metric_comparison(metrics: dict[str, dict], path: Path) -> None:
    """Barras de ROC-AUC, PR-AUC y KS por modelo en el test."""
    rows = [
        {"modelo": _label(m), "métrica": k.upper().replace("_", "-"), "valor": v[k]}
        for m, v in metrics.items() for k in ("roc_auc", "pr_auc", "ks")
    ]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=df, x="métrica", y="valor", hue="modelo", ax=ax,
                palette=[MODEL_COLORS.get(m) for m in metrics])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=7, padding=1)
    ax.set_ylim(0, 1)
    ax.set_title("Comparación de modelos — test out-of-time")
    ax.set_xlabel("")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_calibration(y_true, scores: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, s in scores.items():
        if name == "dummy_prior":
            continue
        frac_pos, mean_pred = calibration_curve(y_true, s, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", color=MODEL_COLORS.get(name),
                label=_label(name))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Calibración perfecta")
    ax.set_xlabel("Probabilidad predicha (media por decil)")
    ax.set_ylabel("Tasa de default observada")
    ax.set_title("Curva de calibración — test out-of-time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(y_true, y_score, threshold: float, title: str, path: Path) -> None:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.3))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred. bueno", "Pred. malo"],
                yticklabels=["Real bueno", "Real malo"])
    ax.set_title(f"{title}\n(umbral = {threshold:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cv_results(cv_table: pd.DataFrame, path: Path) -> None:
    """ROC-AUC medio ± desviación de la validación cruzada temporal."""
    t = cv_table.sort_values("cv_roc_auc_mean")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh([_label(m) for m in t["model"]], t["cv_roc_auc_mean"],
            xerr=t["cv_roc_auc_std"], color=[MODEL_COLORS.get(m) for m in t["model"]],
            capsize=4)
    for i, v in enumerate(t["cv_roc_auc_mean"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.5, max(0.8, t["cv_roc_auc_mean"].max() + 0.05))
    ax.set_xlabel("ROC-AUC medio (TimeSeriesSplit sobre train)")
    ax.set_title("Mejor configuración por familia — validación cruzada temporal")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Análisis de errores por segmentos
# ---------------------------------------------------------------------------
def segment_report(df: pd.DataFrame, segment: str, y_col: str, score_col: str,
                   threshold: float, min_n: int = 500) -> pd.DataFrame:
    """Métricas por segmento: n, tasa real, probabilidad media predicha,
    ROC-AUC dentro del segmento, recall y precision al umbral operativo.
    Los segmentos con menos de `min_n` préstamos se omiten (métricas ruidosas)."""
    rows = []
    for seg, g in df.groupby(segment, observed=True):
        if len(g) < min_n:
            continue
        y = g[y_col].to_numpy()
        s = g[score_col].to_numpy()
        pred = (s >= threshold).astype(int)
        rows.append({
            segment: seg,
            "n": len(g),
            "tasa_default_real": y.mean(),
            "prob_media_predicha": s.mean(),
            "roc_auc": roc_auc_score(y, s) if 0 < y.mean() < 1 else np.nan,
            "recall": recall_score(y, pred, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0),
            "tasa_marcados": pred.mean(),
        })
    out = pd.DataFrame(rows)
    out["sesgo_calibracion"] = out["prob_media_predicha"] - out["tasa_default_real"]
    return out


def plot_segment(seg_df: pd.DataFrame, segment: str, path: Path,
                 title: str | None = None) -> None:
    """Dos paneles: (a) tasa real vs. probabilidad media predicha,
    (b) ROC-AUC dentro de cada segmento."""
    d = seg_df.copy()
    d[segment] = d[segment].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    long = d.melt(id_vars=segment, value_vars=["tasa_default_real", "prob_media_predicha"],
                  var_name="serie", value_name="valor")
    long["serie"] = long["serie"].map({"tasa_default_real": "Tasa real",
                                       "prob_media_predicha": "Prob. media predicha"})
    sns.barplot(data=long, x=segment, y="valor", hue="serie", ax=axes[0],
                palette=["#4C72B0", "#DD8452"])
    axes[0].set_ylabel("Tasa / probabilidad")
    axes[0].set_title("Calibración por segmento")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=8)

    sns.barplot(data=d, x=segment, y="roc_auc", ax=axes[1], color="#55A868")
    axes[1].axhline(0.5, ls="--", color="grey", lw=1)
    axes[1].set_ylim(0.4, 0.85)
    axes[1].set_ylabel("ROC-AUC dentro del segmento")
    axes[1].set_title("Capacidad de ordenar dentro del segmento")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle(title or f"Errores por segmento: {segment}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
