"""
train.py
--------
Pipeline principal de entrenamiento del modelo de default (Lending Club).

Uso (desde la raíz del repositorio):

    python -m src.train --data data/accepted_2007_to_2018Q4.csv.gz
    python -m src.train --data ... --quick      # prueba rápida (10% de los datos)

Flujo (sin ninguna decisión tomada con el test):

  1. Carga + target + split temporal (src/data.py):
       train = préstamos emitidos hasta 2016-10, test out-of-time = 2016-11..2018-12.
  2. Features + winsorización con topes ajustados SOLO en train (src/features.py).
  3. Búsqueda de hiperparámetros por familia de modelos con TimeSeriesSplit
     (4 folds, siempre se entrena con el pasado y se valida con el futuro),
     sobre una submuestra del train ordenada por fecha.
  4. Selección de umbral: cada modelo se entrena con el 85% más antiguo del
     train y se elige el umbral que maximiza F1 en el 15% más reciente del
     train (bloque de selección). El test nunca participa.
  5. Reentrenamiento de cada modelo con TODO el train y evaluación única en el
     test out-of-time.
  6. El modelo final se elige por ROC-AUC medio de la validación cruzada
     temporal (no por su resultado en test).
  7. Se guardan: outputs/metrics.json, outputs/cv_results.csv,
     outputs/predictions_test.parquet, outputs/models/final_model.joblib y las
     figuras de evaluación en reports/figures/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src import evaluate as ev
from src.data import RANDOM_SEED, TARGET, load_dataset
from src.features import (
    CAT_COLS_FINAL,
    NUM_COLS_FINAL,
    build_features,
    build_preprocessor,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "reports" / "figures"

# Columnas que se guardan junto a las predicciones del test para el análisis
# de errores (notebooks/04_analisis_errores.ipynb).
CONTEXT_COLS = ["issue_dt", "loan_status", TARGET] + NUM_COLS_FINAL + CAT_COLS_FINAL


# ---------------------------------------------------------------------------
# Preparación de matrices
# ---------------------------------------------------------------------------
def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona las features finales y pasa las categóricas a `object`
    (los encoders de scikit-learn manejan mejor object + NaN que
    pandas.Categorical con categorías distintas entre train y test)."""
    X = df[NUM_COLS_FINAL + CAT_COLS_FINAL].copy()
    for c in CAT_COLS_FINAL:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


def build_tree_preprocessor() -> ColumnTransformer:
    """Preprocesador para árboles: imputación por mediana (numéricas) y
    codificación ordinal (categóricas). Los árboles no necesitan escalado ni
    one-hot; el orden de columnas es [numéricas..., categóricas...]."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_COLS_FINAL),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1, dtype=np.float32)),
        ]), CAT_COLS_FINAL),
    ])


# ---------------------------------------------------------------------------
# Definición de modelos y espacios de búsqueda
# ---------------------------------------------------------------------------
def model_specs() -> dict[str, dict]:
    """Cinco estrategias comparables (3 familias reales + 2 referencias).

    - dummy_prior:            piso trivial (predice la tasa base).
    - logreg_baseline:        el baseline de la entrega previa, sin tuning.
    - logreg_tuned:           familia lineal, con C y class_weight buscados.
    - random_forest:          familia bagging de árboles.
    - hist_gradient_boosting: familia boosting (categóricas nativas).
    """
    n_cat = len(CAT_COLS_FINAL)
    cat_mask = [False] * len(NUM_COLS_FINAL) + [True] * n_cat

    return {
        "dummy_prior": {
            "pipe": Pipeline([("pre", build_tree_preprocessor()),
                              ("clf", DummyClassifier(strategy="prior"))]),
            "search": None,
        },
        "logreg_baseline": {
            "pipe": Pipeline([("pre", build_preprocessor()),
                              ("clf", LogisticRegression(max_iter=2000,
                                                         random_state=RANDOM_SEED))]),
            "search": None,
        },
        "logreg_tuned": {
            "pipe": Pipeline([("pre", build_preprocessor()),
                              ("clf", LogisticRegression(max_iter=2000,
                                                         random_state=RANDOM_SEED))]),
            "search": ("grid", {
                "clf__C": [0.01, 0.1, 1.0],
                "clf__class_weight": [None, "balanced"],
            }),
        },
        "random_forest": {
            "pipe": Pipeline([("pre", build_tree_preprocessor()),
                              ("clf", RandomForestClassifier(
                                  n_estimators=200, max_samples=0.3, n_jobs=-1,
                                  random_state=RANDOM_SEED))]),
            "search": ("random", {
                "clf__max_depth": [10, 14, 18],
                "clf__min_samples_leaf": [20, 100, 300],
                "clf__max_features": ["sqrt", 0.33],
                "clf__class_weight": [None, "balanced_subsample"],
            }, 6),
        },
        "hist_gradient_boosting": {
            "pipe": Pipeline([("pre", build_tree_preprocessor()),
                              ("clf", HistGradientBoostingClassifier(
                                  categorical_features=cat_mask, max_iter=400,
                                  early_stopping=True, validation_fraction=0.1,
                                  n_iter_no_change=20, random_state=RANDOM_SEED))]),
            "search": ("random", {
                "clf__learning_rate": [0.03, 0.06, 0.1],
                "clf__max_leaf_nodes": [15, 31, 63],
                "clf__min_samples_leaf": [100, 300, 1000],
                "clf__l2_regularization": [0.0, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            }, 10),
        },
    }


# ---------------------------------------------------------------------------
# Etapas del pipeline
# ---------------------------------------------------------------------------
def tune_models(X: pd.DataFrame, y: pd.Series, n_splits: int = 4,
                verbose: bool = True) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Búsqueda de hiperparámetros con TimeSeriesSplit. X debe estar ordenado
    por fecha de emisión. Devuelve (pipelines con los mejores parámetros,
    tabla resumen por familia, todas las configuraciones probadas)."""
    cv = TimeSeriesSplit(n_splits=n_splits)
    best, rows, all_trials = {}, [], []

    for name, spec in model_specs().items():
        t0 = time.time()
        pipe, search = spec["pipe"], spec["search"]
        if search is None:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
            best[name] = pipe
            mean, std, params = scores.mean(), scores.std(), {}
            all_trials.append({"model": name, "params": {}, "cv_roc_auc_mean": mean,
                               "cv_roc_auc_std": std})
        else:
            if search[0] == "grid":
                s = GridSearchCV(pipe, search[1], cv=cv, scoring="roc_auc",
                                 refit=False)
            else:
                s = RandomizedSearchCV(pipe, search[1], n_iter=search[2], cv=cv,
                                       scoring="roc_auc", refit=False,
                                       random_state=RANDOM_SEED)
            s.fit(X, y)
            i = int(np.argmax(s.cv_results_["mean_test_score"]))
            params = s.cv_results_["params"][i]
            mean = s.cv_results_["mean_test_score"][i]
            std = s.cv_results_["std_test_score"][i]
            best[name] = clone(pipe).set_params(**params)
            for p, m, sd in zip(s.cv_results_["params"],
                                s.cv_results_["mean_test_score"],
                                s.cv_results_["std_test_score"]):
                all_trials.append({"model": name, "params": p,
                                   "cv_roc_auc_mean": m, "cv_roc_auc_std": sd})

        rows.append({"model": name, "cv_roc_auc_mean": float(mean),
                     "cv_roc_auc_std": float(std),
                     "best_params": {k.replace("clf__", ""): v
                                     for k, v in params.items()},
                     "search_seconds": round(time.time() - t0, 1)})
        if verbose:
            print(f"  [{name}] CV ROC-AUC = {mean:.4f} ± {std:.4f}  "
                  f"({time.time() - t0:.0f}s)  {rows[-1]['best_params']}")

    return best, pd.DataFrame(rows), all_trials


def select_thresholds(pipes: dict, train_feat: pd.DataFrame, search_idx: pd.Index,
                      sel_frac: float = 0.15, verbose: bool = True) -> dict:
    """Elige el umbral operativo de cada modelo en un bloque de selección que
    es el `sel_frac` más reciente del TRAIN (nunca el test).

    Se entrena con la submuestra de búsqueda anterior al corte (para que el
    costo sea el mismo que en la búsqueda) y se predice todo el bloque."""
    cutoff = train_feat["issue_dt"].quantile(1 - sel_frac)
    fit_rows = train_feat.loc[search_idx]
    fit_rows = fit_rows[fit_rows["issue_dt"] < cutoff]
    sel_rows = train_feat[train_feat["issue_dt"] >= cutoff]
    X_fit, y_fit = to_model_frame(fit_rows), fit_rows[TARGET]
    X_sel, y_sel = to_model_frame(sel_rows), sel_rows[TARGET]

    out = {}
    for name, pipe in pipes.items():
        if name == "dummy_prior":
            out[name] = {"threshold": 0.5, "selection_metrics": None}
            continue
        p = clone(pipe).fit(X_fit, y_fit).predict_proba(X_sel)[:, 1]
        thr = ev.best_f1_threshold(y_sel, p)
        out[name] = {"threshold": thr,
                     "selection_metrics": ev.compute_metrics(y_sel, p, thr)}
        if verbose:
            m = out[name]["selection_metrics"]
            print(f"  [{name}] umbral={thr:.3f}  AUC(sel)={m['roc_auc']:.4f}  "
                  f"F1(sel)={m['at_threshold']['f1']:.3f}")
    out["_block"] = {"cutoff": str(cutoff.date()), "n_fit": int(len(fit_rows)),
                     "n_selection": int(len(sel_rows)),
                     "selection_base_rate": float(y_sel.mean())}
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


def run(data_path: str, frac: float = 1.0, search_frac: float = 0.25,
        n_splits: int = 4, outputs: Path = OUTPUTS, figures: Path = FIGURES) -> dict:
    t_start = time.time()
    np.random.seed(RANDOM_SEED)
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "models").mkdir(exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    # 1-2. Datos y features ----------------------------------------------------
    print("1) Cargando datos y construyendo features...")
    train, test = load_dataset(data_path, frac=frac, seed=RANDOM_SEED)
    train_feat, thresholds = build_features(train, fit_thresholds=True)
    test_feat, _ = build_features(test, thresholds=thresholds)
    train_feat = train_feat.sort_values("issue_dt", kind="mergesort")
    test_feat = test_feat.sort_values("issue_dt", kind="mergesort")
    print(f"   train: {len(train_feat):,} ({train_feat['issue_dt'].min():%Y-%m} a "
          f"{train_feat['issue_dt'].max():%Y-%m}), default={train_feat[TARGET].mean():.3f}")
    print(f"   test:  {len(test_feat):,} ({test_feat['issue_dt'].min():%Y-%m} a "
          f"{test_feat['issue_dt'].max():%Y-%m}), default={test_feat[TARGET].mean():.3f}")

    # 3. Búsqueda de hiperparámetros (submuestra del train, ordenada) --------
    search_df = (train_feat.sample(frac=search_frac, random_state=RANDOM_SEED)
                 .sort_values("issue_dt", kind="mergesort"))
    print(f"2) Búsqueda de hiperparámetros: TimeSeriesSplit({n_splits}) sobre "
          f"{len(search_df):,} préstamos del train...")
    best_pipes, cv_table, all_trials = tune_models(
        to_model_frame(search_df), search_df[TARGET], n_splits=n_splits)

    # 4. Umbral operativo (bloque de selección = 15% más reciente del train) --
    print("3) Selección de umbral en el bloque más reciente del train...")
    thr_info = select_thresholds(best_pipes, train_feat, search_df.index)

    # 5. Reentrenamiento con todo el train + evaluación única en test --------
    print("4) Reentrenando con todo el train y evaluando en test out-of-time...")
    X_tr, y_tr = to_model_frame(train_feat), train_feat[TARGET]
    X_te, y_te = to_model_frame(test_feat), test_feat[TARGET].to_numpy()
    fitted, scores, test_metrics, fit_secs = {}, {}, {}, {}
    for name, pipe in best_pipes.items():
        t0 = time.time()
        fitted[name] = clone(pipe).fit(X_tr, y_tr)
        fit_secs[name] = round(time.time() - t0, 1)
        scores[name] = fitted[name].predict_proba(X_te)[:, 1]
        thr = thr_info[name]["threshold"]
        test_metrics[name] = ev.compute_metrics(y_te, scores[name], thr)
        test_metrics[name]["at_0.5"] = ev.threshold_metrics(y_te, scores[name], 0.5)
        m = test_metrics[name]
        print(f"  [{name}] AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
              f"KS={m['ks']:.4f} Brier={m['brier']:.4f} "
              f"F1@{thr:.3f}={m['at_threshold']['f1']:.3f} ({fit_secs[name]}s)")

    # 6. Modelo final: mejor ROC-AUC en CV temporal (no en test) ------------
    final_name = cv_table.sort_values("cv_roc_auc_mean", ascending=False)["model"].iloc[0]
    print(f"5) Modelo final (por CV): {final_name}")
    joblib.dump({"model": fitted[final_name], "name": final_name,
                 "threshold": thr_info[final_name]["threshold"],
                 "winsor_thresholds": thresholds,
                 "num_cols": NUM_COLS_FINAL, "cat_cols": CAT_COLS_FINAL},
                outputs / "models" / "final_model.joblib", compress=3)
    # Baseline guardado para comparar interpretabilidad en el notebook 04
    joblib.dump({"model": fitted["logreg_tuned"], "name": "logreg_tuned",
                 "threshold": thr_info["logreg_tuned"]["threshold"]},
                outputs / "models" / "logreg_tuned.joblib", compress=3)

    # 7. Guardado de resultados ---------------------------------------------
    preds = test_feat[CONTEXT_COLS].copy()
    for c in CAT_COLS_FINAL:
        preds[c] = preds[c].astype(str)
    for name, s in scores.items():
        preds[f"p_{name}"] = s.astype(np.float32)
    preds.to_parquet(outputs / "predictions_test.parquet", index=False)

    cv_table.to_csv(outputs / "cv_results.csv", index=False)
    pd.DataFrame([{**t, "params": json.dumps(_jsonable(t["params"]))}
                  for t in all_trials]).to_csv(outputs / "cv_all_trials.csv", index=False)

    # Figuras
    ev.plot_cv_results(cv_table, figures / "20_cv_roc_auc.png")
    ev.plot_roc_curves(y_te, scores, figures / "21_roc_curves_test.png")
    ev.plot_pr_curves(y_te, scores, figures / "22_pr_curves_test.png")
    ev.plot_metric_comparison(test_metrics, figures / "23_metric_comparison_test.png")
    ev.plot_calibration(y_te, scores, figures / "24_calibration_test.png")
    ev.plot_confusion(y_te, scores[final_name], thr_info[final_name]["threshold"],
                      f"Matriz de confusión — {ev.MODEL_LABELS[final_name]} (test)",
                      figures / "25_confusion_final_test.png")

    metrics = {
        "config": {
            "data_path": str(data_path), "frac": frac, "search_frac": search_frac,
            "cv": f"TimeSeriesSplit(n_splits={n_splits})", "random_seed": RANDOM_SEED,
            "threshold_rule": "max F1 en el 15% más reciente del train",
            "final_model_rule": "mayor ROC-AUC medio en CV temporal",
        },
        "data": {
            "n_train": int(len(train_feat)), "n_test": int(len(test_feat)),
            "train_period": [f"{train_feat['issue_dt'].min():%Y-%m}",
                             f"{train_feat['issue_dt'].max():%Y-%m}"],
            "test_period": [f"{test_feat['issue_dt'].min():%Y-%m}",
                            f"{test_feat['issue_dt'].max():%Y-%m}"],
            "train_default_rate": float(train_feat[TARGET].mean()),
            "test_default_rate": float(y_te.mean()),
            "winsor_thresholds": thresholds,
            "n_features_raw": len(NUM_COLS_FINAL) + len(CAT_COLS_FINAL),
        },
        "cv": cv_table.set_index("model").to_dict(orient="index"),
        "threshold_selection": thr_info,
        "test": test_metrics,
        "fit_seconds_full_train": fit_secs,
        "final_model": final_name,
        "runtime_minutes": round((time.time() - t_start) / 60, 1),
    }
    with open(outputs / "metrics.json", "w") as f:
        json.dump(_jsonable(metrics), f, indent=2, ensure_ascii=False)
    print(f"Listo en {metrics['runtime_minutes']} min. Resultados en {outputs}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "accepted_2007_to_2018Q4.csv.gz"),
                    help="Ruta al CSV (o .csv.gz) de Lending Club.")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="Fracción de filas a cargar (1.0 = dataset completo).")
    ap.add_argument("--search-frac", type=float, default=0.25,
                    help="Fracción del train usada en la búsqueda de hiperparámetros.")
    ap.add_argument("--quick", action="store_true",
                    help="Prueba rápida: frac=0.05; escribe en outputs/quick/.")
    args = ap.parse_args()

    if args.quick:
        run(args.data, frac=0.05, search_frac=0.5,
            outputs=OUTPUTS / "quick", figures=OUTPUTS / "quick" / "figures")
    else:
        run(args.data, frac=args.frac, search_frac=args.search_frac)


if __name__ == "__main__":
    main()
