"""
================================================================================
Sprint 5 / HU6 — XGBoost mejorado con tres ajustes técnicos
================================================================================

AJUSTES RESPECTO A LA VERSIÓN ANTERIOR:
----------------------------------------

AJUSTE 1 — Dos tipos de target:
  a) confirmed_cases: predecimos los casos de la SEMANA ACTUAL, igual que en
     el Sprint 4 con Random Forest. Permite comparación directa entre modelos.
  b) target_h1..h4: predecimos los casos de semanas FUTURAS (1 a 4 semanas
     adelante). Permite evaluar el horizonte predictivo del sistema de alertas.

AJUSTE 2 — Transformación logarítmica del target:
  La distribución de casos es muy sesgada: la mayoría de semanas tienen 0-5
  casos, pero durante el brote puede haber 1.391. XGBoost tiene dificultad para
  aprender esta distribución asimétrica.
  Solución: aplicar log(1 + casos) al target antes de entrenar, y revertir la
  transformación (exp(pred) - 1) para obtener predicciones en escala real.
  Esto comprime los valores extremos y le da más peso a las semanas moderadas.

  Early stopping: detiene el entrenamiento automáticamente cuando el error en
  validación cruzada deja de mejorar, evitando sobreajuste.

AJUSTE 3 — Features de vecindad espacial:
  Agrega el promedio de casos de las comunas vecinas (lag 1 semana) como
  variable predictora. Captura la dispersión geográfica del dengue.
  La matriz de vecindad fue verificada contra el mapa oficial de CABA (2005).
  Genera: casos_vecinas_lag1, incidencia_vecinas_lag1 (calculadas en lags.py)

PREREQUISITO:
  Ejecutar src/features/lags.py ACTUALIZADO antes de este script.
  Ese script genera casos_vecinas_lag1, target_h1..h4 y los splits.

PIPELINE:
  Paso 1 → Cargar datos y verificar nuevas features
  Paso 2 → XGBoost sobre confirmed_cases (comparación directa con Sprint 4)
  Paso 3 → XGBoost sobre target_h1..h4 (predicción multi-horizonte)
  Paso 4 → Tabla comparativa Sprint 4 vs Sprint 5
  Paso 5 → Importancia de features por horizonte
  Paso 6 → Curva de degradación por horizonte
  Paso 7 → Guardar modelos y resultados
"""

import logging
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/saved")
FIGURES_DIR   = Path("reports/figures")

TRAIN_FILE = PROCESSED_DIR / "train.parquet"
VAL_FILE   = PROCESSED_DIR / "validation.parquet"
TEST_FILE  = PROCESSED_DIR / "test.parquet"

# Horizontes de predicción futura
HORIZONTES = [1, 2, 3, 4]

# =============================================================================
# FEATURES
# Incluye las nuevas features de vecindad (casos_vecinas_lag1) generadas
# en lags.py. El resto es igual al Sprint 4 + Sprint 5 anterior.
# =============================================================================

FEATURES_XGB = (
    # Historial de casos de la propia comuna
    ["cases_lag1", "cases_lag2", "cases_lag3", "cases_lag4",
     "incidencia_lag1", "incidencia_lag2", "incidencia_lag3", "incidencia_lag4"] +
    # AJUSTE 3: casos de comunas vecinas (promedio, lag 1 semana)
    # Captura la dinámica de dispersión espacial del dengue
    ["casos_vecinas_lag1", "incidencia_vecinas_lag1"] +
    # Variables climáticas actuales
    ["temp_mean", "precipitation", "humidity_mean", "heat_index_mean",
     "temp_mean_anomaly", "precipitation_anomaly", "humidity_mean_anomaly"] +
    # Variables climáticas con lags (1-4 semanas)
    [f"temp_mean_lag{i}"             for i in range(1, 5)] +
    [f"precipitation_lag{i}"         for i in range(1, 5)] +
    [f"humidity_mean_lag{i}"         for i in range(1, 5)] +
    [f"heat_index_mean_lag{i}"       for i in range(1, 5)] +
    [f"temp_mean_anomaly_lag{i}"     for i in range(1, 5)] +
    [f"precipitation_anomaly_lag{i}" for i in range(1, 5)] +
    [f"humidity_mean_anomaly_lag{i}" for i in range(1, 5)] +
    # Estacionalidad
    ["semana_sin", "semana_cos", "is_epidemic_season", "mes_aprox"] +
    # Características espaciales de la comuna
    ["comuna_id", "es_comuna_1", "poblacion"]
)

# Hiperparámetros a explorar — 72 combinaciones × 3 folds
PARAM_GRID = {
    "n_estimators":     [100, 300, 500],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "min_child_weight": [1, 5],
}

# Referencia Sprint 4 (en escala original de casos, sin log)
SPRINT4_VAL = {
    "Persistencia (lag 1)": {"MAE": 16.81, "RMSE": 43.10,  "R2": 0.932},
    "Media histórica":      {"MAE": 50.62, "RMSE": 138.82, "R2": 0.300},
    "Ridge climático":      {"MAE": 71.25, "RMSE": 168.80, "R2": -0.035},
    "Random Forest":        {"MAE": 24.93, "RMSE": 103.13, "R2": 0.614},
}

COLOR_XGB  = "#E74C3C"
COLOR_RF   = "#3498DB"
COLOR_PERS = "#F39C12"


# =============================================================================
# PASO 1: CARGA DE DATOS
# =============================================================================

def cargar_splits():
    logger.info("--- PASO 1: Cargando datos ---")

    for f in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not f.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {f}\n"
                "Ejecutá src/features/lags.py actualizado primero."
            )

    df_train = pd.read_parquet(TRAIN_FILE)
    df_val   = pd.read_parquet(VAL_FILE)
    df_test  = pd.read_parquet(TEST_FILE)

    logger.info("  Entrenamiento: %d filas | Validación: %d | Test: %d",
                len(df_train), len(df_val), len(df_test))

    # Verificar features de vecindad (Ajuste 3)
    for col in ["casos_vecinas_lag1", "incidencia_vecinas_lag1"]:
        if col not in df_train.columns:
            raise ValueError(
                f"Columna '{col}' no encontrada.\n"
                "Ejecutá lags.py actualizado con crear_features_vecindad()."
            )
    logger.info("  Features de vecindad verificadas: casos_vecinas_lag1 ✓")

    # Verificar targets multi-horizonte (Ajuste 1b)
    for h in HORIZONTES:
        if f"target_h{h}" not in df_train.columns:
            raise ValueError(
                f"Columna 'target_h{h}' no encontrada.\n"
                "Ejecutá lags.py actualizado con crear_targets_horizonte()."
            )
    logger.info("  Targets multi-horizonte verificados: target_h1 a target_h4 ✓")

    return df_train, df_val, df_test


# =============================================================================
# MÉTRICAS — siempre en escala original (casos reales, sin log)
# =============================================================================

def calcular_metricas(y_real, y_pred, modelo, split, horizonte=0):
    """
    Calcula MAE, RMSE, R² y MAPE.
    Tanto y_real como y_pred deben estar en escala original de casos
    (ya desnormalizados si se usó transformación logarítmica).
    """
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2   = r2_score(y_real, y_pred)

    mask_pos = y_real > 0
    mape = (
        np.abs(y_real[mask_pos] - y_pred[mask_pos]) / y_real[mask_pos]
    ).mean() * 100 if mask_pos.sum() > 0 else np.nan

    return {
        "Modelo":    modelo,
        "Split":     split,
        "Horizonte": horizonte,
        "MAE":       round(mae, 2),
        "RMSE":      round(rmse, 2),
        "MAPE":      round(mape, 1) if not np.isnan(mape) else None,
        "R2":        round(r2, 3),
        "N":         len(y_real)
    }


# =============================================================================
# AJUSTE 2: TRANSFORMACIÓN LOGARÍTMICA
# log(1 + casos) comprime los valores extremos del brote (1391 → ~7.2)
# y amplifica la señal en valores bajos (1 → 0.69, 5 → 1.79).
# np.log1p(x) = log(1+x), evita log(0) cuando no hay casos.
# np.expm1(x) = exp(x)-1, es la inversa exacta de log1p.
# =============================================================================

def aplicar_log(y):
    """Transforma el target a escala logarítmica para entrenar."""
    return np.log1p(y)

def revertir_log(y_log):
    """Revierte la transformación logarítmica a escala original de casos."""
    return np.expm1(y_log)


# =============================================================================
# BÚSQUEDA DE HIPERPARÁMETROS con TimeSeriesSplit + early stopping
# =============================================================================

def buscar_hiperparametros(X_train, y_train_log, horizonte):
    """
    Prueba 72 combinaciones de hiperparámetros con validación cruzada temporal.

    AJUSTE 2: el entrenamiento usa y_train en escala logarítmica.
    AJUSTE 2: early stopping detiene el entrenamiento si el error
              no mejora en 30 rondas consecutivas.

    Retorna los mejores parámetros según MAE en escala original (desnormalizado).
    """
    from itertools import product

    tscv   = TimeSeriesSplit(n_splits=3)
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())

    n_combos = 1
    for v in values:
        n_combos *= len(v)
    logger.info("  Probando %d combinaciones × 3 folds (h=%d)...", n_combos, horizonte)

    mejor_mae      = np.inf
    mejores_params = {}

    for combo in product(*values):
        params = dict(zip(keys, combo))
        params.update({
            "random_state":     42,
            "tree_method":      "hist",
            "objective":        "reg:squarederror",
            "eval_metric":      "mae",
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 30,   # AJUSTE 2: early stopping
        })

        maes_fold = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_vl = X_train[train_idx], X_train[val_idx]
            y_tr, y_vl = y_train_log[train_idx], y_train_log[val_idx]

            m = XGBRegressor(**params, verbosity=0)
            m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

            # Predicciones en escala original para calcular MAE real
            pred_log = m.predict(X_vl).clip(min=0)
            pred_real = revertir_log(pred_log)
            y_real    = revertir_log(y_vl)
            maes_fold.append(mean_absolute_error(y_real, pred_real))

        mae_cv = np.mean(maes_fold)
        if mae_cv < mejor_mae:
            mejor_mae      = mae_cv
            mejores_params = params.copy()

    logger.info("  Mejor MAE CV (escala original) h=%d: %.2f casos", horizonte, mejor_mae)
    logger.info(
        "  Parámetros: %d árboles, profundidad=%d, lr=%.3f",
        mejores_params["n_estimators"],
        mejores_params["max_depth"],
        mejores_params["learning_rate"]
    )
    return mejores_params


# =============================================================================
# ENTRENAMIENTO XGBOOST POR TARGET
# target_col puede ser "confirmed_cases" o "target_h1".."target_h4"
# =============================================================================

def entrenar_xgboost(df_train, df_val, df_test, target_col, label):
    """
    Entrena XGBoost para un target dado usando transformación logarítmica.

    AJUSTE 1: target_col define qué predecimos (semana actual o futura).
    AJUSTE 2: entrena en escala log, evalúa en escala original.
    AJUSTE 3: features incluyen casos_vecinas_lag1.

    Retorna: modelo, predicciones val y test en escala original, métricas.
    """
    horizonte = int(target_col.replace("target_h", "")) if "target_h" in target_col else 0

    logger.info("=" * 60)
    logger.info("  XGBoost — target: %s (%s)", target_col, label)
    logger.info("=" * 60)

    # Features disponibles en el dataset
    features = [f for f in FEATURES_XGB if f in df_train.columns]
    faltantes = [f for f in FEATURES_XGB if f not in df_train.columns]
    if faltantes:
        logger.warning("  Features no encontradas: %s", faltantes)

    # Preparar datos — eliminar NaN en features Y en el target
    def preparar(df):
        mask = df[features].notna().all(axis=1) & df[target_col].notna()
        X = df.loc[mask, features].values
        y = df.loc[mask, target_col].values
        return X, y, mask

    X_train, y_train, _ = preparar(df_train)
    X_val,   y_val,   _ = preparar(df_val)
    X_test,  y_test,  _ = preparar(df_test)

    logger.info(
        "  Datos — Train: %d | Val: %d | Test: %d | Features: %d",
        len(X_train), len(X_val), len(X_test), len(features)
    )

    # AJUSTE 2: transformar target a escala logarítmica
    y_train_log = aplicar_log(y_train)

    # Buscar mejores hiperparámetros
    mejores_params = buscar_hiperparametros(X_train, y_train_log, horizonte)

    # Entrenar modelo final con todos los datos de train
    # El early stopping necesita un eval_set para monitorear el error
    # y detener el entrenamiento cuando deja de mejorar.
    y_val_log = aplicar_log(y_val)
    xgb = XGBRegressor(**mejores_params, verbosity=0)
    xgb.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        verbose=False
    )

    # Predicciones — revertir transformación logarítmica
    pred_val  = revertir_log(xgb.predict(X_val).clip(min=0))
    pred_test = revertir_log(xgb.predict(X_test).clip(min=0))

    met_val  = calcular_metricas(y_val,  pred_val,  f"XGBoost", "Validation", horizonte)
    met_test = calcular_metricas(y_test, pred_test, f"XGBoost", "Test",       horizonte)

    logger.info(
        "  VAL  — MAE: %.2f | RMSE: %.2f | R²: %.3f",
        met_val["MAE"], met_val["RMSE"], met_val["R2"]
    )
    logger.info(
        "  TEST — MAE: %.2f | RMSE: %.2f | R²: %.3f",
        met_test["MAE"], met_test["RMSE"], met_test["R2"]
    )

    return xgb, pred_val, pred_test, met_val, met_test, features


# =============================================================================
# PASO 4: TABLA COMPARATIVA
# =============================================================================

def tabla_comparacion(metricas_actual, metricas_futuro):
    """
    Tabla Sprint 4 (referencia) vs Sprint 5 (XGBoost mejorado).
    Muestra por separado:
      - XGBoost sobre confirmed_cases (comparable con Sprint 4)
      - XGBoost sobre targets futuros (h=1..4)
    """
    print("\n" + "=" * 80)
    print("  SPRINT 4 vs SPRINT 5 — Validación (brote 2024, semanas 1-26)")
    print("=" * 80)
    print(f"  {'Modelo':<35} {'Target':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("  " + "-" * 76)

    print("  --- Sprint 4 (referencia) ---")
    for nombre, m in SPRINT4_VAL.items():
        print(f"  {nombre:<35} {'semana actual':>20} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['R2']:>8.3f}")

    print("  --- Sprint 5 — XGBoost sobre semana actual ---")
    for m in metricas_actual:
        if m["Split"] == "Validation":
            print(f"  {'XGBoost + log + vecindad':<35} {'semana actual':>20} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['R2']:>8.3f}")

    print("  --- Sprint 5 — XGBoost multi-horizonte ---")
    for m in metricas_futuro:
        if m["Split"] == "Validation":
            h = m["Horizonte"]
            semanas = f"{h} semana{'s' if h > 1 else ''} adelante"
            print(f"  {'XGBoost + log + vecindad':<35} {semanas:>20} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['R2']:>8.3f}")

    print("=" * 80)
    print("  MAE=error promedio en casos | R²=varianza explicada (1=perfecto)")
    print("=" * 80 + "\n")


def tabla_horizontes(metricas_futuro):
    """Muestra degradación de métricas por horizonte."""
    print("\n" + "=" * 70)
    print("  XGBOOST — Degradación por horizonte (+ log1p + vecindad)")
    print("=" * 70)
    print(f"  {'Conjunto':<18} {'1 sem':>10} {'2 sem':>10} {'3 sem':>10} {'4 sem':>10}")
    print("  " + "-" * 58)

    for split in ["Validation", "Test"]:
        m_h = {m["Horizonte"]: m for m in metricas_futuro if m["Split"] == split}
        fila_mae = f"  {split+' MAE':<18}"
        fila_r2  = f"  {split+' R²':<18}"
        for h in HORIZONTES:
            if h in m_h:
                fila_mae += f" {m_h[h]['MAE']:>10.2f}"
                fila_r2  += f" {m_h[h]['R2']:>10.3f}"
            else:
                fila_mae += f" {'n/a':>10}"
                fila_r2  += f" {'n/a':>10}"
        print(fila_mae)
        print(fila_r2)
        print()
    print("=" * 70 + "\n")


# =============================================================================
# PASO 5: IMPORTANCIA DE FEATURES
# =============================================================================

def graficar_importancia(modelos, features):
    """
    Importancia de features (gain) para cada modelo.
    Permite ver si casos_vecinas_lag1 aporta información relevante
    y cómo cambia la importancia del clima al aumentar el horizonte.
    """
    logger.info("--- PASO 5: Importancia de features ---")

    nombres = list(modelos.keys())
    n = len(nombres)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 8))
    if n == 1:
        axes = [axes]

    for ax, nombre in zip(axes, nombres):
        xgb = modelos[nombre]
        imp = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=False).head(15)

        colores = [
            "#E74C3C" if any(x in f for x in ["cases_lag", "incidencia_lag"]) else
            "#9B59B6" if "vecinas" in f else   # vecindad en violeta
            "#E67E22" if any(x in f for x in ["temp", "heat"]) else
            "#3498DB" if "precip" in f else
            "#1ABC9C" if "humid" in f else
            "#27AE60" if any(x in f for x in ["semana", "season", "mes"]) else
            "#95A5A6"
            for f in imp.index
        ]

        bars = ax.barh(range(len(imp)), imp.values, color=colores, edgecolor="white", alpha=0.85)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Importancia (gain)")
        ax.set_title(f"{nombre}\nTop 15 features", fontweight="bold")
        for bar, val in zip(bars, imp.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

        print(f"\nTop 10 — {nombre}:")
        for i, (feat, v) in enumerate(imp.head(10).items(), 1):
            print(f"  {i:2}. {feat:<40} {v:.4f}")

    from matplotlib.patches import Patch
    leyenda = [
        Patch(color="#E74C3C", label="Casos/incidencia propios"),
        Patch(color="#9B59B6", label="Vecindad espacial (NUEVO)"),
        Patch(color="#E67E22", label="Temperatura/Heat index"),
        Patch(color="#3498DB", label="Precipitación"),
        Patch(color="#1ABC9C", label="Humedad"),
        Patch(color="#27AE60", label="Estacionalidad"),
        Patch(color="#95A5A6", label="Espaciales/otras"),
    ]
    fig.legend(handles=leyenda, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle("XGBoost mejorado — Importancia de features\nSprint 5 / HU6 (+ log1p + vecindad)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "16_importancia_xgboost_v2.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# PASO 6: CURVA DE DEGRADACIÓN POR HORIZONTE
# =============================================================================

def graficar_degradacion(metricas_futuro):
    """
    MAE y R² por horizonte con líneas de referencia Sprint 4.
    Muestra si los ajustes mejoran el rendimiento respecto a la versión anterior.
    """
    logger.info("--- PASO 6: Degradación por horizonte ---")

    val_h = {m["Horizonte"]: m for m in metricas_futuro if m["Split"] == "Validation"}
    hs    = sorted(val_h.keys())
    maes  = [val_h[h]["MAE"] for h in hs]
    r2s   = [val_h[h]["R2"]  for h in hs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, vals, ylabel, ref_key, titulo in [
        (ax1, maes, "MAE (casos)", "MAE", "Error promedio por horizonte"),
        (ax2, r2s,  "R²",         "R2",  "Varianza explicada por horizonte"),
    ]:
        ax.plot(hs, vals, "o-", color=COLOR_XGB, linewidth=2.5, markersize=9,
                label="XGBoost v2 (log + vecindad)", zorder=3)
        ax.axhline(SPRINT4_VAL["Persistencia (lag 1)"][ref_key],
                   color=COLOR_PERS, linestyle="--", linewidth=1.5, label="Persistencia Sprint 4")
        ax.axhline(SPRINT4_VAL["Random Forest"][ref_key],
                   color=COLOR_RF,   linestyle="--", linewidth=1.5, label="Random Forest Sprint 4")
        if ref_key == "R2":
            ax.axhline(0, color="gray", linestyle=":", linewidth=1, label="R²=0")
        for h, v in zip(hs, vals):
            ax.annotate(f"{v:.2f}", (h, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10,
                        fontweight="bold", color=COLOR_XGB)
        ax.set_xlabel("Horizonte (semanas)")
        ax.set_ylabel(ylabel)
        ax.set_title(titulo, fontweight="bold")
        ax.set_xticks(hs)
        ax.set_xticklabels([f"h={h}" for h in hs])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("XGBoost v2 — Impacto del horizonte\nSprint 5 / HU6 (+ log1p + vecindad espacial)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "17_degradacion_xgboost_v2.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# PASO 7: GUARDAR MODELOS Y RESULTADOS
# =============================================================================

def guardar_modelos(modelos, features, metricas_actual, metricas_futuro):
    logger.info("--- PASO 7: Guardando modelos ---")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for nombre, xgb in modelos.items():
        nombre_archivo = nombre.lower().replace(" ", "_").replace("=", "")
        path = MODELS_DIR / f"xgboost_{nombre_archivo}.pkl"
        with open(path, "wb") as f:
            pickle.dump(xgb, f)
        logger.info("  Guardado: %s", path)

    with open(MODELS_DIR / "xgboost_features_v2.pkl", "wb") as f:
        pickle.dump(features, f)

    todas = metricas_actual + metricas_futuro
    pd.DataFrame(todas).to_csv(MODELS_DIR / "metricas_sprint5_v2.csv", index=False)
    logger.info("  Métricas guardadas: metricas_sprint5_v2.csv")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_xgboost():
    print("\n" + "=" * 65)
    print("  SPRINT 5 / HU6 — XGBoost v2")
    print("  Ajustes: log1p + early stopping + vecindad espacial")
    print("  Targets: semana actual + h=1,2,3,4 semanas adelante")
    print("=" * 65 + "\n")

    # Paso 1
    df_train, df_val, df_test = cargar_splits()

    # Paso 2: AJUSTE 1a — XGBoost sobre confirmed_cases (comparable Sprint 4)
    logger.info("=== AJUSTE 1a: target = confirmed_cases (semana actual) ===")
    (xgb_actual, pv_actual, pt_actual,
     met_actual_val, met_actual_test, features) = entrenar_xgboost(
        df_train, df_val, df_test,
        target_col="confirmed_cases",
        label="semana actual"
    )
    metricas_actual = [met_actual_val, met_actual_test]

    # Paso 3: AJUSTE 1b — XGBoost sobre targets futuros h=1..4
    logger.info("=== AJUSTE 1b: targets futuros h=1,2,3,4 ===")
    modelos_futuro  = {}
    metricas_futuro = []

    for h in HORIZONTES:
        target_col = f"target_h{h}"
        (xgb_h, pv_h, pt_h,
         met_val_h, met_test_h, _) = entrenar_xgboost(
            df_train, df_val, df_test,
            target_col=target_col,
            label=f"{h} semana(s) adelante"
        )
        modelos_futuro[f"h={h}"] = xgb_h
        metricas_futuro.extend([met_val_h, met_test_h])

    # Todos los modelos juntos para graficar importancia
    todos_modelos = {"Semana actual": xgb_actual, **modelos_futuro}

    # Paso 4: comparación
    tabla_comparacion(metricas_actual, metricas_futuro)
    tabla_horizontes(metricas_futuro)

    # Paso 5: importancia
    graficar_importancia(todos_modelos, features)

    # Paso 6: degradación
    graficar_degradacion(metricas_futuro)

    # Métricas por período — XGBoost semana actual en test
    mask_test = (
        df_test[features].notna().all(axis=1) &
        df_test["confirmed_cases"].notna()
    )
    df_te = df_test[mask_test].copy()
    df_te["pred"] = revertir_log(
        xgb_actual.predict(df_test.loc[mask_test, features].values).clip(min=0)
    )
    df_te["error"] = abs(df_te["confirmed_cases"] - df_te["pred"])

    print("\n" + "=" * 60)
    print("  XGBoost (semana actual) — Rendimiento por período (Test)")
    print("=" * 60)
    resumen = df_te.groupby("year").agg(
        casos_reales = ("confirmed_cases", "mean"),
        prediccion   = ("pred",            "mean"),
        MAE          = ("error",           "mean"),
        n_semanas    = ("confirmed_cases", "count")
    ).round(2)
    print(resumen.to_string())
    print("=" * 60 + "\n")

    # Paso 7: guardar
    guardar_modelos(todos_modelos, features, metricas_actual, metricas_futuro)

    print("\n✓ Sprint 5 / HU6 — XGBoost v2 completado.")
    print("  Ajustes aplicados: log1p ✓ | early stopping ✓ | vecindad espacial ✓")
    print("  Próximo paso: LSTM y GRU")

    return {"modelos": todos_modelos, "metricas": metricas_actual + metricas_futuro}


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    resultado = run_xgboost()
