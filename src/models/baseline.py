"""
Sprint 4 / HU6 — Modelos baseline, persistencia, Ridge climático y Random Forest

1. PERSISTENCIA — Predice que esta semana habrá los mismos casos que la semana anterior.
   Es el baseline mínimo de series temporales. No aprende nada: solo recuerda.
   Referencia obligatoria en papers de predicción de dengue (Sebastianelli et al. 2024).

2. MEDIA HISTÓRICA — Predice el promedio de esa semana epidemiológica en años anteriores.
   Captura estacionalidad sin ML.

3. RIDGE CLIMÁTICO — Regresión lineal regularizada usando solo variables climáticas
   y estacionalidad, sin lags de casos. Valida la hipótesis central de la tesis:
   ¿el clima alcanza para predecir brotes?

4. RANDOM FOREST — Ensemble de 300 árboles con las 67 features completas del Sprint 4.

Pipeline:
    Paso 1 → Cargar splits train / validation / test
    Paso 2 → Persistencia (lag 1)
    Paso 3 → Baseline: media histórica por semana
    Paso 4 → Ridge climático
    Paso 5 → Random Forest: entrenamiento y evaluación
    Paso 6 → Comparación de métricas (escalera de complejidad)
    Paso 7 → Análisis de importancia de features (RF y Ridge)
    Paso 8 → Análisis de residuos
    Paso 9 → Guardar modelos y resultados
"""

import logging
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES
# =============================================================================

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/saved")
FIGURES_DIR   = Path("reports/figures")

TRAIN_FILE = PROCESSED_DIR / "train.parquet"
VAL_FILE   = PROCESSED_DIR / "validation.parquet"
TEST_FILE  = PROCESSED_DIR / "test.parquet"

TARGET = "confirmed_cases"

# Features para Random Forest — todas las 67 features del Sprint 4
FEATURES_RF = (
    ["cases_lag1", "cases_lag2", "cases_lag3", "cases_lag4",
     "incidencia_lag1", "incidencia_lag2", "incidencia_lag3", "incidencia_lag4"] +
    ["temp_mean", "precipitation", "humidity_mean", "heat_index_mean",
     "temp_mean_anomaly", "precipitation_anomaly", "humidity_mean_anomaly"] +
    [f"temp_mean_lag{i}"           for i in range(1, 5)] +
    [f"precipitation_lag{i}"       for i in range(1, 5)] +
    [f"humidity_mean_lag{i}"       for i in range(1, 5)] +
    [f"heat_index_mean_lag{i}"     for i in range(1, 5)] +
    [f"temp_mean_anomaly_lag{i}"   for i in range(1, 5)] +
    [f"precipitation_anomaly_lag{i}" for i in range(1, 5)] +
    [f"humidity_mean_anomaly_lag{i}" for i in range(1, 5)] +
    ["semana_sin", "semana_cos", "is_epidemic_season", "mes_aprox"] +
    ["comuna_id", "es_comuna_1", "poblacion"]
)

# Features para Ridge — solo clima y estacionalidad, SIN lags de casos
# Justificación: queremos medir cuánto predice el clima solo, sin información
# autorregresiva. Si Ridge climático es competitivo, el clima aporta señal real.
FEATURES_RIDGE = (
    ["temp_mean", "precipitation", "humidity_mean", "heat_index_mean",
     "temp_mean_anomaly", "precipitation_anomaly", "humidity_mean_anomaly"] +
    [f"temp_mean_lag{i}"             for i in range(1, 5)] +
    [f"precipitation_lag{i}"         for i in range(1, 5)] +
    [f"humidity_mean_lag{i}"         for i in range(1, 5)] +
    [f"heat_index_mean_lag{i}"       for i in range(1, 5)] +
    [f"temp_mean_anomaly_lag{i}"     for i in range(1, 5)] +
    [f"precipitation_anomaly_lag{i}" for i in range(1, 5)] +
    [f"humidity_mean_anomaly_lag{i}" for i in range(1, 5)] +
    ["semana_sin", "semana_cos", "is_epidemic_season", "mes_aprox"] +
    ["comuna_id", "es_comuna_1", "poblacion"]
)

# Paleta de colores
COLOR_BASELINE     = "#95A5A6"
COLOR_PERSISTENCIA = "#F39C12"
COLOR_RIDGE        = "#8E44AD"
COLOR_RF           = "#3498DB"


# =============================================================================
# PASO 1: CARGA DE DATOS
# =============================================================================

def cargar_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los tres splits generados en el Sprint 4."""
    logger.info("--- PASO 1: Cargando splits ---")

    for filepath in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not filepath.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {filepath}\n"
                "Ejecutá primero src/features/lags.py"
            )

    df_train = pd.read_parquet(TRAIN_FILE)
    df_val   = pd.read_parquet(VAL_FILE)
    df_test  = pd.read_parquet(TEST_FILE)

    logger.info("  Train:      %d filas", len(df_train))
    logger.info("  Validation: %d filas", len(df_val))
    logger.info("  Test:       %d filas", len(df_test))

    return df_train, df_val, df_test


# =============================================================================
# PASO 2: PERSISTENCIA — lag 1
# El modelo más simple posible: predice que esta semana habrá los mismos
# casos que la semana anterior. No aprende nada del clima ni de la historia.
# Es el piso mínimo que cualquier modelo más complejo debe superar.
# Referencia estándar en predicción de dengue.
# =============================================================================

def calcular_persistencia(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Persistencia: pred(t) = casos(t-1).

    Usa cases_lag1 que ya está calculado en el dataset.
    Las filas donde cases_lag1 es NaN se rellenan con 0
    (primeras semanas de cada comuna, igual que en RF).
    """
    logger.info("--- PASO 2: Calculando persistencia (lag 1) ---")

    pred_val  = df_val["cases_lag1"].fillna(0).clip(lower=0).values
    pred_test = df_test["cases_lag1"].fillna(0).clip(lower=0).values

    y_val  = df_val[TARGET].values
    y_test = df_test[TARGET].values

    met_val  = calcular_metricas(y_val,  pred_val,  "Persistencia (lag 1)", "Validation")
    met_test = calcular_metricas(y_test, pred_test, "Persistencia (lag 1)", "Test")

    logger.info(
        "  Persistencia val — MAE: %.2f | RMSE: %.2f | R²: %.3f",
        met_val["MAE"], met_val["RMSE"], met_val["R2"]
    )

    return pred_val, pred_test, met_val, met_test


# =============================================================================
# PASO 3: BASELINE — MEDIA HISTÓRICA POR SEMANA
# Predice el promedio histórico de esa semana epidemiológica y comuna.
# Captura estacionalidad sin ML.
# =============================================================================

def calcular_baseline(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Baseline: media histórica por (epi_week, comuna_id) calculada en train.
    Fallback a media global de la semana si no hay datos para esa combinación.
    """
    logger.info("--- PASO 3: Calculando baseline (media histórica) ---")

    media_semana_comuna = (
        df_train.groupby(["epi_week", "comuna_id"])[TARGET]
        .mean()
        .reset_index()
        .rename(columns={TARGET: "pred_baseline"})
    )
    media_semana = (
        df_train.groupby("epi_week")[TARGET]
        .mean()
        .reset_index()
        .rename(columns={TARGET: "pred_baseline_global"})
    )

    def predecir(df):
        df_pred = df.merge(media_semana_comuna, on=["epi_week", "comuna_id"], how="left") \
                    .merge(media_semana, on="epi_week", how="left")
        return df_pred["pred_baseline"].fillna(
            df_pred["pred_baseline_global"]
        ).fillna(0).clip(lower=0).values

    pred_val  = predecir(df_val)
    pred_test = predecir(df_test)

    y_val  = df_val[TARGET].values
    y_test = df_test[TARGET].values

    met_val  = calcular_metricas(y_val,  pred_val,  "Baseline (media histórica)", "Validation")
    met_test = calcular_metricas(y_test, pred_test, "Baseline (media histórica)", "Test")

    logger.info(
        "  Baseline val — MAE: %.2f | RMSE: %.2f | R²: %.3f",
        met_val["MAE"], met_val["RMSE"], met_val["R2"]
    )

    return pred_val, pred_test, met_val, met_test


# =============================================================================
# PASO 4: RIDGE CLIMÁTICO
# Regresión lineal regularizada usando SOLO variables climáticas y estacionalidad.
# No incluye lags de casos — mide el aporte puro del clima.
#
# Decisión técnica: usamos Ridge (L2) en lugar de OLS porque:
# - Estabiliza coeficientes con variables correlacionadas (temp y heat_index)
# - Evita sobreajuste con dataset pequeño (720 filas efectivas)
# - alpha=1.0 es conservador; se puede ajustar en Sprint 5
#
# El scaler se ajusta SOLO con train (sin data leakage).
# =============================================================================

def entrenar_ridge_climatico(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame
) -> tuple[Ridge, np.ndarray, np.ndarray, dict, dict, list]:
    """
    Ridge con variables climáticas y estacionalidad, sin lags de casos.

    Retorna: modelo, pred_val, pred_test, met_val, met_test, features_usadas
    """
    logger.info("--- PASO 4: Entrenando Ridge climático ---")

    features_disponibles = [f for f in FEATURES_RIDGE if f in df_train.columns]
    features_faltantes   = [f for f in FEATURES_RIDGE if f not in df_train.columns]
    if features_faltantes:
        logger.warning("  Features no encontradas: %s", features_faltantes)

    # Eliminar NaN (primeras semanas con lags climáticos faltantes)
    mask_train = df_train[features_disponibles].notna().all(axis=1)
    mask_val   = df_val[features_disponibles].notna().all(axis=1)
    mask_test  = df_test[features_disponibles].notna().all(axis=1)

    X_train_raw = df_train.loc[mask_train, features_disponibles].values
    y_train     = df_train.loc[mask_train, TARGET].values
    X_val_raw   = df_val.loc[mask_val,   features_disponibles].values
    y_val       = df_val.loc[mask_val,   TARGET].values
    X_test_raw  = df_test.loc[mask_test, features_disponibles].values
    y_test      = df_test.loc[mask_test, TARGET].values

    # Normalización — fit solo en train
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    X_test  = scaler.transform(X_test_raw)

    logger.info(
        "  Train efectivo: %d filas (%.1f%% del total)",
        len(X_train), len(X_train) / len(df_train) * 100
    )
    logger.info("  Features climáticas: %d (sin lags de casos)", len(features_disponibles))

    # Entrenar Ridge
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)

    # Predicciones
    pred_val  = ridge.predict(X_val).clip(min=0)
    pred_test = ridge.predict(X_test).clip(min=0)

    met_val  = calcular_metricas(y_val,  pred_val,  "Ridge (climático)", "Validation")
    met_test = calcular_metricas(y_test, pred_test, "Ridge (climático)", "Test")

    logger.info(
        "  Ridge val — MAE: %.2f | RMSE: %.2f | R²: %.3f",
        met_val["MAE"], met_val["RMSE"], met_val["R2"]
    )

    return ridge, pred_val, pred_test, met_val, met_test, features_disponibles, scaler


# =============================================================================
# PASO 5: RANDOM FOREST
# =============================================================================

def entrenar_random_forest(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame
) -> tuple:
    """
    Random Forest Regressor con las 67 features del Sprint 4.

    Parámetros:
    - n_estimators=300: suficientes árboles para estabilidad
    - max_depth=10: evita sobreajuste con dataset pequeño
    - min_samples_leaf=5: cada hoja necesita al menos 5 muestras
    - random_state=42: reproducibilidad
    """
    logger.info("--- PASO 5: Entrenando Random Forest ---")

    features_disponibles = [f for f in FEATURES_RF if f in df_train.columns]
    features_faltantes   = [f for f in FEATURES_RF if f not in df_train.columns]
    if features_faltantes:
        logger.warning("  Features no encontradas: %s", features_faltantes)

    mask_train = df_train[features_disponibles].notna().all(axis=1)
    mask_val   = df_val[features_disponibles].notna().all(axis=1)
    mask_test  = df_test[features_disponibles].notna().all(axis=1)

    X_train = df_train.loc[mask_train, features_disponibles].values
    y_train = df_train.loc[mask_train, TARGET].values
    X_val   = df_val.loc[mask_val,   features_disponibles].values
    y_val   = df_val.loc[mask_val,   TARGET].values
    X_test  = df_test.loc[mask_test, features_disponibles].values
    y_test  = df_test.loc[mask_test, TARGET].values

    logger.info(
        "  Train efectivo: %d filas (%.1f%% del total, eliminados NaN de lags)",
        len(X_train), len(X_train) / len(df_train) * 100
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    logger.info("  Modelo entrenado con %d árboles", rf.n_estimators)

    pred_val  = rf.predict(X_val).clip(min=0)
    pred_test = rf.predict(X_test).clip(min=0)

    met_val  = calcular_metricas(y_val,  pred_val,  "Random Forest", "Validation")
    met_test = calcular_metricas(y_test, pred_test, "Random Forest", "Test")

    logger.info(
        "  RF val  — MAE: %.2f | RMSE: %.2f | R²: %.3f",
        met_val["MAE"], met_val["RMSE"], met_val["R2"]
    )
    logger.info(
        "  RF test — MAE: %.2f | RMSE: %.2f | R²: %.3f",
        met_test["MAE"], met_test["RMSE"], met_test["R2"]
    )

    return (rf, pred_val, pred_test, met_val, met_test,
            features_disponibles, y_val, y_test, mask_val, mask_test)


# =============================================================================
# MÉTRICAS
# =============================================================================

def calcular_metricas(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    modelo: str,
    split: str
) -> dict:
    """
    Calcula MAE, RMSE, MAPE y R².

    MAE  — Error absoluto medio en casos
    RMSE — Raíz del error cuadrático medio (penaliza errores grandes)
    MAPE — Error porcentual absoluto medio (solo semanas con casos > 0)
    R²   — Varianza explicada (1 = perfecto, 0 = no mejor que la media)
    """
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2   = r2_score(y_real, y_pred)

    mask_pos = y_real > 0
    if mask_pos.sum() > 0:
        mape = (
            np.abs(y_real[mask_pos] - y_pred[mask_pos]) / y_real[mask_pos]
        ).mean() * 100
    else:
        mape = np.nan

    return {
        "Modelo": modelo,
        "Split":  split,
        "MAE":    round(mae, 2),
        "RMSE":   round(rmse, 2),
        "MAPE":   round(mape, 1) if not np.isnan(mape) else None,
        "R2":     round(r2, 3),
        "N":      len(y_real)
    }


# =============================================================================
# PASO 6: COMPARACIÓN DE MÉTRICAS — escalera de complejidad
# =============================================================================

def tabla_comparacion(lista_metricas: list[dict]):
    """
    Imprime tabla comparativa en orden de complejidad creciente.
    La escalera permite ver exactamente qué aporta cada componente:
    - Persistencia → precio del olvido
    - Media histórica → precio de ignorar el clima
    - Ridge climático → aporte del clima sin autorregresión
    - Random Forest → aporte de combinar todo
    """
    print("\n" + "=" * 78)
    print("  COMPARACIÓN DE MODELOS — Sprint 4 / HU6 (escalera de complejidad)")
    print("=" * 78)
    print(f"  {'Modelo':<35} {'Split':<12} {'MAE':>8} {'RMSE':>8} "
          f"{'MAPE':>8} {'R²':>8}")
    print("  " + "-" * 74)
    for m in lista_metricas:
        mape_str = f"{m['MAPE']:.1f}%" if m["MAPE"] else "n/a"
        print(
            f"  {m['Modelo']:<35} {m['Split']:<12} "
            f"{m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
            f"{mape_str:>8} {m['R2']:>8.3f}"
        )
    print("=" * 78)
    print("  MAE = error absoluto medio (casos) | RMSE = raíz error cuadrático")
    print("  MAPE = error porcentual medio | R² = varianza explicada (1=perfecto)")
    print("=" * 78 + "\n")


# =============================================================================
# PASO 7: IMPORTANCIA DE FEATURES
# =============================================================================

def graficar_importancia_features(
    rf: RandomForestRegressor,
    features_rf: list[str],
    ridge: Ridge,
    features_ridge: list[str]
):
    """
    Dos gráficos de importancia:
    1. Random Forest — importancia Gini (top 20)
    2. Ridge climático — coeficientes absolutos (top 15)
       Los coeficientes del Ridge muestran qué variables climáticas
       tienen más peso real sobre los casos.
    """
    logger.info("--- PASO 7: Analizando importancia de features ---")

    # --- RF: importancia Gini ---
    importancias_rf = pd.Series(
        rf.feature_importances_, index=features_rf
    ).sort_values(ascending=False)

    top20 = importancias_rf.head(20)

    colores_rf = [
        "#E74C3C" if "lag" in f and "cases" in f else
        "#E67E22" if "lag" in f and any(v in f for v in ["temp", "heat"]) else
        "#3498DB" if "lag" in f and "precip" in f else
        "#27AE60" if any(s in f for s in ["semana", "season", "mes"]) else
        "#9B59B6" if "comuna" in f else
        "#95A5A6"
        for f in top20.index
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel 1 — RF
    ax = axes[0]
    bars = ax.barh(range(len(top20)), top20.values,
                   color=colores_rf, edgecolor="white", alpha=0.85)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia Gini")
    ax.set_title("Random Forest — Top 20 features\nSprint 4 / HU6", fontweight="bold")
    for bar, val in zip(bars, top20.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    from matplotlib.patches import Patch
    leyenda = [
        Patch(color="#E74C3C", label="Lags de casos"),
        Patch(color="#E67E22", label="Lags de temperatura"),
        Patch(color="#3498DB", label="Lags de precipitación"),
        Patch(color="#27AE60", label="Estacionalidad"),
        Patch(color="#9B59B6", label="Espaciales"),
        Patch(color="#95A5A6", label="Otras"),
    ]
    ax.legend(handles=leyenda, loc="lower right", fontsize=8)

    # Panel 2 — Ridge: coeficientes absolutos top 15
    coefs_ridge = pd.Series(
        np.abs(ridge.coef_), index=features_ridge
    ).sort_values(ascending=False).head(15)

    colores_ridge = [
        "#E67E22" if any(v in f for v in ["temp", "heat"]) else
        "#3498DB" if "precip" in f else
        "#1ABC9C" if "humid" in f else
        "#27AE60" if any(s in f for s in ["semana", "season", "mes"]) else
        "#9B59B6" if "comuna" in f else
        "#95A5A6"
        for f in coefs_ridge.index
    ]

    ax2 = axes[1]
    bars2 = ax2.barh(range(len(coefs_ridge)), coefs_ridge.values,
                     color=colores_ridge, edgecolor="white", alpha=0.85)
    ax2.set_yticks(range(len(coefs_ridge)))
    ax2.set_yticklabels(coefs_ridge.index, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Coeficiente absoluto (escalado)")
    ax2.set_title("Ridge climático — Top 15 variables\n(sin lags de casos)", fontweight="bold")
    for bar, val in zip(bars2, coefs_ridge.values):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=8)

    plt.suptitle("Importancia de features — RF vs Ridge climático\nSprint 4 / HU6",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "14_importancia_features_rf_ridge.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    print("\nTop 10 features — Random Forest:")
    for i, (feat, imp) in enumerate(importancias_rf.head(10).items(), 1):
        print(f"  {i:2}. {feat:<40} {imp:.4f}")

    print("\nTop 10 variables — Ridge climático (coef. absoluto):")
    for i, (feat, coef) in enumerate(coefs_ridge.head(10).items(), 1):
        print(f"  {i:2}. {feat:<40} {coef:.4f}")

    return importancias_rf


# =============================================================================
# PASO 8: ANÁLISIS DE RESIDUOS
# =============================================================================

def graficar_residuos(
    y_real: np.ndarray,
    pred_baseline: np.ndarray,
    pred_rf: np.ndarray,
    split: str = "Validation"
):
    """
    Scatter reales vs predichos e histograma de residuos
    para Baseline y Random Forest.
    """
    logger.info("--- PASO 8: Analizando residuos ---")

    n = min(len(y_real), len(pred_baseline), len(pred_rf))
    y_real        = y_real[:n]
    pred_baseline = pred_baseline[:n]
    pred_rf       = pred_rf[:n]

    res_bl = y_real - pred_baseline
    res_rf = y_real - pred_rf

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for ax, pred, res, color, titulo in [
        (axes[0, 0], pred_baseline, res_bl, COLOR_BASELINE, "Baseline"),
        (axes[0, 1], pred_rf,       res_rf, COLOR_RF,       "Random Forest"),
    ]:
        max_val = max(y_real.max(), pred.max())
        ax.scatter(y_real, pred, alpha=0.4, s=20, color=color)
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Predicción perfecta")
        ax.set_xlabel("Casos reales")
        ax.set_ylabel("Casos predichos")
        ax.set_title(f"{titulo} — Reales vs Predichos ({split})")
        ax.legend()
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))

    for ax, res, color, titulo in [
        (axes[1, 0], res_bl, COLOR_BASELINE, "Baseline"),
        (axes[1, 1], res_rf, COLOR_RF,       "Random Forest"),
    ]:
        ax.hist(res, bins=40, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(x=res.mean(), color="orange", linestyle="-", linewidth=1.5,
                   label=f"Media residuos: {res.mean():.1f}")
        ax.set_xlabel("Residuo (casos reales - casos predichos)")
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"{titulo} — Distribución de residuos ({split})")
        ax.legend()

    plt.suptitle(f"Análisis de residuos — Baseline vs Random Forest ({split})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"15_residuos_baseline_vs_rf_{split.lower()}.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nEstadísticas de residuos ({split}):")
    print(f"  {'Modelo':<25} {'Media':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 55)
    for nombre, res in [("Baseline", res_bl), ("Random Forest", res_rf)]:
        print(f"  {nombre:<25} {res.mean():>8.1f} {res.std():>8.1f} "
              f"{res.min():>8.1f} {res.max():>8.1f}")


# =============================================================================
# PASO 9: GUARDAR MODELOS Y RESULTADOS
# =============================================================================

def guardar_modelos(
    rf: RandomForestRegressor,
    ridge: Ridge,
    ridge_scaler: StandardScaler,
    metricas: list[dict]
):
    """Guarda modelos entrenados y métricas en disco."""
    logger.info("--- PASO 9: Guardando modelos y resultados ---")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for obj, nombre in [
        (rf,           "random_forest.pkl"),
        (ridge,        "ridge_climatico.pkl"),
        (ridge_scaler, "ridge_scaler.pkl"),
    ]:
        path = MODELS_DIR / nombre
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("  Guardado: %s", path)

    df_metricas = pd.DataFrame(metricas)
    metricas_path = MODELS_DIR / "metricas_sprint4.csv"
    df_metricas.to_csv(metricas_path, index=False)
    logger.info("  Métricas guardadas en: %s", metricas_path)


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_baseline_rf():
    """Pipeline completo — 4 modelos en escalera de complejidad."""

    print("\n" + "=" * 65)
    print("  SPRINT 4 / HU6 — Escalera de modelos baseline")
    print("  Persistencia → Media histórica → Ridge climático → RF")
    print("=" * 65 + "\n")

    # Paso 1: carga
    df_train, df_val, df_test = cargar_splits()

    # Paso 2: persistencia
    pred_val_pers, pred_test_pers, met_pers_val, met_pers_test = calcular_persistencia(
        df_val, df_test
    )

    # Paso 3: baseline media histórica
    pred_val_bl, pred_test_bl, met_bl_val, met_bl_test = calcular_baseline(
        df_train, df_val, df_test
    )

    # Paso 4: Ridge climático
    (ridge, pred_val_ridge, pred_test_ridge,
     met_ridge_val, met_ridge_test,
     features_ridge, ridge_scaler) = entrenar_ridge_climatico(
        df_train, df_val, df_test
    )

    # Paso 5: Random Forest
    (rf, pred_val_rf, pred_test_rf,
     met_rf_val, met_rf_test,
     features_rf, y_val_rf, y_test_rf,
     mask_val_rf, mask_test_rf) = entrenar_random_forest(
        df_train, df_val, df_test
    )

    # Paso 6: comparación — todas las métricas en orden de complejidad
    todas_metricas = [
        met_pers_val,  met_pers_test,
        met_bl_val,    met_bl_test,
        met_ridge_val, met_ridge_test,
        met_rf_val,    met_rf_test,
    ]
    tabla_comparacion(todas_metricas)

    # Paso 7: importancia de features (RF + Ridge)
    importancias = graficar_importancia_features(
        rf, features_rf, ridge, features_ridge
    )

    # Paso 8: residuos (Baseline vs RF, alineados)
    mask_val_rf_bool = df_val[features_rf].notna().all(axis=1)
    y_val_alineado   = df_val.loc[mask_val_rf_bool, TARGET].values
    pred_bl_alineado = pred_val_bl[mask_val_rf_bool.values]

    graficar_residuos(y_val_alineado, pred_bl_alineado, pred_val_rf, split="Validation")

    # Métricas por período en test (RF vs Baseline)
    mask_test_rf_bool = df_test[features_rf].notna().all(axis=1)
    df_test_eval = df_test[mask_test_rf_bool].copy()
    df_test_eval["pred_rf"]  = pred_test_rf
    df_test_eval["pred_bl"]  = pred_test_bl[mask_test_rf_bool.values]
    df_test_eval["pred_ridge"] = pred_test_ridge[mask_test_rf_bool.values] if len(pred_test_ridge) == mask_test_rf_bool.sum() else np.nan
    df_test_eval["error_rf"]   = abs(df_test_eval[TARGET] - df_test_eval["pred_rf"])
    df_test_eval["error_bl"]   = abs(df_test_eval[TARGET] - df_test_eval["pred_bl"])
    df_test_eval["error_ridge"] = abs(df_test_eval[TARGET] - df_test_eval["pred_ridge"])

    print("\n" + "=" * 70)
    print("  MÉTRICAS POR PERÍODO — Test set")
    print("=" * 70)
    resumen = df_test_eval.groupby("year").agg(
        casos_reales  = (TARGET,          "mean"),
        MAE_baseline  = ("error_bl",      "mean"),
        MAE_ridge     = ("error_ridge",   "mean"),
        MAE_rf        = ("error_rf",      "mean"),
        n_filas       = (TARGET,          "count")
    ).round(2)
    print(resumen.to_string())
    print("=" * 70 + "\n")

    # Paso 9: guardar
    guardar_modelos(rf, ridge, ridge_scaler, todas_metricas)

    print("\nSprint 4 / HU6 completado — 4 modelos entrenados y comparados.")
    print("Próximo paso: Sprint 5 — XGBoost")

    return {
        "rf":           rf,
        "ridge":        ridge,
        "metricas":     todas_metricas,
        "importancias": importancias,
    }


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    resultado = run_baseline_rf()
