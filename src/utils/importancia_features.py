"""
================================================================================
Sprint 6 / HU7 — Importancia de features
================================================================================

La importancia de features responde la pregunta: ¿qué variables son más
relevantes para que cada modelo haga sus predicciones?

Esta información es valiosa para la tesis por tres razones:

1. VALIDACIÓN EPIDEMIOLÓGICA:
   Si el modelo usa variables que tienen sentido biológico (temperatura,
   humedad, historial de casos), podemos confiar en que aprendió patrones
   reales del dengue — no correlaciones espurias. Por ejemplo, es correcto
   que la temperatura de hace 4 semanas importe: el mosquito Aedes aegypti
   tarda 2-4 semanas en completar su ciclo de vida, por lo que las
   condiciones climáticas de hace un mes afectan los casos de hoy.

2. COMPRENSIÓN DEL SISTEMA DE ALERTAS:
   Al comparar la importancia en h=1 vs h=4, podemos ver cómo cambia
   lo que el modelo "mira" según el horizonte de predicción. Esta
   comparación es un hallazgo original de la tesis.

3. MEJORAS FUTURAS:
   Las features con importancia cercana a cero pueden eliminarse en
   versiones futuras sin perder precisión, simplificando el sistema.

MODELOS Y MÉTODOS:
  ┌─────────────────────┬──────────────────────────────────────────┐
  │ Modelo              │ Método                                   │
  ├─────────────────────┼──────────────────────────────────────────┤
  │ XGBoost semana act. │ Importancia nativa (gain)                │
  │ XGBoost h=1..h=4    │ Importancia nativa (gain)                │
  │ LSTM h=3            │ SHAP values (GradientExplainer)          │
  │ LSTM h=4            │ SHAP values (GradientExplainer)          │
  └─────────────────────┴──────────────────────────────────────────┘

¿POR QUÉ GRADIENTEXPLAINER EN LUGAR DE DEEPEXPLAINER?
-------------------------------------------------------
DeepExplainer falla con arquitecturas LSTM modernas en TensorFlow 2.x
porque usa una operación interna (TensorListStack) que SHAP no sabe
derivar automáticamente.

GradientExplainer es más robusto: calcula los SHAP values usando
gradientes estándar de TensorFlow (tape.gradient), que funcionan con
cualquier arquitectura — incluidas las LSTM con capas densas.

La diferencia práctica: GradientExplainer puede ser algo más lento
que DeepExplainer, pero los resultados son equivalentes para nuestro
propósito de comparación de importancia relativa.

PIPELINE:
  Paso 1 → Cargar modelos y datos
  Paso 2 → Calcular importancia nativa de XGBoost (h=0..h=4)
  Paso 3 → Calcular SHAP values para LSTM h=3 y h=4
  Paso 4 → Generar gráficos de barras y por categoría
  Paso 5 → Guardar resultados

TIEMPO ESTIMADO:
  XGBoost:     ~30 segundos
  SHAP h=3:    ~20 minutos en CPU
  SHAP h=4:    ~20 minutos en CPU
  Total:        ~40 minutos

PREREQUISITO:
  pip install shap --break-system-packages
  Ejecutar todos los scripts del Sprint 5 antes de este.
"""

import os
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

tf.random.set_seed(42)
np.random.seed(42)


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/saved")
FIGURES_DIR   = Path("reports/figures")
REPORTS_DIR   = Path("reports")

# Ventana temporal del LSTM — debe coincidir con lstm_model.py
WINDOW_SIZE = 12

# Horizontes a analizar para XGBoost (los que usa el ensemble)
HORIZONTES_XGB = [0, 1, 2, 3, 4]

# Horizontes a analizar para LSTM (los que usa el ensemble: h=3 y h=4)
HORIZONTES_LSTM = [3, 4]

# Top N features a mostrar en los gráficos
TOP_N = 15

# Número de muestras para SHAP
N_BACKGROUND = 100
N_EXPLICAR   = 300


# =============================================================================
# LISTA DE FEATURES
# =============================================================================

FEATURES_XGB = (
    ["cases_lag1", "cases_lag2", "cases_lag3", "cases_lag4",
     "incidencia_lag1", "incidencia_lag2", "incidencia_lag3", "incidencia_lag4"] +
    ["casos_vecinas_lag1", "incidencia_vecinas_lag1"] +
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

FEATURES_LSTM = (
    ["cases_lag1_norm", "cases_lag2_norm", "cases_lag3_norm", "cases_lag4_norm",
     "incidencia_lag1_norm", "incidencia_lag2_norm",
     "incidencia_lag3_norm", "incidencia_lag4_norm"] +
    ["casos_vecinas_lag1", "incidencia_vecinas_lag1"] +
    [f"temp_mean_lag{i}_norm"             for i in range(1, 5)] +
    [f"precipitation_lag{i}_norm"         for i in range(1, 5)] +
    [f"humidity_mean_lag{i}_norm"         for i in range(1, 5)] +
    [f"heat_index_mean_lag{i}_norm"       for i in range(1, 5)] +
    [f"temp_mean_anomaly_lag{i}_norm"     for i in range(1, 5)] +
    [f"precipitation_anomaly_lag{i}_norm" for i in range(1, 5)] +
    [f"humidity_mean_anomaly_lag{i}_norm" for i in range(1, 5)] +
    ["semana_sin", "semana_cos", "is_epidemic_season", "mes_aprox"] +
    ["es_comuna_1", "poblacion"]
)


# =============================================================================
# CATEGORÍAS DE FEATURES
# =============================================================================

CATEGORIAS = {
    "Casos propios": (
        ["cases_lag1", "cases_lag2", "cases_lag3", "cases_lag4",
         "incidencia_lag1", "incidencia_lag2", "incidencia_lag3", "incidencia_lag4"] +
        ["cases_lag1_norm", "cases_lag2_norm", "cases_lag3_norm", "cases_lag4_norm",
         "incidencia_lag1_norm", "incidencia_lag2_norm",
         "incidencia_lag3_norm", "incidencia_lag4_norm"]
    ),
    "Vecindad espacial": ["casos_vecinas_lag1", "incidencia_vecinas_lag1"],
    "Temperatura": (
        ["temp_mean", "temp_mean_anomaly"] +
        [f"temp_mean_lag{i}" for i in range(1, 5)] +
        [f"temp_mean_anomaly_lag{i}" for i in range(1, 5)] +
        [f"temp_mean_lag{i}_norm" for i in range(1, 5)] +
        [f"temp_mean_anomaly_lag{i}_norm" for i in range(1, 5)]
    ),
    "Precipitación": (
        ["precipitation", "precipitation_anomaly"] +
        [f"precipitation_lag{i}" for i in range(1, 5)] +
        [f"precipitation_anomaly_lag{i}" for i in range(1, 5)] +
        [f"precipitation_lag{i}_norm" for i in range(1, 5)] +
        [f"precipitation_anomaly_lag{i}_norm" for i in range(1, 5)]
    ),
    "Humedad/Calor": (
        ["humidity_mean", "heat_index_mean", "humidity_mean_anomaly"] +
        [f"humidity_mean_lag{i}" for i in range(1, 5)] +
        [f"heat_index_mean_lag{i}" for i in range(1, 5)] +
        [f"humidity_mean_anomaly_lag{i}" for i in range(1, 5)] +
        [f"humidity_mean_lag{i}_norm" for i in range(1, 5)] +
        [f"heat_index_mean_lag{i}_norm" for i in range(1, 5)] +
        [f"humidity_mean_anomaly_lag{i}_norm" for i in range(1, 5)]
    ),
    "Estacionalidad": ["semana_sin", "semana_cos", "is_epidemic_season", "mes_aprox"],
    "Geografía": ["comuna_id", "es_comuna_1", "poblacion"],
}

COLORES_CAT = {
    "Casos propios":     "#E74C3C",
    "Vecindad espacial": "#9B59B6",
    "Temperatura":       "#E67E22",
    "Precipitación":     "#3498DB",
    "Humedad/Calor":     "#1ABC9C",
    "Estacionalidad":    "#F39C12",
    "Geografía":         "#95A5A6",
}


def categorizar_feature(nombre):
    for cat, features in CATEGORIAS.items():
        if nombre in features:
            return cat
    return "Otras"


def limpiar_nombre(nombre):
    n = nombre.replace("_norm", "").replace("_mean", " mean")
    n = n.replace("_anomaly", " anomaly")
    if "_lag" in n:
        partes = n.split("_lag")
        n = partes[0].replace("_", " ") + f" (lag {partes[1]})"
    else:
        n = n.replace("_", " ")
    return n


# =============================================================================
# PASO 1: CARGA DE DATOS Y MODELOS
# =============================================================================

def cargar_datos_y_modelos():
    logger.info("--- PASO 1: Cargando datos y modelos ---")

    train_path = PROCESSED_DIR / "train_augmented.parquet"
    if not train_path.exists():
        train_path = PROCESSED_DIR / "train.parquet"
        logger.warning("  train_augmented.parquet no encontrado — usando train.parquet")

    df_train = pd.read_parquet(train_path)
    logger.info("  Train: %d filas | comunas: %d | features disponibles: ~%d",
                len(df_train), df_train["comuna_id"].nunique(), len(df_train.columns))

    modelos = {}

    for h in HORIZONTES_XGB:
        nombre = "xgboost_semana_actual" if h == 0 else f"xgboost_h{h}"
        path   = MODELS_DIR / f"{nombre}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                modelos[f"xgb_h{h}"] = pickle.load(f)
            logger.info("  ✓ XGBoost h=%d cargado", h)
        else:
            logger.warning("  ✗ XGBoost h=%d no encontrado: %s", h, path)

    feat_path = MODELS_DIR / "xgboost_features_v2.pkl"
    if feat_path.exists():
        with open(feat_path, "rb") as f:
            modelos["xgb_features"] = pickle.load(f)
        logger.info("  ✓ Features XGBoost cargadas: %d", len(modelos["xgb_features"]))

    for h in HORIZONTES_LSTM:
        path = MODELS_DIR / f"lstm_lstm_simple_h{h}.keras"
        if path.exists():
            modelos[f"lstm_h{h}"] = tf.keras.models.load_model(path)
            logger.info("  ✓ LSTM h=%d cargado", h)
        else:
            logger.warning("  ✗ LSTM h=%d no encontrado: %s", h, path)

    scaler_path = MODELS_DIR / "lstm_target_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            modelos["lstm_scaler"] = pickle.load(f)

    return df_train, modelos


# =============================================================================
# PASO 2: IMPORTANCIA NATIVA DE XGBOOST
# =============================================================================

def importancia_xgboost(modelos):
    logger.info("--- PASO 2: Calculando importancia XGBoost (gain) ---")

    features_guardadas = modelos.get("xgb_features", FEATURES_XGB)
    resultados = []
    # ► CAMBIO 1: agregado h=2 al diccionario de labels
    h_labels = {0: "actual", 1: "h=1", 2: "h=2", 3: "h=3", 4: "h=4"}

    for h in HORIZONTES_XGB:
        xgb_key = f"xgb_h{h}"
        if xgb_key not in modelos:
            logger.warning("  XGBoost h=%d no disponible — saltando", h)
            continue

        modelo = modelos[xgb_key]

        try:
            imp_dict = modelo.get_booster().get_score(importance_type="gain")
            imp_real = {}
            for k, v in imp_dict.items():
                if k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if idx < len(features_guardadas):
                        imp_real[features_guardadas[idx]] = v
                else:
                    imp_real[k] = v
        except Exception:
            logger.info("  Usando feature_importances_ como fallback para h=%d", h)
            imp_arr  = modelo.feature_importances_
            imp_real = {
                features_guardadas[i]: float(imp_arr[i])
                for i in range(min(len(features_guardadas), len(imp_arr)))
            }

        total = sum(imp_real.values())
        if total > 0:
            imp_real = {k: v / total * 100 for k, v in imp_real.items()}

        n_positivas = sum(1 for v in imp_real.values() if v > 0)
        logger.info("  h=%d: %d features con importancia > 0 | top: %s (%.1f%%)",
                    h, n_positivas,
                    max(imp_real, key=imp_real.get) if imp_real else "n/a",
                    max(imp_real.values()) if imp_real else 0)

        for feat, imp in imp_real.items():
            resultados.append({
                "Feature":     feat,
                "Importancia": round(imp, 3),
                "Horizonte":   h,
                "Label":       h_labels.get(h, f"h={h}"),
                "Categoria":   categorizar_feature(feat),
            })

    return pd.DataFrame(resultados)


# =============================================================================
# PASO 3: SHAP VALUES PARA EL LSTM  — GradientExplainer
#
# ¿POR QUÉ GRADIENTEXPLAINER?
# ----------------------------
# DeepExplainer falla con LSTM en TF2 porque intenta trazar el gradiente
# de una operación interna llamada TensorListStack, que no está registrada
# en el registro de gradientes de TF.
#
# GradientExplainer evita ese problema porque usa tf.GradientTape
# directamente sobre la función de forward pass, sin necesidad de acceder
# a operaciones internas. Funciona con cualquier modelo de Keras.
#
# La diferencia matemática es mínima para nuestro propósito:
# - DeepExplainer: SHAP via integrated gradients con muestras de background
# - GradientExplainer: gradiente esperado (esperanza de gradiente × input)
# Ambos miden la sensibilidad del output a cada feature de entrada.
# =============================================================================

def construir_secuencias_para_shap(df, features, horizonte, max_muestras=500):
    target_col = "confirmed_cases" if horizonte == 0 else f"target_h{horizonte}"
    if target_col not in df.columns:
        logger.warning("  Columna %s no encontrada en el dataset", target_col)
        return None, None

    X_list, y_list = [], []

    for comuna_id in sorted(df["comuna_id"].unique()):
        mask  = df["comuna_id"] == comuna_id
        df_c  = df[mask].sort_values(["year", "epi_week"]).copy()
        X_c   = df_c[features].fillna(0).values
        y_c   = df_c[target_col].values
        n     = len(df_c)

        for t in range(WINDOW_SIZE, n):
            val = y_c[t]
            if not np.isnan(val):
                X_list.append(X_c[t - WINDOW_SIZE : t])
                y_list.append(val)

    if not X_list:
        return None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.float32)

    if len(X) > max_muestras:
        idx = np.random.choice(len(X), max_muestras, replace=False)
        X   = X[idx]
        y   = y[idx]

    logger.info("  Secuencias construidas: %d muestras × %d semanas × %d features",
                len(X), WINDOW_SIZE, len(features))
    return X, y


def calcular_shap_lstm(modelos, df_train):
    """
    Calcula los SHAP values para los modelos LSTM h=3 y h=4.

    CAMBIO v2: usa GradientExplainer en lugar de DeepExplainer.
    GradientExplainer es compatible con LSTM en TF2 porque no requiere
    acceder a operaciones internas del grafo computacional.

    Si GradientExplainer también falla, el fallback de permutation
    importance sigue disponible como último recurso.
    """
    try:
        import shap
        logger.info("  SHAP disponible — versión %s", shap.__version__)
    except ImportError:
        logger.error("  SHAP no instalado.")
        logger.error("  Instalá con: pip install shap --break-system-packages")
        return pd.DataFrame()

    logger.info("--- PASO 3: Calculando SHAP values para LSTM (GradientExplainer) ---")
    logger.info("  Este proceso tarda ~20 min por horizonte en CPU. Paciencia...")

    features   = [f for f in FEATURES_LSTM if f in df_train.columns]
    resultados = []
    h_labels   = {3: "h=3", 4: "h=4"}

    for horizonte in HORIZONTES_LSTM:
        lstm_key = f"lstm_h{horizonte}"
        if lstm_key not in modelos:
            logger.warning("  LSTM h=%d no disponible — saltando", horizonte)
            continue

        logger.info("  Procesando LSTM h=%d...", horizonte)
        modelo = modelos[lstm_key]

        X_all, _ = construir_secuencias_para_shap(
            df_train, features, horizonte, max_muestras=N_BACKGROUND + N_EXPLICAR
        )
        if X_all is None:
            logger.warning("  Sin secuencias para h=%d", horizonte)
            continue

        idx_todos   = np.arange(len(X_all))
        idx_bg      = np.random.choice(idx_todos,
                                        min(N_BACKGROUND, len(X_all) // 2),
                                        replace=False)
        idx_exp_set = set(idx_bg)
        idx_exp     = np.array([i for i in idx_todos if i not in idx_exp_set])

        if len(idx_exp) > N_EXPLICAR:
            idx_exp = np.random.choice(idx_exp, N_EXPLICAR, replace=False)

        X_background = X_all[idx_bg]
        X_explain    = X_all[idx_exp]

        logger.info("  Background: %d muestras | A explicar: %d muestras",
                    len(X_background), len(X_explain))

        try:
            # ► CAMBIO PRINCIPAL: GradientExplainer en lugar de DeepExplainer
            #
            # GradientExplainer recibe el modelo de Keras directamente y
            # usa tf.GradientTape internamente, evitando el error de
            # TensorListStack que ocurría con DeepExplainer.
            #
            # La API es idéntica: explainer.shap_values(X_explain)
            # devuelve un array (muestras, timesteps, features) igual que antes.
            explainer   = shap.GradientExplainer(modelo, X_background)
            shap_values = explainer.shap_values(X_explain)

            # Manejar distintas formas del array de salida
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if shap_values.ndim == 4:
                shap_values = shap_values[:, :, :, 0]

            # shap_values: (muestras, 12_semanas, n_features)
            # Promediar sobre la dimensión temporal → (muestras, n_features)
            shap_abs = np.abs(shap_values).mean(axis=1)

            # Importancia global = media del |SHAP| sobre todas las muestras
            importancia_global = shap_abs.mean(axis=0)  # (n_features,)

            total   = importancia_global.sum()
            imp_pct = importancia_global / total * 100 if total > 0 else importancia_global

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            np.save(MODELS_DIR / f"shap_lstm_h{horizonte}.npy", shap_values)

            logger.info("  SHAP h=%d completado | top feature: %s (%.1f%%)",
                        horizonte,
                        features[importancia_global.argmax()],
                        imp_pct.max())

            for i, feat in enumerate(features):
                resultados.append({
                    "Feature":     feat,
                    "Importancia": round(float(imp_pct[i]), 3),
                    "Horizonte":   horizonte,
                    "Label":       h_labels.get(horizonte, f"h={horizonte}"),
                    "Categoria":   categorizar_feature(feat),
                    "Metodo":      "SHAP GradientExplainer",
                })

        except Exception as e:
            # FALLBACK: permutation importance
            logger.warning("  GradientExplainer falló para h=%d: %s", horizonte, str(e))
            logger.info("  Usando permutation importance como fallback...")

            baseline = modelo.predict(X_explain, verbose=0).flatten()
            importancias = []

            for j in range(len(features)):
                X_perm = X_explain.copy()
                X_perm[:, :, j] = np.random.permutation(X_perm[:, :, j].flatten()).reshape(
                    X_perm[:, :, j].shape
                )
                perm_pred = modelo.predict(X_perm, verbose=0).flatten()
                cambio    = np.mean((perm_pred - baseline) ** 2)  # MSE del cambio
                importancias.append(cambio)

                if (j + 1) % 10 == 0:
                    logger.info("  Permutation: %d/%d features procesadas",
                                j + 1, len(features))

            importancias = np.array(importancias)
            total        = importancias.sum()
            imp_pct      = importancias / total * 100 if total > 0 else importancias

            top_feat = features[importancias.argmax()]
            logger.info("  Permutation importance h=%d completado | top: %s (%.1f%%)",
                        horizonte, top_feat, imp_pct.max())

            for i, feat in enumerate(features):
                resultados.append({
                    "Feature":     feat,
                    "Importancia": round(float(imp_pct[i]), 3),
                    "Horizonte":   horizonte,
                    "Label":       h_labels.get(horizonte, f"h={horizonte}"),
                    "Categoria":   categorizar_feature(feat),
                    "Metodo":      "Permutation Importance (fallback)",
                })

    return pd.DataFrame(resultados)


# =============================================================================
# PASO 4: TABLAS DE RESULTADOS
# =============================================================================

def tabla_top_features(df_imp_xgb, df_imp_shap):
    print("\n" + "=" * 72)
    print("  TOP FEATURES POR MODELO Y HORIZONTE")
    print("  (% de importancia sobre el total de features usadas)")
    print("=" * 72)

    # ► CAMBIO 2: h=2 agregado al diccionario de labels
    h_labels = {0: "Semana actual", 1: "h=1 (1 sem)", 2: "h=2 (2 sem)",
                3: "h=3 (3 sem)", 4: "h=4 (4 sem)"}

    print("\n  === XGBoost (importancia gain) ===")
    for h in HORIZONTES_XGB:
        df_h = df_imp_xgb[df_imp_xgb["Horizonte"] == h].nlargest(10, "Importancia")
        if df_h.empty:
            continue
        print(f"\n  {h_labels.get(h, f'h={h}')}:")
        print(f"  {'Variable':<38} {'Importancia':>12}  {'Categoría'}")
        print("  " + "-" * 65)
        for _, row in df_h.iterrows():
            print(f"  {limpiar_nombre(row['Feature']):<38} "
                  f"{row['Importancia']:>10.1f}%  [{row['Categoria']}]")

    if not df_imp_shap.empty:
        metodo = df_imp_shap["Metodo"].iloc[0] if "Metodo" in df_imp_shap else "SHAP"
        print(f"\n  === LSTM ({metodo}) ===")
        for h in HORIZONTES_LSTM:
            df_h = df_imp_shap[df_imp_shap["Horizonte"] == h].nlargest(10, "Importancia")
            if df_h.empty:
                continue
            print(f"\n  {h_labels.get(h, f'h={h}')}:")
            print(f"  {'Variable':<38} {'Importancia':>12}  {'Categoría'}")
            print("  " + "-" * 65)
            for _, row in df_h.iterrows():
                print(f"  {limpiar_nombre(row['Feature']):<38} "
                      f"{row['Importancia']:>10.1f}%  [{row['Categoria']}]")

    print("=" * 72 + "\n")


def reporte_hallazgos(df_imp_xgb, df_imp_shap):
    print("\n" + "=" * 65)
    print("  HALLAZGOS EPIDEMIOLÓGICOS — IMPORTANCIA DE FEATURES")
    print("=" * 65)

    # ► CAMBIO 3: h=2 en el reporte de hallazgos
    h_labels = {0: "actual", 1: "h=1", 2: "h=2", 3: "h=3", 4: "h=4"}

    print("\n  Feature más importante por horizonte (XGBoost):")
    for h in HORIZONTES_XGB:
        df_h = df_imp_xgb[df_imp_xgb["Horizonte"] == h]
        if df_h.empty:
            continue
        top = df_h.nlargest(1, "Importancia").iloc[0]
        print(f"    {h_labels[h]:>8}: {limpiar_nombre(top['Feature']):<30} "
              f"{top['Importancia']:.1f}%  [{top['Categoria']}]")

    print("\n  Importancia acumulada por categoría (XGBoost):")
    df_cat = df_imp_xgb.groupby(["Horizonte", "Categoria"])["Importancia"].sum()
    for h in HORIZONTES_XGB:
        if h not in df_cat:
            continue
        top3  = df_cat[h].sort_values(ascending=False).head(3)
        linea = f"    {h_labels[h]:>8}: "
        linea += " | ".join(f"{cat} {val:.0f}%" for cat, val in top3.items())
        print(linea)

    print("\n  ¿Cambia la importancia relativa entre horizontes?")
    for h_corto, h_largo in [(0, 4), (1, 4)]:
        if not all(h in df_imp_xgb["Horizonte"].values for h in [h_corto, h_largo]):
            continue
        df_corto = df_imp_xgb[df_imp_xgb["Horizonte"] == h_corto]
        df_largo = df_imp_xgb[df_imp_xgb["Horizonte"] == h_largo]
        cat_c    = df_corto.groupby("Categoria")["Importancia"].sum().idxmax()
        cat_l    = df_largo.groupby("Categoria")["Importancia"].sum().idxmax()
        if cat_c != cat_l:
            print(f"    {h_labels[h_corto]} → {h_labels[h_largo]}: "
                  f"domina '{cat_c}' → domina '{cat_l}'")
        else:
            print(f"    {h_labels[h_corto]} → {h_labels[h_largo]}: "
                  f"'{cat_c}' domina en ambos horizontes")

    print("=" * 65 + "\n")


# =============================================================================
# PASO 4: GRÁFICOS
# =============================================================================

def grafico_importancia_xgb(df_imp_xgb):
    # ► CAMBIO 4: h=2 en el gráfico de XGBoost
    h_labels = {0: "Semana actual", 1: "h=1 (1 semana)", 2: "h=2 (2 semanas)",
                3: "h=3 (3 semanas)", 4: "h=4 (4 semanas)"}
    horizontes_disp = [h for h in HORIZONTES_XGB
                       if h in df_imp_xgb["Horizonte"].values]
    n = len(horizontes_disp)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 9))
    if n == 1:
        axes = [axes]

    for ax, h in zip(axes, horizontes_disp):
        df_h = (df_imp_xgb[df_imp_xgb["Horizonte"] == h]
                .nlargest(TOP_N, "Importancia")
                .sort_values("Importancia"))

        colores = [COLORES_CAT.get(categorizar_feature(f), "#95A5A6")
                   for f in df_h["Feature"]]

        ax.barh(range(len(df_h)), df_h["Importancia"],
                color=colores, alpha=0.85, edgecolor="white", height=0.7)
        ax.set_yticks(range(len(df_h)))
        ax.set_yticklabels([limpiar_nombre(f) for f in df_h["Feature"]], fontsize=8)
        ax.set_xlabel("Importancia (%)", fontsize=9)
        ax.set_title(h_labels.get(h, f"h={h}"), fontweight="bold", fontsize=11)
        ax.grid(axis="x", alpha=0.3)

        for i, val in enumerate(df_h["Importancia"]):
            ax.text(val + 0.1, i, f"{val:.1f}%", va="center", fontsize=7, color="dimgray")

    leyenda = [mpatches.Patch(color=c, label=cat, alpha=0.85)
               for cat, c in COLORES_CAT.items()]
    axes[-1].legend(handles=leyenda, loc="lower right",
                    fontsize=8, title="Categoría", title_fontsize=8)

    plt.suptitle(
        f"Importancia de features — XGBoost (gain)\n"
        f"Top {TOP_N} features por horizonte · Sprint 6 / HU7",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "29_importancia_xgboost.png", dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("  Gráfico guardado: 29_importancia_xgboost.png")


def grafico_importancia_shap(df_imp_shap):
    if df_imp_shap.empty:
        logger.warning("  Sin datos SHAP para graficar")
        return

    horizontes_disp = sorted(df_imp_shap["Horizonte"].unique())
    n        = len(horizontes_disp)
    h_labels = {3: "h=3 (3 semanas)", 4: "h=4 (4 semanas)"}

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 9))
    if n == 1:
        axes = [axes]

    for ax, h in zip(axes, horizontes_disp):
        df_h = (df_imp_shap[df_imp_shap["Horizonte"] == h]
                .nlargest(TOP_N, "Importancia")
                .sort_values("Importancia"))

        colores = [COLORES_CAT.get(categorizar_feature(f), "#95A5A6")
                   for f in df_h["Feature"]]

        ax.barh(range(len(df_h)), df_h["Importancia"],
                color=colores, alpha=0.85, edgecolor="white", height=0.7)
        ax.set_yticks(range(len(df_h)))
        ax.set_yticklabels([limpiar_nombre(f) for f in df_h["Feature"]], fontsize=8)
        ax.set_xlabel("Importancia SHAP (%)", fontsize=9)
        ax.set_title(h_labels.get(h, f"h={h}"), fontweight="bold", fontsize=11)
        ax.grid(axis="x", alpha=0.3)

        for i, val in enumerate(df_h["Importancia"]):
            ax.text(val + 0.1, i, f"{val:.1f}%", va="center", fontsize=7, color="dimgray")

    leyenda = [mpatches.Patch(color=c, label=cat, alpha=0.85)
               for cat, c in COLORES_CAT.items()]
    axes[-1].legend(handles=leyenda, loc="lower right",
                    fontsize=8, title="Categoría", title_fontsize=8)

    metodo = df_imp_shap["Metodo"].iloc[0] if "Metodo" in df_imp_shap else "SHAP"
    plt.suptitle(
        f"Importancia de features — LSTM ({metodo})\n"
        f"Top {TOP_N} features por horizonte · Sprint 6 / HU7",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "30_importancia_shap_lstm.png", dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("  Gráfico guardado: 30_importancia_shap_lstm.png")


def grafico_por_categoria(df_imp_xgb, df_imp_shap):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel izquierdo: XGBoost — ahora incluye h=2
    ax = axes[0]
    df_xgb_cat = (df_imp_xgb
                  .groupby(["Horizonte", "Categoria"])["Importancia"]
                  .sum()
                  .reset_index())
    horiz_xgb = sorted(df_xgb_cat["Horizonte"].unique())
    x      = np.arange(len(horiz_xgb))
    bottom = np.zeros(len(horiz_xgb))

    for cat in COLORES_CAT.keys():
        vals = []
        for h in horiz_xgb:
            sub = df_xgb_cat[
                (df_xgb_cat["Horizonte"] == h) & (df_xgb_cat["Categoria"] == cat)
            ]["Importancia"]
            vals.append(float(sub.values[0]) if len(sub) else 0.0)
        ax.bar(x, vals, bottom=bottom,
               label=cat, color=COLORES_CAT[cat], alpha=0.85)
        bottom += np.array(vals)

    # ► CAMBIO 5: labels del eje X incluyen h=2
    ax.set_xticks(x)
    ax.set_xticklabels(["actual", "h=1", "h=2", "h=3", "h=4"], fontsize=10)
    ax.set_ylabel("Importancia acumulada (%)", fontsize=10)
    ax.set_title("XGBoost\nImportancia por categoría y horizonte", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", title="Categoría")
    ax.grid(axis="y", alpha=0.3)

    # Panel derecho: LSTM SHAP
    ax = axes[1]
    if not df_imp_shap.empty:
        df_shap_cat = (df_imp_shap
                       .groupby(["Horizonte", "Categoria"])["Importancia"]
                       .sum()
                       .reset_index())
        horiz_shap = sorted(df_shap_cat["Horizonte"].unique())
        x      = np.arange(len(horiz_shap))
        bottom = np.zeros(len(horiz_shap))

        for cat in COLORES_CAT.keys():
            vals = []
            for h in horiz_shap:
                sub = df_shap_cat[
                    (df_shap_cat["Horizonte"] == h) & (df_shap_cat["Categoria"] == cat)
                ]["Importancia"]
                vals.append(float(sub.values[0]) if len(sub) else 0.0)
            ax.bar(x, vals, bottom=bottom,
                   label=cat, color=COLORES_CAT[cat], alpha=0.85)
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(["h=3", "h=4"], fontsize=10)
        ax.set_ylabel("Importancia SHAP acumulada (%)", fontsize=10)
        ax.set_title("LSTM (SHAP values)\nImportancia por categoría y horizonte",
                     fontweight="bold")
        ax.legend(fontsize=8, loc="upper right", title="Categoría")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "SHAP no disponible",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("LSTM — SHAP no disponible")

    plt.suptitle(
        "¿Qué tipo de variable importa más en cada horizonte?\n"
        "Sprint 6 / HU7 · Importancia por categoría epidemiológica",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "31_importancia_por_categoria.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("  Gráfico guardado: 31_importancia_por_categoria.png")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_importancia():
    print("\n" + "=" * 65)
    print("  SPRINT 6 / HU7 — Importancia de features v2")
    print("  XGBoost: importancia nativa (gain) — h=0,1,2,3,4")
    print("  LSTM:    SHAP GradientExplainer — h=3,4")
    print(f"  Top {TOP_N} features por modelo y horizonte")
    print("=" * 65 + "\n")

    df_train, modelos = cargar_datos_y_modelos()

    df_imp_xgb = importancia_xgboost(modelos)
    if df_imp_xgb.empty:
        logger.error("  Sin datos de importancia XGBoost — verificar modelos")
        return pd.DataFrame(), pd.DataFrame()

    df_imp_shap = calcular_shap_lstm(modelos, df_train)

    tabla_top_features(df_imp_xgb, df_imp_shap)
    reporte_hallazgos(df_imp_xgb, df_imp_shap)

    grafico_importancia_xgb(df_imp_xgb)
    if not df_imp_shap.empty:
        grafico_importancia_shap(df_imp_shap)
    grafico_por_categoria(df_imp_xgb, df_imp_shap)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_imp_xgb.to_csv(REPORTS_DIR / "importancia_xgboost.csv", index=False)
    logger.info("  importancia_xgboost.csv guardado")

    if not df_imp_shap.empty:
        df_imp_shap.to_csv(REPORTS_DIR / "importancia_shap_lstm.csv", index=False)
        logger.info("  importancia_shap_lstm.csv guardado")

    print("\n✓ Importancia de features completada.")
    print("  Archivos generados:")
    print("  - reports/importancia_xgboost.csv")
    print("  - reports/importancia_shap_lstm.csv  (si SHAP disponible)")
    print("  - reports/figures/29_importancia_xgboost.png")
    print("  - reports/figures/30_importancia_shap_lstm.png")
    print("  - reports/figures/31_importancia_por_categoria.png")

    return df_imp_xgb, df_imp_shap


if __name__ == "__main__":
    df_xgb, df_shap = run_importancia()
