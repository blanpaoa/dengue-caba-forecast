"""
================================================================================
Sprint 6 / HU7 — Evaluación exhaustiva por comuna y horizonte
================================================================================

¿QUÉ HACE ESTE SCRIPT?
-----------------------
Hasta ahora evaluamos los modelos con una sola métrica global — el MAE promedio
sobre las 390 filas del período de validación. Eso oculta información importante:

  ¿Hay comunas donde el modelo funciona mucho mejor o peor?
  ¿La Comuna 1 (la más afectada por el brote) es más difícil de predecir?
  ¿El modelo ganador es siempre el mismo en todas las comunas?
  ¿En qué horizonte cada modelo supera a los demás?

Este script responde esas preguntas evaluando los modelos de forma DESAGREGADA:
  - Por cada una de las 15 comunas de CABA
  - Por cada horizonte de predicción (semana actual, h=1, h=2, h=3, h=4)
  - Sobre los tres conjuntos de datos (train, validation, test)

MODELOS EVALUADOS:
  1. Persistencia (lag 1):  predice que habrá los mismos casos que la semana pasada
  2. Random Forest:         mejor modelo del Sprint 4
  3. XGBoost:               mejor modelo de árboles del Sprint 5
  4. LSTM simple + aug:     mejor modelo de redes neuronales del Sprint 5

¿POR QUÉ EVALUAR POR COMUNA?
------------------------------
Las 15 comunas de CABA tienen características muy diferentes:
  - Población: desde 22.000 hab (C1) hasta 220.000 hab (C9)
  - Historial de brotes: C1 concentró el 39.9% de todos los casos históricos
  - Posición geográfica: comunas costeras vs. interiores

Es posible que el LSTM sea excelente para predecir el brote de la C1 pero
peor que el Random Forest para predecir las comunas del oeste de CABA.
Identificar estos patrones permite ajustar el sistema de alertas por zona.

OUTPUTS GENERADOS:
  - Tabla de MAE promedio por modelo y horizonte
  - Tabla de ganadores: qué modelo gana en cada (comuna × horizonte)
  - Heatmap de MAE por comuna y horizonte (mapa de calor)
  - Gráfico de barras por comuna para h=1 y h=4
  - Gráfico de victorias por modelo
  - CSV con todas las métricas desagregadas

PREREQUISITO:
  Ejecutar todos los scripts de modelado del Sprint 5 antes de este.
  Requiere los archivos .pkl y .keras en models/saved/.
"""

import os
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Silenciar warnings de TensorFlow
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

TRAIN_FILE = PROCESSED_DIR / "train.parquet"
VAL_FILE   = PROCESSED_DIR / "validation.parquet"
TEST_FILE  = PROCESSED_DIR / "test.parquet"

# Ventana temporal del LSTM — debe coincidir con lstm_model.py
WINDOW_SIZE = 12

# Horizontes a evaluar (0 = semana actual, 1-4 = semanas futuras)
HORIZONTES = [0, 1, 2, 3, 4]

# Nombres descriptivos de las 15 comunas para los gráficos
# Fuente: mapa oficial de comunas de CABA
NOMBRES_COMUNAS = {
    1:  "C1 Puerto Madero",   2:  "C2 Recoleta",
    3:  "C3 Balvanera",       4:  "C4 La Boca",
    5:  "C5 Almagro",         6:  "C6 Caballito",
    7:  "C7 Flores",          8:  "C8 Lugano",
    9:  "C9 Liniers",         10: "C10 Floresta",
    11: "C11 Villa del Parque", 12: "C12 Coghlan",
    13: "C13 Belgrano",       14: "C14 Palermo",
    15: "C15 Agronomía",
}

# =============================================================================
# FEATURES — deben coincidir exactamente con los scripts de modelado
#
# Es crítico que las listas de features sean idénticas a las usadas durante
# el entrenamiento. Si usamos features diferentes al predecir, el modelo
# recibirá inputs que nunca vio y las predicciones serán incorrectas.
# =============================================================================

# Features del LSTM (versiones normalizadas _norm)
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

# Features del XGBoost (versiones sin normalizar — XGBoost no las necesita)
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

# Colores consistentes con los gráficos de los sprints anteriores
COLORES_MODELO = {
    "Persistencia":  "#F39C12",   # naranja
    "Random Forest": "#3498DB",   # azul
    "XGBoost":       "#E74C3C",   # rojo
    "LSTM+aug":      "#8E44AD",   # violeta
}


# =============================================================================
# PASO 1: CARGA DE DATOS Y MODELOS
#
# Cargamos los tres conjuntos de datos y todos los modelos entrenados.
# Los modelos se cargan una sola vez para no repetir el proceso en cada
# combinación de (comuna, horizonte) — sería muy lento.
# =============================================================================

def cargar_datos():
    """
    Carga los tres conjuntos de datos generados en lags.py.

    Train:      2023 completo (780 filas originales)
    Validation: 2024 semanas 1-26 — el brote masivo (390 filas)
    Test:       2024 sem 27-52 + 2025 (1170 filas)
    """
    logger.info("--- PASO 1: Cargando datos ---")

    for f in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not f.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {f}\n"
                "Ejecutá src/features/lags.py primero."
            )

    df_train = pd.read_parquet(TRAIN_FILE)
    df_val   = pd.read_parquet(VAL_FILE)
    df_test  = pd.read_parquet(TEST_FILE)

    logger.info("  Train: %d filas | Validation: %d | Test: %d",
                len(df_train), len(df_val), len(df_test))
    logger.info("  Comunas: %d | Años: %s",
                df_train["comuna_id"].nunique(),
                sorted(pd.concat([df_train, df_val, df_test])["year"].unique()))
    return df_train, df_val, df_test


def cargar_modelos():
    """
    Carga todos los modelos entrenados del Sprint 5 desde disco.

    ¿Por qué cargamos desde disco en lugar de reentrenar?
    Los modelos ya fueron entrenados y guardados en el Sprint 5.
    Cargarlos desde disco es instantáneo — reentrenar llevaría horas.

    Retorna un diccionario con todos los modelos indexados por nombre.
    Si un modelo no existe, lo omite y avisa con un warning.
    """
    logger.info("--- Cargando modelos entrenados del Sprint 5 ---")
    modelos = {}

    # Random Forest — un solo modelo para semana actual
    rf_path = MODELS_DIR / "random_forest.pkl"
    if rf_path.exists():
        with open(rf_path, "rb") as f:
            modelos["rf"] = pickle.load(f)
        logger.info("  ✓ Random Forest cargado")
    else:
        logger.warning("  ✗ Random Forest no encontrado: %s", rf_path)

    # XGBoost — un modelo por horizonte (semana actual + h=1,2,3,4)
    for h in HORIZONTES:
        nombre = "xgboost_semana_actual" if h == 0 else f"xgboost_h{h}"
        path   = MODELS_DIR / f"{nombre}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                modelos[f"xgb_h{h}"] = pickle.load(f)
        else:
            logger.warning("  ✗ XGBoost h=%d no encontrado", h)

    # Features usadas por XGBoost (guardadas durante el entrenamiento)
    # Features del Random Forest (50 features — Sprint 4, sin vecindad espacial)
    # El RF fue entrenado antes de agregar casos_vecinas_lag1 en Sprint 5
    rf_feat_path = MODELS_DIR / "random_forest_features.pkl"
    if rf_feat_path.exists():
        with open(rf_feat_path, "rb") as f:
            modelos["rf_features"] = pickle.load(f)
        logger.info("  RF features cargadas: %d", len(modelos["rf_features"]))
    else:
        logger.info("  RF features no encontradas — se reconstruyen automaticamente")

    # Features del XGBoost v2 (52 features — Sprint 5, con vecindad)
    feat_path = MODELS_DIR / "xgboost_features_v2.pkl"
    if feat_path.exists():
        with open(feat_path, "rb") as f:
            modelos["xgb_features"] = pickle.load(f)

    # LSTM simple + augmentation — un modelo por horizonte
    for h in HORIZONTES:
        path = MODELS_DIR / f"lstm_lstm_simple_h{h}.keras"
        if path.exists():
            modelos[f"lstm_h{h}"] = tf.keras.models.load_model(path)
        else:
            logger.warning("  ✗ LSTM h=%d no encontrado", h)

    # Normalizador del target del LSTM (necesario para desnormalizar predicciones)
    scaler_path = MODELS_DIR / "lstm_target_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            modelos["lstm_scaler"] = pickle.load(f)

    n_cargados = sum(1 for k in modelos if k not in ["xgb_features", "lstm_scaler"])
    logger.info("  Total modelos cargados: %d", n_cargados)
    return modelos


# =============================================================================
# PASO 2: GENERAR PREDICCIONES POR MODELO
#
# Cada función de predicción devuelve tres valores:
#   y_real: los casos reales observados
#   y_pred: los casos predichos por el modelo
#   idx:    los índices del dataset donde hay predicciones válidas
#
# Los índices son importantes para luego filtrar por comuna.
# =============================================================================

def predecir_persistencia(df):
    """
    El modelo de persistencia predice que esta semana habrá los mismos
    casos que la semana pasada: pred(t) = casos(t-1) = cases_lag1.

    Es el modelo de referencia epidemiológica más simple posible.
    Su fortaleza es que durante un brote activo, los casos de hoy son
    un buen predictor de los de mañana (alta autocorrelación).

    Solo tiene sentido como predictor de h=1 (1 semana adelante).
    Para otros horizontes lo usamos como referencia de comparación.
    """
    mask   = df["cases_lag1"].notna() & df["confirmed_cases"].notna()
    y_real = df.loc[mask, "confirmed_cases"].values
    y_pred = df.loc[mask, "cases_lag1"].fillna(0).values
    idx    = df[mask].index.tolist()
    return y_real, y_pred, idx


def predecir_rf(df, modelos):
    """
    Random Forest predice los casos de la semana actual (confirmed_cases).

    El RF recibe una fila por semana con todas las features y devuelve
    una predicción. No necesita normalización porque los árboles de
    decisión son invariantes a la escala de las variables.

    Solo evalúa semana actual (h=0) — el RF del Sprint 4 no fue entrenado
    para horizontes futuros.

    IMPORTANTE: el RF fue entrenado en el Sprint 4 con 50 features —
    ANTES de agregar casos_vecinas_lag1 e incidencia_vecinas_lag1 en Sprint 5.
    Por eso usamos las features guardadas durante su entrenamiento
    (rf_features.pkl) y no la lista global FEATURES_XGB (que tiene 52).
    Si no existe el archivo de features, reconstruimos las 50 originales
    excluyendo las dos features de vecindad.
    """
    if "rf" not in modelos:
        return None, None, None

    # Usar las features con las que fue entrenado el RF (50, sin vecindad)
    if "rf_features" in modelos:
        features = [f for f in modelos["rf_features"] if f in df.columns]
    else:
        # Reconstruir las 50 features originales excluyendo las de vecindad
        # que se agregaron en Sprint 5 después de entrenar el RF
        features = [f for f in FEATURES_XGB
                    if f in df.columns
                    and f not in ["casos_vecinas_lag1", "incidencia_vecinas_lag1"]]

    # Verificar que la cantidad de features coincide con lo esperado por el RF
    n_esperadas = modelos["rf"].n_features_in_
    if len(features) != n_esperadas:
        logger.warning(
            "  RF: features disponibles (%d) != features esperadas (%d). "
            "Ajustando automáticamente.",
            len(features), n_esperadas
        )
        # Tomar solo las primeras n_esperadas features que existan en el dataset
        features = features[:n_esperadas]

    mask   = df[features].notna().all(axis=1) & df["confirmed_cases"].notna()
    X      = df.loc[mask, features].values
    y_real = df.loc[mask, "confirmed_cases"].values
    y_pred = modelos["rf"].predict(X).clip(min=0)
    idx    = df[mask].index.tolist()
    return y_real, y_pred, idx


def predecir_xgboost(df, modelos, horizonte):
    """
    XGBoost predice los casos de la semana actual (h=0) o de h semanas
    adelante (h=1,2,3,4).

    El XGBoost fue entrenado con transformación logarítmica del target
    (log1p). Por eso las predicciones vienen en escala log y necesitamos
    aplicar la función inversa (expm1) para obtener casos reales.

    expm1(x) = e^x - 1  es la inversa exacta de log1p(x) = log(1+x).
    """
    xgb_key = f"xgb_h{horizonte}"
    if xgb_key not in modelos:
        return None, None, None

    # Usar las features guardadas del entrenamiento (o la lista por defecto)
    features = modelos.get("xgb_features", FEATURES_XGB)
    features = [f for f in features if f in df.columns]

    # El target depende del horizonte
    target_col = "confirmed_cases" if horizonte == 0 else f"target_h{horizonte}"
    if target_col not in df.columns:
        return None, None, None

    mask   = df[features].notna().all(axis=1) & df[target_col].notna()
    X      = df.loc[mask, features].values
    y_real = df.loc[mask, target_col].values

    # Predecir en escala log y desnormalizar con expm1
    pred_log = modelos[xgb_key].predict(X).clip(min=0)
    y_pred   = np.expm1(pred_log).clip(min=0)
    idx      = df[mask].index.tolist()
    return y_real, y_pred, idx


def construir_secuencias_lstm(df, features, horizonte=0):
    """
    Convierte el dataset plano en ventanas temporales para el LSTM.

    El LSTM no puede recibir filas individuales como XGBoost — necesita
    ver las últimas WINDOW_SIZE semanas en orden cronológico.

    Para cada semana t de cada comuna, construimos una ventana:
      Entrada: datos de semanas [t-12, ..., t-1]  → forma (12, n_features)
      Target:  casos en t (horizonte=0) o t+h (horizonte=h)

    Las ventanas se construyen DENTRO de cada comuna para no mezclar
    series temporales de comunas diferentes.

    Las primeras 12 semanas de cada comuna no tienen suficiente historia
    y se descartan automáticamente.
    """
    X_list, y_list, idx_list = [], [], []

    target_col = "confirmed_cases" if horizonte == 0 else f"target_h{horizonte}"
    if target_col not in df.columns:
        return None, None, None

    for comuna_id in sorted(df["comuna_id"].unique()):
        mask  = df["comuna_id"] == comuna_id
        df_c  = df[mask].sort_values(["year", "epi_week"]).copy()
        idx_c = df_c.index.tolist()
        X_c   = df_c[features].fillna(0).values
        y_c   = df_c[target_col].values
        n     = len(df_c)

        for t in range(WINDOW_SIZE, n):
            ventana    = X_c[t - WINDOW_SIZE : t]
            target_val = y_c[t]
            if not np.isnan(target_val):
                X_list.append(ventana)
                # Guardar el target en escala REAL de casos (sin normalizar)
                # La desnormalización la hace predecir_lstm() con el scaler del train
                y_list.append(target_val)
                idx_list.append(idx_c[t])

    if not X_list:
        return None, None, None

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list,  dtype=np.float32),
            idx_list)


def predecir_lstm(df, modelos, horizonte):
    """
    LSTM simple + augmentation — predice usando ventanas de 12 semanas.

    El proceso de predicción tiene tres etapas:
    1. Construir las ventanas de 12 semanas en orden cronológico por comuna.
    2. Obtener predicciones del modelo LSTM (en escala normalizada [0, 1]).
    3. Desnormalizar con el scaler guardado del entrenamiento para obtener casos reales.

    IMPORTANTE SOBRE LA NORMALIZACIÓN:
    El LSTM fue entrenado con un MinMaxScaler ajustado sobre el train set
    (máximo 649 casos → 1.0). Al predecir sobre validation o test, el modelo
    SIEMPRE devuelve valores en la escala [0, 1] del entrenamiento.
    Por eso usamos el scaler guardado del entrenamiento para desnormalizar
    — NO uno nuevo ajustado con validation o test.

    Si usáramos un scaler nuevo ajustado con validation (máximo 1391 casos),
    la desnormalización sería incorrecta y generaría valores imposibles.
    El target real (y_real) también se obtiene directamente del dataset
    en escala de casos reales — sin pasar por el scaler.
    """
    lstm_key = f"lstm_h{horizonte}"
    if lstm_key not in modelos or "lstm_scaler" not in modelos:
        return None, None, None

    features   = [f for f in FEATURES_LSTM if f in df.columns]
    target_col = "confirmed_cases" if horizonte == 0 else f"target_h{horizonte}"
    if target_col not in df.columns:
        return None, None, None

    # Scaler del entrenamiento — SIEMPRE usamos este para desnormalizar
    scaler = modelos["lstm_scaler"]

    # Construir secuencias temporales (las features ya están normalizadas en el dataset)
    X_seq, y_seq_real, idx_list = construir_secuencias_lstm(df, features, horizonte)
    if X_seq is None:
        return None, None, None

    # Predicción del LSTM en escala normalizada [0, 1]
    pred_norm = modelos[lstm_key].predict(X_seq, verbose=0).flatten()

    # Desnormalizar con el scaler del entrenamiento → casos reales
    # El scaler fue ajustado con train (max=649), por eso pred_norm=1.0 → 649 casos
    # Valores de validation que superan 1.0 (hasta 2.14) se desnormalizan
    # proporcionalmente: pred_norm=2.14 → 2.14 × 649 ≈ 1391 casos
    y_pred = scaler.inverse_transform(
        pred_norm.reshape(-1, 1)
    ).flatten().clip(min=0)

    # El target real viene directamente del dataset en escala de casos reales
    # NO se pasa por el scaler — ya está en la unidad correcta (casos/semana)
    y_real = y_seq_real.clip(min=0)

    return y_real, y_pred, idx_list


# =============================================================================
# CÁLCULO DE MÉTRICAS DE ERROR
#
# Para cada combinación (modelo, split, horizonte, comuna) calculamos:
#
# MAE  — Error promedio absoluto en casos reales.
#   "El modelo se equivoca en promedio X casos por semana en esta comuna."
#   Es la métrica principal de esta evaluación.
#
# RMSE — Raíz del error cuadrático medio. Penaliza más los errores grandes.
#   Útil para detectar semanas muy mal predichas (picos de brote).
#
# R²   — Fracción de la variación real que explica el modelo.
#   Puede ser negativo por distribution shift (ver informe de comparación final).
#
# Además registramos la media y el máximo de casos reales por comuna
# para contextualizar los errores absolutos.
# =============================================================================

def calcular_metricas(y_real, y_pred, modelo, split, horizonte, comuna_id):
    """
    Calcula MAE, RMSE y R² para una combinación específica.
    Retorna None si no hay datos disponibles.
    """
    if y_real is None or len(y_real) == 0:
        return None

    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2   = r2_score(y_real, y_pred)

    return {
        "Modelo":      modelo,
        "Split":       split,
        "Horizonte":   horizonte,
        "Comuna":      comuna_id,
        "Nombre":      NOMBRES_COMUNAS.get(comuna_id, f"C{comuna_id}"),
        "MAE":         round(mae, 2),
        "RMSE":        round(rmse, 2),
        "R2":          round(r2, 3),
        "N":           len(y_real),
        "media_real":  round(float(np.mean(y_real)), 2),
        "max_real":    round(float(np.max(y_real)), 1),
    }


# =============================================================================
# PASO 3: CALCULAR MÉTRICAS POR COMUNA
#
# Para cada split de datos (train, validation, test) y cada horizonte,
# generamos las predicciones de todos los modelos y luego las filtramos
# por comuna para calcular las métricas desagregadas.
#
# El filtrado por comuna se hace DESPUÉS de predecir para todos — así
# evitamos predecir 15 veces por modelo (una por comuna), lo que sería
# mucho más lento.
# =============================================================================

def evaluar_por_comuna(df, modelos, nombre_split):
    """
    Calcula métricas de todos los modelos para cada (comuna, horizonte).

    Proceso:
    1. Para cada horizonte, genera predicciones globales de los 4 modelos
    2. Filtra las predicciones por cada una de las 15 comunas
    3. Calcula MAE, RMSE y R² para cada combinación

    Retorna una lista de diccionarios con los resultados desagregados.
    """
    logger.info("  Evaluando: %s", nombre_split)
    resultados = []
    comunas    = sorted(df["comuna_id"].unique())

    for horizonte in HORIZONTES:

        # Generar predicciones globales para este horizonte
        # (se filtran por comuna en el siguiente paso)
        y_r_pers, y_p_pers, idx_pers = predecir_persistencia(df)
        y_r_rf,   y_p_rf,   idx_rf   = predecir_rf(df, modelos)
        y_r_xgb,  y_p_xgb,  idx_xgb  = predecir_xgboost(df, modelos, horizonte)
        y_r_lstm, y_p_lstm, idx_lstm  = predecir_lstm(df, modelos, horizonte)

        for comuna_id in comunas:

            # Helper interno: filtra las predicciones para una comuna específica
            def filtrar_comuna(y_r, y_p, idx):
                """
                Dado el vector de predicciones globales y los índices originales,
                devuelve solo las filas que corresponden a esta comuna.
                """
                if idx is None or y_r is None:
                    return None, None
                idx_arr    = np.array(idx)
                mask_c     = df.loc[idx_arr, "comuna_id"].values == comuna_id
                if mask_c.sum() == 0:
                    return None, None
                return y_r[mask_c], y_p[mask_c]

            # Calcular métricas para cada modelo en esta (comuna, horizonte)

            # 1. Persistencia — disponible para todos los horizontes
            yr, yp = filtrar_comuna(y_r_pers, y_p_pers, idx_pers)
            if yr is not None and len(yr) > 0:
                m = calcular_metricas(yr, yp, "Persistencia", nombre_split, horizonte, comuna_id)
                if m:
                    resultados.append(m)

            # 2. Random Forest — solo para semana actual (h=0)
            if horizonte == 0 and y_r_rf is not None:
                yr, yp = filtrar_comuna(y_r_rf, y_p_rf, idx_rf)
                if yr is not None and len(yr) > 0:
                    m = calcular_metricas(yr, yp, "Random Forest", nombre_split, horizonte, comuna_id)
                    if m:
                        resultados.append(m)

            # 3. XGBoost — disponible para todos los horizontes
            if y_r_xgb is not None:
                yr, yp = filtrar_comuna(y_r_xgb, y_p_xgb, idx_xgb)
                if yr is not None and len(yr) > 0:
                    m = calcular_metricas(yr, yp, "XGBoost", nombre_split, horizonte, comuna_id)
                    if m:
                        resultados.append(m)

            # 4. LSTM simple + augmentation — disponible para todos los horizontes
            if y_r_lstm is not None:
                yr, yp = filtrar_comuna(y_r_lstm, y_p_lstm, idx_lstm)
                if yr is not None and len(yr) > 0:
                    m = calcular_metricas(yr, yp, "LSTM+aug", nombre_split, horizonte, comuna_id)
                    if m:
                        resultados.append(m)

    logger.info("  Resultados calculados: %d", len(resultados))
    return resultados


# =============================================================================
# PASO 4: IDENTIFICAR MODELO GANADOR POR COMUNA Y HORIZONTE
#
# Para cada combinación (comuna, horizonte) identificamos qué modelo
# tiene el menor MAE — ese es el "ganador" de esa combinación.
#
# También marcamos con * las combinaciones donde el modelo ganador
# supera a la persistencia (el baseline epidemiológico más simple).
# =============================================================================

def tabla_ganadores(df_metricas, split="Validation"):
    """
    Para cada (comuna, horizonte) identifica el modelo con menor MAE.
    Muestra una tabla donde cada celda indica el modelo ganador.

    Leyenda de abreviaciones:
      PERS = Persistencia
      RF   = Random Forest
      XGB  = XGBoost
      LSTM = LSTM simple + augmentation
      *    = supera a la persistencia
    """
    print("\n" + "=" * 82)
    print(f"  MODELO GANADOR POR COMUNA Y HORIZONTE — {split}")
    print("  Criterio: menor MAE | * = supera a la persistencia")
    print("=" * 82)

    df_s   = df_metricas[df_metricas["Split"] == split].copy()
    horiz  = sorted(df_s["Horizonte"].unique())
    comunas = sorted(df_s["Comuna"].unique())

    # Encabezado de la tabla
    h_labels = ["actual" if h == 0 else f"h={h}" for h in horiz]
    print(f"\n  {'Comuna':<26} " + "  ".join(f"{h:>8}" for h in h_labels))
    print("  " + "-" * (26 + 10 * len(h_labels)))

    conteo = {m: 0 for m in ["Persistencia", "Random Forest", "XGBoost", "LSTM+aug"]}

    for comuna_id in comunas:
        nombre = NOMBRES_COMUNAS.get(comuna_id, f"C{comuna_id}")
        fila   = f"  {nombre:<26}"

        for h in horiz:
            sub = df_s[(df_s["Comuna"] == comuna_id) & (df_s["Horizonte"] == h)]
            if sub.empty:
                fila += f"  {'---':>8}"
                continue

            # Identificar el modelo con menor MAE
            idx_min  = sub["MAE"].idxmin()
            ganador  = sub.loc[idx_min, "Modelo"]
            mae_gan  = sub.loc[idx_min, "MAE"]

            # Verificar si supera a la persistencia
            mae_pers_arr = sub[sub["Modelo"] == "Persistencia"]["MAE"].values
            supera = (len(mae_pers_arr) > 0 and
                      mae_gan < mae_pers_arr[0] and
                      ganador != "Persistencia")

            # Abreviación para la tabla
            abrev = {"Persistencia": "PERS", "Random Forest": "RF",
                     "XGBoost": "XGB", "LSTM+aug": "LSTM"}
            label = abrev.get(ganador, ganador[:4])
            if supera:
                label += "*"

            fila += f"  {label:>8}"
            conteo[ganador] = conteo.get(ganador, 0) + 1

        print(fila)

    print("\n  CONTEO DE VICTORIAS POR MODELO (combinaciones ganadas):")
    total = sum(conteo.values())
    for modelo, n in sorted(conteo.items(), key=lambda x: -x[1]):
        pct   = n / total * 100 if total > 0 else 0
        barra = "█" * n
        print(f"    {modelo:<20}: {n:>3} ({pct:.1f}%)  {barra}")

    print("=" * 82 + "\n")
    return conteo


def tabla_mae_por_horizonte(df_metricas, split="Validation"):
    """
    Tabla de MAE promedio por modelo y horizonte, promediado sobre las 15 comunas.

    Esta tabla permite ver de un vistazo qué modelo gana en cada horizonte
    sin tener que revisar las 15 comunas individualmente.
    """
    print("\n" + "=" * 78)
    print(f"  MAE PROMEDIO SOBRE LAS 15 COMUNAS — {split}")
    print("  (menor es mejor — en casos por semana por comuna)")
    print("=" * 78)

    df_s  = df_metricas[df_metricas["Split"] == split]
    pivot = df_s.groupby(["Modelo", "Horizonte"])["MAE"].mean().round(2).unstack()

    print(f"\n  {'Modelo':<22}", end="")
    for h in sorted(pivot.columns):
        label = "actual" if h == 0 else f"h={h}"
        print(f"  {label:>8}", end="")
    print()
    print("  " + "-" * (22 + 10 * len(pivot.columns)))

    for modelo in ["Persistencia", "Random Forest", "XGBoost", "LSTM+aug"]:
        if modelo not in pivot.index:
            continue
        print(f"  {modelo:<22}", end="")
        for h in sorted(pivot.columns):
            val = pivot.loc[modelo, h] if h in pivot.columns else float("nan")
            if np.isnan(val):
                print(f"  {'---':>8}", end="")
            else:
                print(f"  {val:>8.2f}", end="")
        print()

    print("=" * 78 + "\n")


# =============================================================================
# PASO 5: GRÁFICOS
# =============================================================================

def heatmap_mae_por_comuna(df_metricas, split="Validation"):
    """
    Mapa de calor (heatmap) de MAE por comuna y horizonte.

    ¿Cómo leer este gráfico?
    Cada celda muestra el MAE de un modelo para una (comuna, horizonte).
    Colores fríos (verde) = error bajo = buen desempeño.
    Colores cálidos (rojo) = error alto = mal desempeño.

    Permite identificar de un vistazo:
    - Qué comunas son sistemáticamente más difíciles de predecir
    - En qué horizontes cada modelo tiene más dificultades
    - Si los errores se concentran en las comunas con más casos (C1, C9)
    """
    logger.info("--- Generando heatmaps de MAE por comuna ---")

    df_s    = df_metricas[df_metricas["Split"] == split]
    modelos = ["Persistencia", "Random Forest", "XGBoost", "LSTM+aug"]
    comunas = sorted(df_s["Comuna"].unique())
    horiz   = sorted(df_s["Horizonte"].unique())
    n_mod   = sum(1 for m in modelos if m in df_s["Modelo"].unique())

    fig, axes = plt.subplots(1, n_mod, figsize=(5 * n_mod, 8))
    if n_mod == 1:
        axes = [axes]

    ax_idx = 0
    for modelo in modelos:
        df_m = df_s[df_s["Modelo"] == modelo]
        if df_m.empty:
            continue

        ax = axes[ax_idx]
        ax_idx += 1

        # Construir la matriz de MAE: filas=comunas, columnas=horizontes
        matriz = np.full((len(comunas), len(horiz)), np.nan)
        for i, c in enumerate(comunas):
            for j, h in enumerate(horiz):
                sub = df_m[(df_m["Comuna"] == c) & (df_m["Horizonte"] == h)]
                if not sub.empty:
                    matriz[i, j] = sub["MAE"].values[0]

        # Usar el percentil 90 como máximo de la escala para no distorsionar
        # por valores extremos (por ejemplo el error enorme de la C1 durante el brote)
        vmax = np.nanpercentile(matriz, 90)
        im   = ax.imshow(matriz, cmap="RdYlGn_r", aspect="auto",
                         vmin=0, vmax=vmax)

        ax.set_xticks(range(len(horiz)))
        ax.set_xticklabels(["actual" if h == 0 else f"h={h}" for h in horiz],
                           fontsize=9)
        ax.set_yticks(range(len(comunas)))
        ax.set_yticklabels([NOMBRES_COMUNAS.get(c, f"C{c}") for c in comunas],
                           fontsize=8)
        ax.set_title(modelo, fontweight="bold", fontsize=11)
        plt.colorbar(im, ax=ax, label="MAE (casos)")

        # Anotar el valor de MAE en cada celda
        for i in range(len(comunas)):
            for j in range(len(horiz)):
                if not np.isnan(matriz[i, j]):
                    color_texto = "white" if matriz[i, j] > vmax * 0.65 else "black"
                    ax.text(j, i, f"{matriz[i,j]:.1f}",
                            ha="center", va="center",
                            fontsize=7, color=color_texto)

    plt.suptitle(
        f"MAE por comuna y horizonte — {split}\n"
        "Sprint 6 / HU7 · Verde=error bajo, Rojo=error alto",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / f"22_heatmap_mae_comunas_{split.lower()}.png",
                dpi=150, bbox_inches="tight")
    plt.show()


def grafico_mae_por_comuna(df_metricas, split="Validation", horizonte=4):
    """
    Gráfico de barras comparando los 4 modelos en cada una de las 15 comunas
    para un horizonte específico.

    Permite ver si un modelo domina en todas las comunas o si hay
    comunas donde un modelo diferente es mejor.
    """
    df_s = df_metricas[(df_metricas["Split"] == split) &
                       (df_metricas["Horizonte"] == horizonte)]
    if df_s.empty:
        return

    comunas = sorted(df_s["Comuna"].unique())
    modelos = [m for m in ["Persistencia", "Random Forest", "XGBoost", "LSTM+aug"]
               if m in df_s["Modelo"].unique()]
    n_mod   = len(modelos)
    x       = np.arange(len(comunas))
    ancho   = 0.8 / n_mod

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, modelo in enumerate(modelos):
        df_m = df_s[df_s["Modelo"] == modelo]
        maes = []
        for c in comunas:
            sub = df_m[df_m["Comuna"] == c]["MAE"]
            maes.append(sub.values[0] if not sub.empty else np.nan)

        ax.bar(x + i * ancho - (n_mod - 1) * ancho / 2,
               maes, ancho,
               label=modelo,
               color=COLORES_MODELO.get(modelo, "#95A5A6"),
               alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([NOMBRES_COMUNAS.get(c, f"C{c}").replace(" ", "\n")
                        for c in comunas],
                       fontsize=8, ha="center")
    ax.set_ylabel("MAE (casos promedio por semana)")
    ax.set_xlabel("Comuna")
    label_h = "semana actual" if horizonte == 0 else f"{horizonte} semanas adelante"
    ax.set_title(
        f"MAE por comuna — horizonte: {label_h} — {split}\nSprint 6 / HU7",
        fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"23_mae_comunas_h{horizonte}_{split.lower()}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()


def grafico_victorias(conteo_ganadores, split="Validation"):
    """
    Gráfico de barras horizontales mostrando cuántas combinaciones
    (comuna × horizonte) gana cada modelo.

    Responde: ¿qué modelo es más frecuentemente el mejor?
    """
    modelos = [m for m, v in conteo_ganadores.items() if v > 0]
    valores = [conteo_ganadores[m] for m in modelos]
    colores = [COLORES_MODELO.get(m, "#95A5A6") for m in modelos]

    # Ordenar de mayor a menor
    orden   = np.argsort(valores)[::-1]
    modelos = [modelos[i] for i in orden]
    valores = [valores[i] for i in orden]
    colores = [colores[i] for i in orden]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(modelos, valores, color=colores, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, valores):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val} combinaciones",
                va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Número de combinaciones ganadas (menor MAE)")
    ax.set_title(
        f"Victorias por modelo — {split}\n"
        "Una 'victoria' = menor MAE en una (comuna × horizonte)",
        fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"24_victorias_modelos_{split.lower()}.png",
                dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# PASO 6: REPORTE FINAL
# =============================================================================

def reporte_hallazgos(df_metricas, split="Validation"):
    """
    Imprime los hallazgos más relevantes del análisis por comuna:
    - Comunas más y menos difíciles de predecir
    - Horizontes donde LSTM supera a la persistencia
    - Modelo ganador global por horizonte
    """
    print("\n" + "=" * 65)
    print(f"  HALLAZGOS PRINCIPALES — HU7 / {split}")
    print("=" * 65)

    df_s = df_metricas[df_metricas["Split"] == split]

    # Comunas más difíciles (mayor MAE promedio en LSTM+aug)
    lstm_s = df_s[df_s["Modelo"] == "LSTM+aug"]
    if not lstm_s.empty:
        por_comuna = lstm_s.groupby("Nombre")["MAE"].mean().sort_values(ascending=False)
        print("\n  Comunas más difíciles de predecir (LSTM+aug, MAE promedio):")
        for nombre, mae in por_comuna.head(3).items():
            print(f"    {nombre}: MAE={mae:.2f} casos/semana")

        print("\n  Comunas mejor predichas (LSTM+aug, MAE promedio):")
        for nombre, mae in por_comuna.tail(3).items():
            print(f"    {nombre}: MAE={mae:.2f} casos/semana")

    # Horizontes donde LSTM supera a la persistencia
    print("\n  ¿En qué horizontes LSTM+aug supera a la persistencia?")
    print("  (promedio sobre las 15 comunas)")
    for h in HORIZONTES:
        lstm_h = df_s[(df_s["Modelo"] == "LSTM+aug") & (df_s["Horizonte"] == h)]["MAE"].mean()
        pers_h = df_s[(df_s["Modelo"] == "Persistencia") & (df_s["Horizonte"] == h)]["MAE"].mean()
        if not np.isnan(lstm_h) and not np.isnan(pers_h):
            label  = "actual" if h == 0 else f"h={h}"
            supera = "✓ LSTM gana" if lstm_h < pers_h else "✗ Persistencia gana"
            print(f"    {label}: LSTM={lstm_h:.2f} vs Pers={pers_h:.2f} → {supera}")

    # Modelo ganador por horizonte
    print("\n  Modelo ganador por horizonte (menor MAE promedio):")
    for h in HORIZONTES:
        sub_h   = df_s[df_s["Horizonte"] == h]
        if sub_h.empty:
            continue
        avg_mae = sub_h.groupby("Modelo")["MAE"].mean()
        ganador = avg_mae.idxmin()
        mae_gan = avg_mae.min()
        label   = "actual" if h == 0 else f"h={h}"
        print(f"    {label}: {ganador} (MAE={mae_gan:.2f})")

    print("=" * 65 + "\n")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_hu7():
    """
    Ejecuta el pipeline completo de evaluación exhaustiva por comuna.
    HU7 — Sprint 6.
    """
    print("\n" + "=" * 65)
    print("  SPRINT 6 / HU7 — Evaluación por comuna y horizonte")
    print("  Modelos: Persistencia | RF | XGBoost | LSTM+aug")
    print("  Comunas: 15 | Horizontes: actual, h=1, h=2, h=3, h=4")
    print("  Conjuntos: Train | Validation | Test")
    print("=" * 65 + "\n")

    # Paso 1: cargar
    df_train, df_val, df_test = cargar_datos()
    modelos = cargar_modelos()

    # Pasos 2-3: evaluar por conjunto
    todos = []
    for df, nombre in [(df_train, "Train"), (df_val, "Validation"), (df_test, "Test")]:
        todos.extend(evaluar_por_comuna(df, modelos, nombre))

    df_metricas = pd.DataFrame(todos)

    # Paso 4: tablas
    for split in ["Train", "Validation", "Test"]:
        tabla_mae_por_horizonte(df_metricas, split)

    conteo = tabla_ganadores(df_metricas, "Validation")

    # Paso 5: gráficos
    heatmap_mae_por_comuna(df_metricas, "Validation")
    grafico_mae_por_comuna(df_metricas, "Validation", horizonte=4)
    grafico_mae_por_comuna(df_metricas, "Validation", horizonte=1)
    grafico_victorias(conteo, "Validation")

    # Paso 6: reporte y guardado
    reporte_hallazgos(df_metricas, "Validation")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = REPORTS_DIR / "hu7_metricas_por_comuna.csv"
    df_metricas.to_csv(output_csv, index=False)
    logger.info("Métricas guardadas: %s", output_csv)

    print("✓ HU7 completado.")
    print("  Archivos generados:")
    print("  - reports/hu7_metricas_por_comuna.csv")
    print("  - reports/figures/22_heatmap_mae_comunas_validation.png")
    print("  - reports/figures/23_mae_comunas_h4_validation.png")
    print("  - reports/figures/23_mae_comunas_h1_validation.png")
    print("  - reports/figures/24_victorias_modelos_validation.png")

    return df_metricas


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    df_metricas = run_hu7()
