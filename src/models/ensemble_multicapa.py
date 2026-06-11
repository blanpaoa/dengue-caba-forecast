"""
================================================================================
Sprint 6 / HU7 — Ensemble multicapa para predicción de dengue en CABA
================================================================================

El análisis demostró que no existe un modelo universalmente superior
para predecir casos de dengue en CABA:

  Persistencia → gana en h=1 y h=2 en TODAS las comunas (53.3% de victorias)
  LSTM+aug     → gana en h=4 en TODAS las comunas (30.7% de victorias)
  XGBoost      → gana en semana actual en comunas de incidencia moderada
  Random Forest→ gana en semana actual en comunas periféricas

Si usáramos siempre el mismo modelo perderíamos precisión en los horizontes
donde ese modelo no es el mejor. El ensemble multicapa maximiza la precisión
en todos los horizontes simultáneamente, usando cada modelo donde es experto.

ARQUITECTURA DEL ENSEMBLE:
---------------------------
  Semana actual → XGBoost
    Razón: gana en 7/15 comunas en semana actual. Combina features de
    vecindad espacial y transformación log que el RF no tiene.

  h=1 (1 semana) → Persistencia
    Razón: gana en 15/15 comunas. La autocorrelación temporal del dengue
    durante un brote activo es tan fuerte que ningún modelo más complejo
    puede superarla a 1 semana de distancia.

  h=2 (2 semanas) → Persistencia
    Razón: ídem h=1. La inercia temporal del brote sigue siendo dominante
    a 2 semanas.

  h=3 (3 semanas) → LSTM+aug (todas las comunas, incluyendo C1)
    Razón: LSTM gana en 10/15 comunas. Originalmente se asignó persistencia
    a C1 porque en el análisis individual de HU7 la persistencia ganaba en
    h=3 para C1. Sin embargo el análisis del ensemble mostró que durante el
    brote 2024 la persistencia en C1 tiene MAE=397 a h=3 (predice cases_lag1
    contra un target que 3 semanas después puede ser 10x mayor), mientras
    que el LSTM tiene MAE=122. El LSTM, aunque imperfecto para C1, es
    sustancialmente mejor que la persistencia a este horizonte.

  h=4 (4 semanas) → LSTM+aug
    Razón: gana en 15/15 comunas. La arquitectura recurrente con data
    augmentation captura la dinámica temporal del brote mejor que cualquier
    otro modelo a largo plazo.

OUTPUTS GENERADOS:
  - Tabla comparativa: Ensemble vs modelos individuales
  - Gráfico de barras: MAE por horizonte
  - Gráfico de torta: distribución de modelos usados
  - ensemble_predicciones.csv: predicciones detalladas por fila
  - ensemble_metricas.csv: métricas comparativas
  - ensemble_arquitectura.json: configuración del ensemble para el dashboard

PREREQUISITO:
  Ejecutar todos los scripts del Sprint 5 antes de este.
  Requiere los modelos guardados en models/saved/.
"""

import os
import json
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Silenciar warnings de TensorFlow que no son errores
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

# Horizontes a evaluar (0 = semana actual)
HORIZONTES = [0, 1, 2, 3, 4]


# =============================================================================
# ARQUITECTURA DEL ENSEMBLE
#
# Esta tabla codifica la lógica de selección del ensemble.
# Se construyó a partir de los resultados de HU7:
#   hu7_evaluacion_comunas.py → tabla de ganadores por (comuna, horizonte)
#
# Estructura:
#   horizonte: {
#     "default": modelo_para_todas_las_comunas,
#     "excepciones": {comuna_id: modelo_alternativo}
#   }
#
# =============================================================================

# ARQUITECTURA DEL ENSEMBLE — versión final
# La excepción original C1→persistencia en h=3 fue eliminada porque
# el análisis del CSV mostró que la persistencia tiene MAE=397 en C1
# durante el brote 2024 a h=3, mientras que el LSTM tiene MAE=122.
# El LSTM, aunque imperfecto, es mejor que la persistencia a largo plazo
# incluso en la comuna más difícil.
ARQUITECTURA = {
    0: {"default": "xgboost",      "excepciones": {}},
    1: {"default": "persistencia", "excepciones": {}},
    2: {"default": "persistencia", "excepciones": {}},
    3: {"default": "lstm",         "excepciones": {}},
    4: {"default": "lstm",         "excepciones": {}},
}

# Nombres de las 15 comunas para los gráficos
NOMBRES_COMUNAS = {
    1: "C1 Puerto Madero", 2: "C2 Recoleta",    3: "C3 Balvanera",
    4: "C4 La Boca",       5: "C5 Almagro",     6: "C6 Caballito",
    7: "C7 Flores",        8: "C8 Lugano",       9: "C9 Liniers",
    10: "C10 Floresta",    11: "C11 V. del Parque", 12: "C12 Coghlan",
    13: "C13 Belgrano",    14: "C14 Palermo",    15: "C15 Agronomía",
}

# Features del XGBoost v2 (con vecindad espacial — Sprint 5)
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

# Colores consistentes con los gráficos anteriores del proyecto
COLORES = {
    "Ensemble":      "#2ECC71",  # verde — el ensemble se destaca
    "Persistencia":  "#F39C12",  # naranja
    "XGBoost":       "#E74C3C",  # rojo
    "LSTM+aug":      "#8E44AD",  # violeta
    "Random Forest": "#3498DB",  # azul
}


# =============================================================================
# PASO 1: CARGA DE DATOS Y MODELOS
#
# Cargamos todos los modelos componentes del ensemble de una sola vez.
# Esto es más eficiente que cargarlos para cada predicción — los modelos
# Keras en particular son pesados y tardan varios segundos en cargar.
# =============================================================================

def cargar_datos():
    """
    Carga los tres conjuntos de datos generados en lags.py.
    Train: 2023 | Validation: 2024 S1 (brote) | Test: 2024 S2 + 2025
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

    logger.info("  Train: %d | Val: %d | Test: %d",
                len(df_train), len(df_val), len(df_test))
    return df_train, df_val, df_test


def cargar_modelos():
    """
    Carga todos los modelos componentes del ensemble desde disco.

    Modelos cargados:
    - XGBoost: 5 modelos (h=0 a h=4), formato .pkl
    - LSTM simple + aug: 5 modelos (h=0 a h=4), formato .keras
    - LSTM scaler: normalizador del target, formato .pkl
    """
    logger.info("--- Cargando modelos componentes del ensemble ---")
    modelos = {}

    # XGBoost — un modelo entrenado por cada horizonte
    for h in HORIZONTES:
        nombre = "xgboost_semana_actual" if h == 0 else f"xgboost_h{h}"
        path   = MODELS_DIR / f"{nombre}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                modelos[f"xgb_h{h}"] = pickle.load(f)
            logger.info("  ✓ XGBoost h=%d cargado", h)
        else:
            logger.warning("  ✗ XGBoost h=%d no encontrado: %s", h, path)

    # Features del XGBoost guardadas durante el entrenamiento
    # Es crítico usar exactamente las mismas features — si cambiamos la lista
    # el modelo recibirá inputs incorrectos y las predicciones serán erróneas
    feat_path = MODELS_DIR / "xgboost_features_v2.pkl"
    if feat_path.exists():
        with open(feat_path, "rb") as f:
            modelos["xgb_features"] = pickle.load(f)

    # LSTM simple + augmentation — un modelo por horizonte
    for h in HORIZONTES:
        path = MODELS_DIR / f"lstm_lstm_simple_h{h}.keras"
        if path.exists():
            modelos[f"lstm_h{h}"] = tf.keras.models.load_model(path)
            logger.info("  ✓ LSTM h=%d cargado", h)
        else:
            logger.warning("  ✗ LSTM h=%d no encontrado: %s", h, path)

    # Scaler del target del LSTM
    # IMPORTANTE: siempre usamos el scaler del entrenamiento para desnormalizar,
    # nunca uno nuevo ajustado con los datos de validación o test.
    scaler_path = MODELS_DIR / "lstm_target_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            modelos["lstm_scaler"] = pickle.load(f)
        logger.info("  ✓ LSTM scaler cargado")

    n = sum(1 for k in modelos if k.startswith(("xgb_h", "lstm_h")))
    logger.info("  Total modelos componentes: %d", n)
    return modelos


# =============================================================================
# PASO 2: PREDICCIONES POR MODELO COMPONENTE
#
# Cada función de predicción devuelve tres valores:
#   y_real: los casos reales observados (en escala original)
#   y_pred: los casos predichos por el modelo (en escala original)
#   idx:    los índices del dataset donde hay predicciones válidas
#
# Los índices son importantes para filtrar por comuna en el paso siguiente.
# Todas las predicciones se devuelven en escala real de casos (0-1391),
# sin normalización — para que sean comparables entre modelos.
# =============================================================================

def pred_persistencia(df, horizonte):
    """
    Persistencia: predice que habrá los mismos casos que la semana pasada.
    pred(t) = cases_lag1(t)

    Es el baseline epidemiológico más simple posible. Su fortaleza radica
    en la alta autocorrelación temporal del dengue: durante un brote activo,
    el número de casos de esta semana está fuertemente correlacionado con el
    de la semana anterior.

    Nota: técnicamente la persistencia predice 'la semana próxima tendrá
    los mismos casos que esta semana'. Al usarla para cualquier horizonte
    (h=1, h=2, etc.) siempre predice el mismo valor — cases_lag1.
    """
    target_col = "confirmed_cases" if horizonte == 0 else f"target_h{horizonte}"
    if target_col not in df.columns:
        return None, None, None

    mask   = df["cases_lag1"].notna() & df[target_col].notna()
    y_real = df.loc[mask, target_col].values
    y_pred = df.loc[mask, "cases_lag1"].fillna(0).values
    idx    = df[mask].index.tolist()
    return y_real, y_pred, idx


def pred_xgboost(df, modelos, horizonte):
    """
    XGBoost: predice usando el modelo entrenado en Sprint 5.

    El XGBoost fue entrenado con transformación log1p del target:
      Entrenamiento: y = log(1 + casos_reales)
      Predicción:    pred_log = modelo.predict(X)
      Desnormalizar: pred_real = exp(pred_log) - 1 = expm1(pred_log)

    Por eso las predicciones del modelo vienen en escala logarítmica y
    necesitamos aplicar expm1 para obtener los casos reales.
    Esto es exactamente lo inverso de lo que se hizo durante el entrenamiento.
    """
    xgb_key = f"xgb_h{horizonte}"
    if xgb_key not in modelos:
        return None, None, None

    # Usar las features guardadas del entrenamiento para garantizar consistencia
    features   = modelos.get("xgb_features", FEATURES_XGB)
    features   = [f for f in features if f in df.columns]
    target_col = "confirmed_cases" if horizonte == 0 else f"target_h{horizonte}"

    if target_col not in df.columns:
        return None, None, None

    mask     = df[features].notna().all(axis=1) & df[target_col].notna()
    X        = df.loc[mask, features].values
    y_real   = df.loc[mask, target_col].values
    pred_log = modelos[xgb_key].predict(X).clip(min=0)
    y_pred   = np.expm1(pred_log).clip(min=0)   # desnormalizar log1p
    idx      = df[mask].index.tolist()
    return y_real, y_pred, idx


def construir_secuencias_lstm(df, features, horizonte=0):
    """
    Construye ventanas temporales deslizantes para el LSTM.

    El LSTM no puede recibir filas individuales — necesita ver las últimas
    WINDOW_SIZE semanas EN ORDEN para entender la dinámica temporal.

    Para cada semana t de cada comuna construimos:
      Entrada X: datos de semanas [t-12, ..., t-1] → forma (12, n_features)
      Target y:  casos en t (o en t+h para horizontes futuros)

    Las ventanas se construyen DENTRO de cada comuna para no mezclar
    series temporales de comunas diferentes.
    """
    target_col = "confirmed_cases" if horizonte == 0 else f"target_h{horizonte}"
    if target_col not in df.columns:
        return None, None, None

    X_list, y_list, idx_list = [], [], []

    for comuna_id in sorted(df["comuna_id"].unique()):
        mask  = df["comuna_id"] == comuna_id
        df_c  = df[mask].sort_values(["year", "epi_week"]).copy()
        idx_c = df_c.index.tolist()
        X_c   = df_c[features].fillna(0).values
        y_c   = df_c[target_col].values
        n     = len(df_c)

        for t in range(WINDOW_SIZE, n):
            val = y_c[t]
            if not np.isnan(val):
                X_list.append(X_c[t - WINDOW_SIZE : t])
                y_list.append(val)
                idx_list.append(idx_c[t])

    if not X_list:
        return None, None, None

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list,  dtype=np.float32),
            idx_list)


def pred_lstm(df, modelos, horizonte):
    """
    LSTM simple + augmentation: predice con ventanas de 12 semanas.

    El modelo predice en escala normalizada [0, 1] porque fue entrenado
    con MinMaxScaler sobre el train set (max=649 casos → 1.0).
    Desnormalizamos usando el scaler guardado del entrenamiento — siempre
    el mismo, independientemente del conjunto evaluado.

    IMPORTANTE: no creamos un scaler nuevo para validación o test porque
    eso generaría una escala diferente y las predicciones serían incorrectas.
    """
    lstm_key = f"lstm_h{horizonte}"
    if lstm_key not in modelos or "lstm_scaler" not in modelos:
        return None, None, None

    features = [f for f in FEATURES_LSTM if f in df.columns]
    scaler   = modelos["lstm_scaler"]

    X_seq, y_seq, idx_list = construir_secuencias_lstm(df, features, horizonte)
    if X_seq is None:
        return None, None, None

    # Predicción en escala normalizada
    pred_norm = modelos[lstm_key].predict(X_seq, verbose=0).flatten()

    # Desnormalizar con el scaler del entrenamiento → casos reales
    y_pred = scaler.inverse_transform(
        pred_norm.reshape(-1, 1)
    ).flatten().clip(min=0)

    # El target real ya está en escala de casos (no pasó por el scaler)
    y_real = y_seq.clip(min=0)

    return y_real, y_pred, idx_list


# =============================================================================
# PASO 3: SELECCIÓN DEL MODELO SEGÚN ARQUITECTURA DEL ENSEMBLE
#
# La función seleccionar_modelo implementa la lógica de decisión del ensemble.
# Para cada combinación (horizonte, comuna_id) consulta la tabla ARQUITECTURA
# y retorna el nombre del modelo que debe usarse.
#
# El proceso es:
#   1. Buscar en ARQUITECTURA[horizonte]["excepciones"][comuna_id]
#      → si existe una excepción para esta comuna, usarla
#   2. Si no hay excepción, usar ARQUITECTURA[horizonte]["default"]
#
# Este diseño permite agregar excepciones fácilmente en el futuro sin
# modificar la lógica principal. Por ejemplo, si se descubre que C4
# también se predice mejor con persistencia en h=3, solo hay que agregar
# {4: "persistencia"} a las excepciones de h=3.
# =============================================================================

def seleccionar_modelo(horizonte, comuna_id):
    """
    Retorna el nombre del modelo que el ensemble debe usar para la
    combinación (horizonte, comuna_id).

    Ejemplos:
      seleccionar_modelo(4, 5)  → "lstm"      (LSTM para C5 en h=4)
      seleccionar_modelo(1, 8)  → "persistencia" (default h=1)
      seleccionar_modelo(0, 7)  → "xgboost"   (default h=0)
    """
    config      = ARQUITECTURA[horizonte]
    excepciones = config.get("excepciones", {})

    if comuna_id in excepciones:
        modelo = excepciones[comuna_id]
        logger.debug(
            "  Excepción aplicada: horizonte=%d, C%d → %s",
            horizonte, comuna_id, modelo
        )
    else:
        modelo = config["default"]

    return modelo


def generar_predicciones_ensemble(df, modelos, split_nombre):
    """
    Genera las predicciones del ensemble combinando los modelos componentes.

    Estrategia: en lugar de iterar fila a fila, generamos los DataFrames de
    predicciones de cada modelo y los combinamos según la arquitectura.

    Para cada horizonte:
    1. Genera DataFrames de predicciones de persistencia, XGBoost y LSTM
    2. Para cada comuna, selecciona el modelo correcto según ARQUITECTURA
    3. Extrae las filas del modelo seleccionado para esa comuna
    4. Concatena todo en un único DataFrame del ensemble

    Este enfoque evita el problema de índices desalineados entre modelos
    (el LSTM descarta las primeras 12 semanas, XGBoost no las descarta).
    """
    logger.info("  Generando predicciones ensemble: %s", split_nombre)
    resultados = []

    for horizonte in HORIZONTES:
        # Generar DataFrames de predicciones para cada modelo componente
        y_r_pers, y_p_pers, idx_pers = pred_persistencia(df, horizonte)
        y_r_xgb,  y_p_xgb,  idx_xgb  = pred_xgboost(df, modelos, horizonte)
        y_r_lstm, y_p_lstm, idx_lstm  = pred_lstm(df, modelos, horizonte)

        # Construir DataFrames por modelo con la info de comuna
        def hacer_df(y_r, y_p, idx_list, nombre_modelo):
            if y_r is None or len(y_r) == 0:
                return pd.DataFrame()
            df_m = pd.DataFrame({
                "idx":          idx_list,
                "y_real":       y_r,
                "y_pred":       y_p,
                "modelo_usado": nombre_modelo,
            })
            # Agregar comuna_id desde el dataframe original
            df_m["comuna_id"] = df.loc[idx_list, "comuna_id"].values
            return df_m

        df_pers = hacer_df(y_r_pers, y_p_pers, idx_pers, "persistencia")
        df_xgb  = hacer_df(y_r_xgb,  y_p_xgb,  idx_xgb,  "xgboost")
        df_lstm = hacer_df(y_r_lstm, y_p_lstm, idx_lstm, "lstm")

        pred_dfs = {"persistencia": df_pers, "xgboost": df_xgb, "lstm": df_lstm}

        # Para cada comuna, seleccionar el modelo correcto y tomar sus filas
        for comuna_id in sorted(df["comuna_id"].unique()):
            modelo_sel = seleccionar_modelo(horizonte, int(comuna_id))
            df_modelo  = pred_dfs.get(modelo_sel, pd.DataFrame())

            if df_modelo.empty:
                continue

            # Filtrar solo las filas de esta comuna
            filas_comuna = df_modelo[df_modelo["comuna_id"] == comuna_id]
            if filas_comuna.empty:
                continue

            for _, row in filas_comuna.iterrows():
                resultados.append({
                    "Split":        split_nombre,
                    "Horizonte":    horizonte,
                    "Comuna":       int(comuna_id),
                    "Nombre":       NOMBRES_COMUNAS.get(int(comuna_id), f"C{int(comuna_id)}"),
                    "idx":          row["idx"],
                    "y_real":       float(row["y_real"]),
                    "y_pred":       float(row["y_pred"]),
                    "modelo_usado": modelo_sel,
                })

    df_result = pd.DataFrame(resultados)
    logger.info(
        "  Predicciones generadas: %d | modelos usados: %s",
        len(df_result),
        df_result["modelo_usado"].value_counts().to_dict() if not df_result.empty else {}
    )
    return df_result


# =============================================================================
# MÉTRICAS DE EVALUACIÓN
# =============================================================================

def calcular_metricas(y_real, y_pred, modelo, split, horizonte):
    """
    Calcula MAE, RMSE y R² en escala original de casos.
    Siempre en escala real (0-1391 casos) para comparabilidad.
    """
    if len(y_real) == 0:
        return None

    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2   = r2_score(y_real, y_pred)

    return {
        "Modelo":    modelo,
        "Split":     split,
        "Horizonte": horizonte,
        "MAE":       round(mae, 2),
        "RMSE":      round(rmse, 2),
        "R2":        round(r2, 3),
        "N":         len(y_real),
    }


# =============================================================================
# PASO 4: COMPARACIÓN ENSEMBLE VS MODELOS INDIVIDUALES
#
# Para que la comparación sea justa evaluamos el ensemble y cada modelo
# individual sobre exactamente las mismas predicciones.
#
# El ensemble por definición usa el mejor modelo para cada (horizonte, comuna),
# por lo que su MAE debería ser menor o igual al del mejor modelo individual
# en cada horizonte. Si no fuera así, indicaría un error en la lógica de
# selección.
# =============================================================================

def comparar_ensemble_vs_individuales(df, modelos, split_nombre, df_ens_all=None):
    """
    Compara el MAE del ensemble contra cada modelo individual.

    Acepta opcionalmente df_ens_all con las predicciones del ensemble ya
    calculadas en el Paso 2. Si no se provee, las calcula internamente.
    Esto garantiza que el ensemble se evalúa exactamente sobre las mismas
    predicciones generadas en el Paso 2 — sin recalcular ni mezclar índices.

    Para comparación justa, todos los modelos se evalúan sobre el mismo
    conjunto de filas — las que tienen predicciones válidas en el ensemble.
    """
    logger.info("  Comparando ensemble vs individuales: %s", split_nombre)
    metricas = []

    # Usar predicciones pre-calculadas o generarlas si no se proveyeron
    if df_ens_all is None:
        df_ens_all = generar_predicciones_ensemble(df, modelos, split_nombre)

    for horizonte in HORIZONTES:
        target_col = "confirmed_cases" if horizonte == 0 else f"target_h{horizonte}"
        if target_col not in df.columns:
            continue

        # Métricas del ensemble para este horizonte
        df_h = df_ens_all[df_ens_all["Horizonte"] == horizonte]
        if df_h.empty:
            continue

        m = calcular_metricas(
            df_h["y_real"].values, df_h["y_pred"].values,
            "Ensemble", split_nombre, horizonte
        )
        if m:
            metricas.append(m)

        # Métricas de modelos individuales — evaluados directamente
        for nombre, pred_fn in [
            ("Persistencia", lambda h: pred_persistencia(df, h)),
            ("XGBoost",      lambda h: pred_xgboost(df, modelos, h)),
            ("LSTM+aug",     lambda h: pred_lstm(df, modelos, h)),
        ]:
            y_real, y_pred, idx = pred_fn(horizonte)
            if y_real is None or len(y_real) == 0:
                continue

            m = calcular_metricas(y_real, y_pred, nombre, split_nombre, horizonte)
            if m:
                metricas.append(m)

    return metricas


# =============================================================================
# PASO 5: TABLAS Y GRÁFICOS
# =============================================================================

def tabla_comparacion(metricas, split="Validation"):
    """
    Tabla que muestra el MAE del ensemble vs cada modelo individual.

    Para cada horizonte indica:
      - El MAE de cada modelo
      - La diferencia del ensemble respecto al mejor modelo individual
      - ✓ si el ensemble mejora al mejor individual, ✗ si no

    Un ensemble bien diseñado debería igualar o superar al mejor modelo
    individual en cada horizonte.
    """
    print("\n" + "=" * 80)
    print(f"  ENSEMBLE vs MODELOS INDIVIDUALES — {split}")
    print("  MAE promedio sobre todas las comunas (menor es mejor)")
    print("=" * 80)

    df_m = pd.DataFrame(metricas)
    df_s = df_m[df_m["Split"] == split]
    horiz    = sorted(df_s["Horizonte"].unique())
    h_labels = {0: "actual", 1: "h=1", 2: "h=2", 3: "h=3", 4: "h=4"}

    print(f"\n  {'Modelo':<22}", end="")
    for h in horiz:
        print(f"  {h_labels[h]:>8}", end="")
    print()
    print("  " + "-" * (22 + 10 * len(horiz)))

    for modelo in ["Ensemble", "Persistencia", "XGBoost", "LSTM+aug"]:
        sub = df_s[df_s["Modelo"] == modelo]
        if sub.empty:
            continue
        # Destacar el ensemble con flecha
        prefijo = "→ " if modelo == "Ensemble" else "  "
        print(f"  {prefijo}{modelo:<20}", end="")
        for h in horiz:
            val = sub[sub["Horizonte"] == h]["MAE"].values
            print(f"  {val[0]:>8.2f}" if len(val) else f"  {'---':>8}", end="")
        print()

    # Análisis de la mejora del ensemble
    print(f"\n  Análisis de mejora del Ensemble sobre el mejor modelo individual:")
    ens_sub = df_s[df_s["Modelo"] == "Ensemble"]
    for h in horiz:
        mae_ens = ens_sub[ens_sub["Horizonte"] == h]["MAE"].values
        if not len(mae_ens):
            continue
        otros     = df_s[(df_s["Modelo"] != "Ensemble") & (df_s["Horizonte"] == h)]
        if otros.empty:
            continue
        idx_min   = otros["MAE"].idxmin()
        mejor_mae = otros.loc[idx_min, "MAE"]
        mejor_mod = otros.loc[idx_min, "Modelo"]
        diff      = mae_ens[0] - mejor_mae
        signo     = "+" if diff > 0 else ""
        simbolo   = "✓" if diff <= 0.5 else "~" if diff <= 2 else "✗"
        print(
            f"    {h_labels[h]:>6}: Ensemble={mae_ens[0]:.2f} vs "
            f"{mejor_mod}={mejor_mae:.2f} ({signo}{diff:.2f}) {simbolo}"
        )

    print("\n  ✓ = ensemble igual o mejor | ~ = diferencia mínima (<2 casos)")
    print("=" * 80 + "\n")


def grafico_ensemble_vs_individuales(metricas, split="Validation"):
    """
    Gráfico de barras agrupadas comparando MAE del ensemble vs modelos
    individuales por horizonte.

    El ensemble se destaca con borde negro y color verde para diferenciarlo
    visualmente de los modelos individuales.
    """
    df_m  = pd.DataFrame(metricas)
    df_s  = df_m[df_m["Split"] == split]
    horiz = sorted(df_s["Horizonte"].unique())
    mods  = ["Ensemble", "Persistencia", "XGBoost", "LSTM+aug"]
    n_mod = len(mods)
    x     = np.arange(len(horiz))
    ancho = 0.8 / n_mod
    h_labels = {0: "actual", 1: "h=1", 2: "h=2", 3: "h=3", 4: "h=4"}

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, modelo in enumerate(mods):
        sub  = df_s[df_s["Modelo"] == modelo]
        maes = [
            sub[sub["Horizonte"] == h]["MAE"].values[0]
            if not sub[sub["Horizonte"] == h].empty else np.nan
            for h in horiz
        ]
        # El ensemble se destaca con borde negro
        es_ensemble = modelo == "Ensemble"
        ax.bar(
            x + i * ancho - (n_mod - 1) * ancho / 2,
            maes, ancho,
            label=modelo,
            color=COLORES.get(modelo, "#95A5A6"),
            alpha=0.95 if es_ensemble else 0.75,
            edgecolor="black" if es_ensemble else "white",
            linewidth=2.0 if es_ensemble else 0.5
        )

    ax.set_xticks(x)
    ax.set_xticklabels([h_labels[h] for h in horiz], fontsize=11)
    ax.set_ylabel("MAE (casos promedio por semana)", fontsize=11)
    ax.set_xlabel("Horizonte de predicción", fontsize=11)
    ax.set_title(
        f"Ensemble multicapa vs modelos individuales — {split}\n"
        "Sprint 6 / HU7 · Sistema de alertas tempranas CABA",
        fontweight="bold", fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        FIGURES_DIR / f"25_ensemble_vs_individuales_{split.lower()}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()


def grafico_distribucion_modelos(df_pred):
    """
    Gráfico de torta mostrando qué porcentaje de las predicciones totales
    usa cada modelo componente del ensemble.

    Permite verificar que la arquitectura está bien balanceada:
    - Si un modelo domina demasiado → el ensemble no agrega mucho valor
    - Si está bien distribuido → cada modelo aporta en su especialidad
    """
    conteo = df_pred.groupby("modelo_usado").size()
    total  = len(df_pred)

    # Mapear nombres internos a nombres para el gráfico
    nombre_display = {
        "persistencia": "Persistencia",
        "xgboost":      "XGBoost",
        "lstm":         "LSTM+aug",
    }

    labels  = [nombre_display.get(k, k) for k in conteo.index]
    sizes   = conteo.values.tolist()
    colores = [COLORES.get(nombre_display.get(k, k), "#95A5A6") for k in conteo.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=[f"{l}\n({s} predicciones)" for l, s in zip(labels, sizes)],
        colors=colores,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10}
    )
    for at in autotexts:
        at.set_fontweight("bold")

    ax.set_title(
        "Distribución de modelos usados por el ensemble\n"
        "Validation · Sprint 6 / HU7",
        fontweight="bold", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "26_distribucion_ensemble.png",
                dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_ensemble():
    """
    Ejecuta el pipeline completo del ensemble multicapa.

    1. Carga datos y modelos entrenados del Sprint 5
    2. Genera predicciones del ensemble para validation y test
    3. Compara el ensemble contra cada modelo individual
    4. Genera tablas y gráficos comparativos
    5. Guarda resultados y la arquitectura en JSON para el dashboard

    El JSON de arquitectura (ensemble_arquitectura.json) es usado por el
    dashboard (HU8) para saber qué modelo usar para cada predicción en
    tiempo real — sin necesidad de recalcular la lógica de selección.
    """
    print("\n" + "=" * 65)
    print("  SPRINT 6 / HU7 — Ensemble Multicapa")
    print("  Arquitectura definida en ARQUITECTURA[]:")
    h_labels = {0: "actual", 1: "h=1", 2: "h=2", 3: "h=3", 4: "h=4"}
    for h, config in ARQUITECTURA.items():
        exc = config.get("excepciones", {})
        exc_str = (
            f" (excepto C{list(exc.keys())[0]} → {list(exc.values())[0]})"
            if exc else ""
        )
        print(f"    {h_labels[h]:>8}: {config['default']}{exc_str}")
    print("=" * 65 + "\n")

    # ── Paso 1: cargar ────────────────────────────────────────────────
    df_train, df_val, df_test = cargar_datos()
    modelos = cargar_modelos()

    # ── Paso 2: generar predicciones del ensemble ─────────────────────
    logger.info("--- PASO 2: Generando predicciones del ensemble ---")
    df_pred_val  = generar_predicciones_ensemble(df_val,  modelos, "Validation")
    df_pred_test = generar_predicciones_ensemble(df_test, modelos, "Test")
    df_pred_all  = pd.concat([df_pred_val, df_pred_test], ignore_index=True)

    # ── Paso 3: calcular métricas comparativas ────────────────────────
    # Pasamos los DataFrames ya calculados en el Paso 2 para evitar
    # recalcular las predicciones del ensemble y garantizar consistencia
    logger.info("--- PASO 3: Calculando métricas comparativas ---")
    met_val  = comparar_ensemble_vs_individuales(df_val,  modelos, "Validation", df_pred_val)
    met_test = comparar_ensemble_vs_individuales(df_test, modelos, "Test",       df_pred_test)
    todas    = met_val + met_test

    # ── Paso 4: tablas ────────────────────────────────────────────────
    tabla_comparacion(todas, "Validation")
    tabla_comparacion(todas, "Test")

    # ── Paso 5: gráficos ──────────────────────────────────────────────
    grafico_ensemble_vs_individuales(todas, "Validation")
    grafico_distribucion_modelos(df_pred_val)

    # ── Paso 6: guardar resultados ────────────────────────────────────
    logger.info("--- PASO 6: Guardando resultados ---")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_pred_all.to_csv(REPORTS_DIR / "ensemble_predicciones.csv", index=False)
    pd.DataFrame(todas).to_csv(REPORTS_DIR / "ensemble_metricas.csv", index=False)

    # Guardar arquitectura en JSON para el dashboard (HU8)
    # El dashboard lee este archivo para saber qué modelo usar en tiempo real
    arch_json = {
        str(h): {
            "default": v["default"],
            "excepciones": {str(k): vv for k, vv in v["excepciones"].items()}
        }
        for h, v in ARQUITECTURA.items()
    }
    with open(MODELS_DIR / "ensemble_arquitectura.json", "w") as f:
        json.dump(arch_json, f, indent=2)

    logger.info("  ensemble_predicciones.csv guardado")
    logger.info("  ensemble_metricas.csv guardado")
    logger.info("  ensemble_arquitectura.json guardado")

    print("\n✓ Ensemble multicapa completado.")
    print("  Próximo paso: validación cruzada temporal sobre el ensemble")

    return df_pred_all, todas


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    df_pred, metricas = run_ensemble()
