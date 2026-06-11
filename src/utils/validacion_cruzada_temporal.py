"""
================================================================================
Sprint 6 / HU7 — Validación cruzada temporal
================================================================================

¿QUÉ ES LA VALIDACIÓN CRUZADA TEMPORAL Y POR QUÉ ES NECESARIA?
----------------------------------------------------------------
Hasta ahora evaluamos el ensemble sobre un único período de validación:
2024 semanas 1-26. Ese período cubre el brote más severo registrado en CABA,
lo que es ideal para el peor caso pero no responde preguntas importantes:

  ¿El sistema funciona bien en TODOS los períodos o solo en el brote 2024?
  ¿Hay semanas específicas donde el sistema falla sistemáticamente?
  ¿El rendimiento es estable a lo largo del tiempo o muy variable?
  ¿Si entrenáramos con más datos el sistema mejoraría?

La validación cruzada temporal responde estas preguntas evaluando el sistema
sobre múltiples períodos temporales distintos.

¿POR QUÉ "TEMPORAL" Y NO LA VALIDACIÓN CRUZADA CLÁSICA?
---------------------------------------------------------
La validación cruzada clásica (K-Fold) divide los datos aleatoriamente en K
grupos. En series temporales esto genera DATA LEAKAGE — el modelo vería datos
del futuro durante el entrenamiento, lo que no es posible en producción real.

La validación cruzada TEMPORAL siempre respeta el orden cronológico:
el conjunto de evaluación siempre es POSTERIOR al de entrenamiento.
Esto replica fielmente el escenario de uso real del sistema de alertas.

DOS ENFOQUES IMPLEMENTADOS:
----------------------------

1. WALK-FORWARD (ventana deslizante):
   Entrena con N semanas, evalúa la siguiente, avanza una semana y repite.
   Genera muchos folds (una evaluación por semana) lo que permite calcular
   la distribución del error a lo largo del tiempo.

   Ejemplo con step=4 semanas:
     Fold 1: train SE1-SE52  → eval SE53-SE56
     Fold 2: train SE1-SE56  → eval SE57-SE60
     Fold 3: train SE1-SE60  → eval SE61-SE64
     ...

   Responde: ¿hay períodos específicos donde el sistema falla?

2. BLOQUES ANUALES (expanding window):
   Divide en bloques epidemiológicamente significativos.
   Cada bloque incluye todos los datos anteriores más un período nuevo.

     Bloque 1: train=2023           → eval=2024 S1 (brote masivo)
     Bloque 2: train=2023+2024 S1   → eval=2024 S2 (temporada baja)
     Bloque 3: train=2023+2024      → eval=2025    (brote moderado)

   Responde: ¿el sistema generaliza entre años con diferentes intensidades?

MODELOS EVALUADOS:
------------------
  1. Ensemble multicapa  — el sistema final propuesto en la tesis
  2. LSTM+aug            — el mejor modelo individual de largo plazo
  3. Persistencia        — el baseline epidemiológico de referencia

NOTA IMPORTANTE SOBRE EL REENTRENAMIENTO:
------------------------------------------
Para el ensemble y el LSTM se usan los modelos ya entrenados en Sprint 5
SIN reentrenamiento por fold. Reentrenar el LSTM 50+ veces tomaría días
en CPU. Esta es una limitación documentada — los resultados muestran cómo
se comportan los modelos actuales en distintos períodos, no cómo se
comportarían si se reentrenaran con cada fold.

Para la persistencia no hay reentrenamiento porque es un modelo sin
parámetros — siempre predice el valor de la semana anterior.

PIPELINE:
  Paso 1 → Cargar datos completos y modelos entrenados
  Paso 2 → Walk-forward: definir folds y evaluar por semana
  Paso 3 → Bloques anuales: evaluar por período epidemiológico
  Paso 4 → Comparar estabilidad entre modelos
  Paso 5 → Gráficos de MAE a lo largo del tiempo
  Paso 6 → Guardar resultados

PREREQUISITO:
  Ejecutar ensemble_multicapa.py antes de este script.
  Requiere los modelos en models/saved/ y ensemble_arquitectura.json.
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

WINDOW_SIZE = 12   # ventana del LSTM — debe coincidir con lstm_model.py
HORIZONTES  = [1, 2, 3, 4]   # horizontes a evaluar (excluimos semana actual)

# Features del LSTM — deben coincidir exactamente con lstm_model.py
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

# Features del XGBoost — para el ensemble en semana actual
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

# Arquitectura del ensemble — igual que en ensemble_multicapa.py
ARQUITECTURA = {
    0: {"default": "xgboost",      "excepciones": {}},
    1: {"default": "persistencia", "excepciones": {}},
    2: {"default": "persistencia", "excepciones": {}},
    3: {"default": "lstm",         "excepciones": {}},
    4: {"default": "lstm",         "excepciones": {}},
}

# Colores consistentes con los gráficos anteriores
COLORES = {
    "Ensemble":     "#2ECC71",  # verde
    "LSTM+aug":     "#8E44AD",  # violeta
    "Persistencia": "#F39C12",  # naranja
}

# Nombres de las comunas
NOMBRES_COMUNAS = {
    1: "C1 Puerto Madero", 2: "C2 Recoleta",    3: "C3 Balvanera",
    4: "C4 La Boca",       5: "C5 Almagro",     6: "C6 Caballito",
    7: "C7 Flores",        8: "C8 Lugano",       9: "C9 Liniers",
    10: "C10 Floresta",    11: "C11 V. del Parque", 12: "C12 Coghlan",
    13: "C13 Belgrano",    14: "C14 Palermo",    15: "C15 Agronomía",
}


# =============================================================================
# PASO 1: CARGA DE DATOS Y MODELOS
# =============================================================================

def cargar_todo():
    """
    Carga el dataset completo (todos los años) y los modelos entrenados.

    Para la validación cruzada necesitamos el dataset completo —
    no los splits predefinidos de train/val/test — para poder
    construir nuestros propios folds temporales.

    El dataset completo se construye concatenando train, val y test
    en orden cronológico, preservando la columna year/epi_week para
    poder dividir por períodos.
    """
    logger.info("--- PASO 1: Cargando datos y modelos ---")

    # Cargar los tres conjuntos y concatenar
    df_train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    df_val   = pd.read_parquet(PROCESSED_DIR / "validation.parquet")
    df_test  = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    df_total = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df_total = df_total.sort_values(["year", "epi_week", "comuna_id"]).reset_index(drop=True)

    años = sorted(df_total["year"].unique())
    logger.info(
        "  Dataset completo: %d filas | %d comunas | años: %s",
        len(df_total), df_total["comuna_id"].nunique(), años
    )

    # Cargar modelos
    modelos = {}

    # XGBoost — semana actual
    xgb_path = MODELS_DIR / "xgboost_semana_actual.pkl"
    if xgb_path.exists():
        with open(xgb_path, "rb") as f:
            modelos["xgb_h0"] = pickle.load(f)

    # XGBoost — horizontes futuros
    for h in HORIZONTES:
        path = MODELS_DIR / f"xgboost_h{h}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                modelos[f"xgb_h{h}"] = pickle.load(f)

    # Features XGBoost
    feat_path = MODELS_DIR / "xgboost_features_v2.pkl"
    if feat_path.exists():
        with open(feat_path, "rb") as f:
            modelos["xgb_features"] = pickle.load(f)

    # LSTM
    for h in [0] + HORIZONTES:
        path = MODELS_DIR / f"lstm_lstm_simple_h{h}.keras"
        if path.exists():
            modelos[f"lstm_h{h}"] = tf.keras.models.load_model(path)

    # Scaler LSTM
    scaler_path = MODELS_DIR / "lstm_target_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            modelos["lstm_scaler"] = pickle.load(f)

    n = sum(1 for k in modelos if k.startswith(("xgb_h", "lstm_h")))
    logger.info("  Modelos cargados: %d", n)

    return df_total, modelos


# =============================================================================
# PREDICCIONES DE CADA MODELO
# Funciones auxiliares para obtener predicciones sobre un subset del dataset.
# =============================================================================

def pred_persistencia_df(df, horizonte):
    """
    Persistencia: pred(t) = cases_lag1(t).
    Devuelve arrays de y_real e y_pred filtrados por filas válidas.
    """
    target_col = "confirmed_cases" if horizonte == 0 else f"target_h{horizonte}"
    if target_col not in df.columns:
        return np.array([]), np.array([])

    mask   = df["cases_lag1"].notna() & df[target_col].notna()
    y_real = df.loc[mask, target_col].values
    y_pred = df.loc[mask, "cases_lag1"].fillna(0).values
    return y_real, y_pred


def pred_xgboost_df(df, modelos, horizonte):
    """
    XGBoost con desnormalización log1p → expm1.
    Devuelve predicciones en escala real de casos.
    """
    key = f"xgb_h{horizonte}"
    if key not in modelos:
        return np.array([]), np.array([])

    features   = modelos.get("xgb_features", FEATURES_XGB)
    features   = [f for f in features if f in df.columns]
    target_col = "confirmed_cases" if horizonte == 0 else f"target_h{horizonte}"
    if target_col not in df.columns:
        return np.array([]), np.array([])

    mask   = df[features].notna().all(axis=1) & df[target_col].notna()
    if mask.sum() == 0:
        return np.array([]), np.array([])

    X      = df.loc[mask, features].values
    y_real = df.loc[mask, target_col].values
    y_pred = np.expm1(modelos[key].predict(X).clip(min=0)).clip(min=0)
    return y_real, y_pred


def construir_secuencias(df, features, horizonte):
    """
    Ventanas deslizantes de WINDOW_SIZE semanas para el LSTM.
    Construidas por comuna para no mezclar series temporales.
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


def pred_lstm_df(df, modelos, horizonte):
    """
    LSTM con desnormalización usando el scaler del entrenamiento.
    Devuelve predicciones en escala real de casos.
    """
    key = f"lstm_h{horizonte}"
    if key not in modelos or "lstm_scaler" not in modelos:
        return np.array([]), np.array([])

    features = [f for f in FEATURES_LSTM if f in df.columns]
    scaler   = modelos["lstm_scaler"]

    X_seq, y_seq, _ = construir_secuencias(df, features, horizonte)
    if X_seq is None:
        return np.array([]), np.array([])

    pred_norm = modelos[key].predict(X_seq, verbose=0).flatten()
    y_pred    = scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten().clip(min=0)
    y_real    = y_seq.clip(min=0)

    return y_real, y_pred


def pred_ensemble_df(df, modelos, horizonte):
    """
    Ensemble multicapa: selecciona el modelo correcto según la arquitectura.

    Para cada comuna del subset, usa:
      h=0 → XGBoost
      h=1,2 → Persistencia
      h=3,4 → LSTM
    y combina las predicciones en un único array.
    """
    config    = ARQUITECTURA.get(horizonte, {"default": "persistencia", "excepciones": {}})
    y_reals   = []
    y_preds   = []

    # Predicciones de todos los modelos componentes
    yr_pers, yp_pers = pred_persistencia_df(df, horizonte)
    yr_xgb,  yp_xgb  = pred_xgboost_df(df, modelos, horizonte)
    yr_lstm, yp_lstm = pred_lstm_df(df, modelos, horizonte)

    # Construir DataFrames por modelo con la info de comuna
    target_col = "confirmed_cases" if horizonte == 0 else f"target_h{horizonte}"
    if target_col not in df.columns:
        return np.array([]), np.array([])

    def hacer_df_pred(yr, yp, df_orig, modelo_nombre):
        if len(yr) == 0:
            return pd.DataFrame()
        features_check = (
            FEATURES_LSTM if modelo_nombre == "lstm"
            else modelos.get("xgb_features", FEATURES_XGB) if modelo_nombre == "xgboost"
            else ["cases_lag1"]
        )
        # Reconstruir índices válidos
        if modelo_nombre == "persistencia":
            mask = df_orig["cases_lag1"].notna() & df_orig[target_col].notna()
            idx  = df_orig[mask].index.tolist()
        elif modelo_nombre == "xgboost":
            feats = modelos.get("xgb_features", FEATURES_XGB)
            feats = [f for f in feats if f in df_orig.columns]
            mask  = df_orig[feats].notna().all(axis=1) & df_orig[target_col].notna()
            idx   = df_orig[mask].index.tolist()
        else:  # lstm
            feats = [f for f in FEATURES_LSTM if f in df_orig.columns]
            _, _, idx = construir_secuencias(df_orig, feats, horizonte)
            if idx is None:
                return pd.DataFrame()

        n = min(len(yr), len(idx))
        return pd.DataFrame({
            "idx":       idx[:n],
            "y_real":    yr[:n],
            "y_pred":    yp[:n],
            "comuna_id": df_orig.loc[idx[:n], "comuna_id"].values,
        })

    df_pers = hacer_df_pred(yr_pers, yp_pers, df, "persistencia")
    df_xgb  = hacer_df_pred(yr_xgb,  yp_xgb,  df, "xgboost")
    df_lstm = hacer_df_pred(yr_lstm, yp_lstm, df, "lstm")

    pred_map = {"persistencia": df_pers, "xgboost": df_xgb, "lstm": df_lstm}

    for comuna_id in sorted(df["comuna_id"].unique()):
        exc      = config.get("excepciones", {})
        modelo   = exc.get(int(comuna_id), config["default"])
        df_m     = pred_map.get(modelo, pd.DataFrame())
        if df_m.empty:
            continue
        filas = df_m[df_m["comuna_id"] == comuna_id]
        if filas.empty:
            continue
        y_reals.extend(filas["y_real"].tolist())
        y_preds.extend(filas["y_pred"].tolist())

    return np.array(y_reals), np.array(y_preds)


# =============================================================================
# CÁLCULO DE MÉTRICAS
# =============================================================================

def mae_seguro(y_real, y_pred):
    """
    Calcula el MAE de forma segura — retorna NaN si no hay suficientes datos.
    Con pocos datos el MAE puede ser muy ruidoso y distorsionar los gráficos.
    """
    if len(y_real) < 5:
        return np.nan
    return mean_absolute_error(y_real, y_pred)


# =============================================================================
# PASO 2: WALK-FORWARD VALIDATION
#
# La validación walk-forward evalúa el sistema semana a semana.
# En lugar de un único período de evaluación, tenemos tantos folds
# como semanas hay en el dataset (menos el período inicial de entrenamiento).
#
# IMPORTANTE: no reentrenamos el modelo en cada fold porque eso tomaría
# horas para el LSTM. Evaluamos los modelos actuales (entrenados con 2023)
# sobre diferentes ventanas temporales.
#
# Parámetros clave:
#   min_train_semanas: mínimo de semanas necesarias para que el LSTM
#     tenga ventana suficiente (WINDOW_SIZE=12).
#   step: cada cuántas semanas avanzamos el fold.
#     step=1 → una evaluación por semana (muy granular)
#     step=4 → una evaluación por mes (más manejable)
#   eval_semanas: cuántas semanas evaluamos en cada fold.
# =============================================================================

def walk_forward_validation(df_total, modelos,
                             step=4,
                             min_train_semanas=26,
                             eval_semanas=4):
    """
    Evalúa los modelos con validación walk-forward sobre el dataset completo.

    Para cada fold:
    1. Define un subconjunto de evaluación de eval_semanas semanas
    2. Calcula las predicciones de ensemble, LSTM y persistencia
    3. Registra el MAE y el período evaluado

    El resultado es una serie temporal de MAE que permite detectar
    en qué semanas o períodos el sistema tiene más dificultades.

    Parámetros:
      step:             avance en semanas entre folds (default 4 = mensual)
      min_train_semanas: semanas mínimas antes del primer fold
      eval_semanas:     semanas por fold de evaluación
    """
    logger.info("--- PASO 2: Walk-forward validation ---")
    logger.info("  Step: %d semanas | Eval por fold: %d semanas", step, eval_semanas)

    # Obtener todas las semanas únicas en orden cronológico
    semanas = (df_total[["year", "epi_week"]]
               .drop_duplicates()
               .sort_values(["year", "epi_week"])
               .reset_index(drop=True))
    n_semanas = len(semanas)

    logger.info("  Total semanas disponibles: %d", n_semanas)
    logger.info("  Folds estimados: ~%d", (n_semanas - min_train_semanas) // step)

    resultados = []
    fold_num   = 0

    for inicio_eval in range(min_train_semanas, n_semanas - eval_semanas, step):
        fin_eval = min(inicio_eval + eval_semanas, n_semanas)

        # Semanas del fold de evaluación
        sem_eval = semanas.iloc[inicio_eval:fin_eval]
        if len(sem_eval) < 2:
            continue

        # Filtrar el subset de evaluación del dataset completo
        mask_eval = df_total.apply(
            lambda row: any(
                (row["year"] == s["year"] and row["epi_week"] == s["epi_week"])
                for _, s in sem_eval.iterrows()
            ),
            axis=1
        )
        df_eval = df_total[mask_eval].copy()

        if len(df_eval) < 5:
            continue

        # Período de evaluación para el registro
        primer_año  = int(sem_eval.iloc[0]["year"])
        primer_se   = int(sem_eval.iloc[0]["epi_week"])
        ultimo_año  = int(sem_eval.iloc[-1]["year"])
        ultimo_se   = int(sem_eval.iloc[-1]["epi_week"])
        label_fold  = f"{primer_año}-SE{primer_se:02d}"
        casos_media = df_eval["confirmed_cases"].mean()

        fold_num += 1

        # Evaluar los tres modelos sobre este fold
        for nombre_modelo in ["Ensemble", "LSTM+aug", "Persistencia"]:
            for horizonte in HORIZONTES:
                if nombre_modelo == "Persistencia":
                    yr, yp = pred_persistencia_df(df_eval, horizonte)
                elif nombre_modelo == "LSTM+aug":
                    yr, yp = pred_lstm_df(df_eval, modelos, horizonte)
                else:  # Ensemble
                    yr, yp = pred_ensemble_df(df_eval, modelos, horizonte)

                mae = mae_seguro(yr, yp)
                if not np.isnan(mae):
                    resultados.append({
                        "Fold":         fold_num,
                        "Label":        label_fold,
                        "Año_inicio":   primer_año,
                        "SE_inicio":    primer_se,
                        "Año_fin":      ultimo_año,
                        "SE_fin":       ultimo_se,
                        "Modelo":       nombre_modelo,
                        "Horizonte":    horizonte,
                        "MAE":          round(mae, 2),
                        "N":            len(yr),
                        "Casos_media":  round(casos_media, 2),
                    })

    df_res = pd.DataFrame(resultados)
    logger.info("  Walk-forward completado: %d folds | %d resultados",
                fold_num, len(df_res))
    return df_res


# =============================================================================
# PASO 3: VALIDACIÓN POR BLOQUES ANUALES
#
# Divide el dataset en bloques epidemiológicamente significativos:
#
# Bloque 1: train=2023 → eval=2024 S1
#   El escenario real del Sprint 5: un año de datos vs el brote más severo.
#
# Bloque 2: train=2023+2024S1 → eval=2024 S2
#   ¿Si incluyéramos el brote 2024 en el train, mejoraría en temporada baja?
#
# Bloque 3: train=2023+2024 → eval=2025
#   ¿Con dos años de historia el sistema generaliza mejor?
#
# Nota: para los bloques 2 y 3 los modelos no fueron reentrenados con esos
# datos adicionales — evaluamos los modelos originales sobre esos períodos.
# Esta es una limitación documentada: los resultados muestran el comportamiento
# out-of-sample de los modelos actuales, no de modelos reentrenados.
# =============================================================================

def validacion_bloques_anuales(df_total, modelos):
    """
    Evalúa los modelos en tres bloques anuales distintos.

    Cada bloque representa un escenario epidemiológico diferente:
    - Brote masivo (2024 S1): el caso más exigente
    - Temporada baja (2024 S2): casi cero casos
    - Brote moderado (2025): intensidad intermedia

    Esto permite responder: ¿el sistema funciona igual en todos los
    contextos epidemiológicos o tiene sesgo hacia algún tipo de período?
    """
    logger.info("--- PASO 3: Validación por bloques anuales ---")

    # Definir los bloques de evaluación
    # Cada bloque es una lista de (año, semana) que forman el fold
    bloques = [
        {
            "nombre":      "2024 S1 — Brote masivo",
            "descripcion": "Brote 2024 semanas 1-26 (máx 1391 casos/semana)",
            "año_eval":    2024,
            "se_min_eval": 1,
            "se_max_eval": 26,
        },
        {
            "nombre":      "2024 S2 — Temporada baja",
            "descripcion": "Post-brote 2024 semanas 27-52 (casi cero casos)",
            "año_eval":    2024,
            "se_min_eval": 27,
            "se_max_eval": 52,
        },
        {
            "nombre":      "2025 — Brote moderado",
            "descripcion": "Brote 2025 (intensidad intermedia)",
            "año_eval":    2025,
            "se_min_eval": 1,
            "se_max_eval": 52,
        },
    ]

    resultados = []

    for bloque in bloques:
        logger.info("  Bloque: %s", bloque["nombre"])

        # Filtrar el subset de evaluación de este bloque
        mask_eval = (
            (df_total["year"] == bloque["año_eval"]) &
            (df_total["epi_week"] >= bloque["se_min_eval"]) &
            (df_total["epi_week"] <= bloque["se_max_eval"])
        )
        df_eval = df_total[mask_eval].copy()

        if len(df_eval) == 0:
            logger.warning("  Sin datos para bloque: %s", bloque["nombre"])
            continue

        casos_media = df_eval["confirmed_cases"].mean()
        casos_max   = df_eval["confirmed_cases"].max()
        logger.info("  Filas: %d | Casos media: %.1f | máx: %.0f",
                    len(df_eval), casos_media, casos_max)

        # Evaluar los tres modelos
        for nombre_modelo in ["Ensemble", "LSTM+aug", "Persistencia"]:
            for horizonte in HORIZONTES:
                if nombre_modelo == "Persistencia":
                    yr, yp = pred_persistencia_df(df_eval, horizonte)
                elif nombre_modelo == "LSTM+aug":
                    yr, yp = pred_lstm_df(df_eval, modelos, horizonte)
                else:
                    yr, yp = pred_ensemble_df(df_eval, modelos, horizonte)

                mae = mae_seguro(yr, yp)
                if not np.isnan(mae):
                    resultados.append({
                        "Bloque":       bloque["nombre"],
                        "Descripcion":  bloque["descripcion"],
                        "Modelo":       nombre_modelo,
                        "Horizonte":    horizonte,
                        "MAE":          round(mae, 2),
                        "N":            len(yr),
                        "Casos_media":  round(casos_media, 2),
                        "Casos_max":    round(casos_max, 1),
                    })

    df_res = pd.DataFrame(resultados)
    logger.info("  Bloques completados: %d resultados", len(df_res))
    return df_res


# =============================================================================
# PASO 4: TABLAS DE RESULTADOS
# =============================================================================

def tabla_walk_forward(df_wf):
    """
    Tabla resumen del walk-forward: MAE promedio y desviación estándar
    por modelo y horizonte sobre todos los folds.

    La desviación estándar mide la ESTABILIDAD del modelo a lo largo
    del tiempo — un modelo estable tiene baja std aunque su MAE promedio
    sea algo más alto que otro menos estable.
    """
    print("\n" + "=" * 80)
    print("  WALK-FORWARD — MAE promedio y estabilidad por modelo y horizonte")
    print("  (promedio y desviación estándar sobre todos los folds)")
    print("=" * 80)

    h_labels = {1: "h=1", 2: "h=2", 3: "h=3", 4: "h=4"}

    print(f"\n  {'Modelo':<22}", end="")
    for h in HORIZONTES:
        print(f"  {h_labels[h]+' (μ±σ)':>16}", end="")
    print()
    print("  " + "-" * (22 + 18 * len(HORIZONTES)))

    for modelo in ["Ensemble", "LSTM+aug", "Persistencia"]:
        sub = df_wf[df_wf["Modelo"] == modelo]
        print(f"  {modelo:<22}", end="")
        for h in HORIZONTES:
            sub_h = sub[sub["Horizonte"] == h]["MAE"]
            if len(sub_h) > 0:
                print(f"  {sub_h.mean():>6.1f}±{sub_h.std():>5.1f}", end="")
            else:
                print(f"  {'---':>16}", end="")
        print()

    print("=" * 80 + "\n")


def tabla_bloques(df_bloques):
    """
    Tabla comparativa por bloque anual: muestra el MAE de cada modelo
    en cada contexto epidemiológico.

    Permite identificar en qué escenario cada modelo es más confiable:
    - ¿El ensemble es mejor durante el brote o durante la temporada baja?
    - ¿El LSTM mejora cuando hay más datos disponibles (2025)?
    """
    print("\n" + "=" * 82)
    print("  BLOQUES ANUALES — MAE por modelo, bloque y horizonte")
    print("=" * 82)

    bloques = df_bloques["Bloque"].unique()

    for bloque in bloques:
        df_b = df_bloques[df_bloques["Bloque"] == bloque]
        casos_med = df_b["Casos_media"].iloc[0]
        casos_max = df_b["Casos_max"].iloc[0]

        print(f"\n  {bloque} | media={casos_med:.1f} | máx={casos_max:.0f} casos/semana")
        print(f"  {'Modelo':<22}", end="")
        for h in HORIZONTES:
            print(f"  {'h='+str(h):>8}", end="")
        print()
        print("  " + "-" * (22 + 10 * len(HORIZONTES)))

        for modelo in ["Ensemble", "LSTM+aug", "Persistencia"]:
            sub = df_b[df_b["Modelo"] == modelo]
            prefijo = "→ " if modelo == "Ensemble" else "  "
            print(f"  {prefijo}{modelo:<20}", end="")
            for h in HORIZONTES:
                val = sub[sub["Horizonte"] == h]["MAE"].values
                print(f"  {val[0]:>8.2f}" if len(val) else f"  {'---':>8}", end="")
            print()

    print("=" * 82 + "\n")


# =============================================================================
# PASO 5: GRÁFICOS
# =============================================================================

def grafico_walk_forward_temporal(df_wf, horizonte=4):
    """
    Gráfico de líneas: MAE a lo largo del tiempo para un horizonte específico.

    El eje X es el tiempo (fold), el eje Y es el MAE.
    Permite visualizar si hay períodos específicos donde el sistema falla
    — por ejemplo, semanas de pico de brote donde el error aumenta.

    Una curva suave → sistema estable.
    Picos pronunciados → períodos difíciles (típicamente picos de brote).
    """
    df_h = df_wf[df_wf["Horizonte"] == horizonte].copy()
    if df_h.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    for modelo in ["Ensemble", "LSTM+aug", "Persistencia"]:
        sub = df_h[df_h["Modelo"] == modelo].sort_values("Fold")
        if sub.empty:
            continue
        ax.plot(sub["Fold"], sub["MAE"],
                label=modelo,
                color=COLORES.get(modelo, "#95A5A6"),
                linewidth=2.0 if modelo == "Ensemble" else 1.5,
                alpha=1.0 if modelo == "Ensemble" else 0.8)

    # Anotar el período del brote 2024
    brote_folds = df_h[
        (df_h["Año_inicio"] == 2024) & (df_h["SE_inicio"] <= 26)
    ]["Fold"].values
    if len(brote_folds) > 0:
        ax.axvspan(brote_folds.min(), brote_folds.max(),
                   alpha=0.1, color="red", label="Brote 2024 S1")

    ax.set_xlabel("Fold (ventana temporal)", fontsize=11)
    ax.set_ylabel("MAE (casos promedio)", fontsize=11)
    ax.set_title(
        f"Walk-forward — MAE a lo largo del tiempo (h={horizonte})\n"
        "Sprint 6 / HU7 · Validación cruzada temporal",
        fontweight="bold", fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / f"27_walkforward_h{horizonte}.png",
                dpi=150, bbox_inches="tight")
    plt.show()


def grafico_mae_por_bloque(df_bloques):
    """
    Gráfico de barras: MAE por bloque anual para h=3 y h=4.

    Permite comparar visualmente cómo se comporta cada modelo en los
    tres contextos epidemiológicos: brote masivo, temporada baja y
    brote moderado.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bloques   = df_bloques["Bloque"].unique()
    modelos   = ["Ensemble", "LSTM+aug", "Persistencia"]
    x         = np.arange(len(bloques))
    ancho     = 0.25

    for ax, horizonte in zip(axes, [3, 4]):
        df_h = df_bloques[df_bloques["Horizonte"] == horizonte]

        for i, modelo in enumerate(modelos):
            maes = [
                df_h[(df_h["Bloque"] == b) & (df_h["Modelo"] == modelo)]["MAE"].values[0]
                if not df_h[(df_h["Bloque"] == b) & (df_h["Modelo"] == modelo)].empty
                else np.nan
                for b in bloques
            ]
            ax.bar(x + i * ancho - ancho,
                   maes, ancho,
                   label=modelo,
                   color=COLORES.get(modelo, "#95A5A6"),
                   alpha=0.9 if modelo == "Ensemble" else 0.75,
                   edgecolor="black" if modelo == "Ensemble" else "white",
                   linewidth=1.5 if modelo == "Ensemble" else 0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [b.split("—")[0].strip() for b in bloques],
            rotation=15, ha="right", fontsize=9
        )
        ax.set_ylabel("MAE (casos promedio)", fontsize=10)
        ax.set_title(f"h={horizonte} — MAE por bloque anual", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Validación por bloques anuales — Ensemble vs LSTM vs Persistencia\n"
        "Sprint 6 / HU7",
        fontweight="bold", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "28_bloques_anuales_h3_h4.png",
                dpi=150, bbox_inches="tight")
    plt.show()


def grafico_estabilidad(df_wf):
    """
    Gráfico de boxplots: distribución del MAE por modelo y horizonte.

    Un boxplot angosto → modelo estable (bajo error en la mayoría de folds)
    Un boxplot ancho  → modelo inestable (alto error en algunos folds)
    Outliers hacia arriba → folds difíciles (picos de brote)

    Permite comparar no solo el MAE promedio sino la CONSISTENCIA de cada
    modelo a lo largo del tiempo.
    """
    fig, axes = plt.subplots(1, len(HORIZONTES), figsize=(14, 5))
    h_labels  = {1: "h=1", 2: "h=2", 3: "h=3", 4: "h=4"}
    modelos   = ["Ensemble", "LSTM+aug", "Persistencia"]

    for ax, horizonte in zip(axes, HORIZONTES):
        data   = [df_wf[(df_wf["Modelo"] == m) & (df_wf["Horizonte"] == horizonte)]["MAE"].values
                  for m in modelos]
        colors = [COLORES.get(m, "#95A5A6") for m in modelos]

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_xticklabels([m.split("+")[0] for m in modelos],
                           rotation=15, ha="right", fontsize=9)
        ax.set_title(h_labels[horizonte], fontweight="bold")
        ax.set_ylabel("MAE" if horizonte == 1 else "")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Estabilidad walk-forward — Distribución del MAE por fold\n"
        "Sprint 6 / HU7 · Cajas angostas = modelo estable",
        fontweight="bold", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "29_estabilidad_walkforward.png",
                dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_validacion_cruzada():
    """
    Ejecuta el pipeline completo de validación cruzada temporal.

    Combina walk-forward (estabilidad semana a semana) y bloques anuales
    (generalización entre períodos epidemiológicos distintos) sobre los
    tres modelos más relevantes del sistema de alertas.
    """
    print("\n" + "=" * 65)
    print("  SPRINT 6 / HU7 — Validación cruzada temporal")
    print("  Modelos: Ensemble | LSTM+aug | Persistencia")
    print("  Walk-forward (step=4 semanas) + Bloques anuales (3 períodos)")
    print("  SIN reentrenamiento — modelos del Sprint 5")
    print("=" * 65 + "\n")

    # Paso 1: cargar
    df_total, modelos = cargar_todo()

    # Paso 2: walk-forward
    df_wf = walk_forward_validation(
        df_total, modelos,
        step=4,
        min_train_semanas=26,
        eval_semanas=4
    )

    # Paso 3: bloques anuales
    df_bloques = validacion_bloques_anuales(df_total, modelos)

    # Paso 4: tablas
    tabla_walk_forward(df_wf)
    tabla_bloques(df_bloques)

    # Paso 5: gráficos
    grafico_walk_forward_temporal(df_wf, horizonte=4)
    grafico_walk_forward_temporal(df_wf, horizonte=3)
    grafico_mae_por_bloque(df_bloques)
    grafico_estabilidad(df_wf)

    # Paso 6: guardar
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_wf.to_csv(REPORTS_DIR / "validacion_walkforward.csv", index=False)
    df_bloques.to_csv(REPORTS_DIR / "validacion_bloques.csv", index=False)
    logger.info("  validacion_walkforward.csv guardado")
    logger.info("  validacion_bloques.csv guardado")

    print("\n✓ Validación cruzada temporal completada.")
    print("  Archivos generados:")
    print("  - reports/validacion_walkforward.csv")
    print("  - reports/validacion_bloques.csv")
    print("  - reports/figures/27_walkforward_h4.png")
    print("  - reports/figures/27_walkforward_h3.png")
    print("  - reports/figures/28_bloques_anuales_h3_h4.png")
    print("  - reports/figures/29_estabilidad_walkforward.png")

    return df_wf, df_bloques


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    df_wf, df_bloques = run_validacion_cruzada()
