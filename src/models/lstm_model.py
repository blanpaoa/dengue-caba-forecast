"""
================================================================================
Sprint 5 / HU6 — LSTM para predicción de brotes de dengue en CABA
================================================================================

¿QUÉ ES LSTM Y POR QUÉ LO USAMOS?
------------------------------------
LSTM significa Long Short-Term Memory (Memoria de Largo y Corto Plazo).
Es un tipo de red neuronal diseñada especialmente para aprender de secuencias
de datos en el tiempo — como las semanas epidemiológicas de dengue.

La diferencia clave con XGBoost y Random Forest:
  - XGBoost ve cada semana como una observación INDEPENDIENTE. Para darle
    contexto temporal, le agregamos manualmente los lags (cases_lag1, lag2...).
  - LSTM recibe directamente las últimas N semanas EN ORDEN y aprende solo
    qué patrones en esa secuencia son útiles para predecir el futuro.

Analogía: es como la diferencia entre leer palabras sueltas (XGBoost) vs.
leer una oración completa y entender su significado en contexto (LSTM).

¿POR QUÉ LSTM PODRÍA SER MEJOR PARA DENGUE A LARGO PLAZO?
-----------------------------------------------------------
El dengue tiene dinámicas temporales complejas:
  - Un brote tarda varias semanas en desarrollarse, alcanzar el pico y decaer.
  - El calor ACUMULADO de las últimas semanas importa, no solo el de hoy.
  - La VELOCIDAD de crecimiento (cuánto aumentan los casos semana a semana)
    puede predecir el pico antes de que ocurra.

XGBoost captura esto de forma indirecta con los lags. LSTM lo captura
directamente al procesar la secuencia completa de semanas.

En los resultados de XGBoost vimos que a horizontes largos (h=3, h=4) el modelo
depende casi solo de la estacionalidad. LSTM, al entender la dinámica temporal
completa, debería poder predecir mejor cuándo terminará un brote o cuándo
empezará el próximo — información más valiosa para el sistema de alertas.

¿QUÉ ARQUITECTURAS COMPARAMOS?
--------------------------------
A) LSTM simple (1 capa):
   Las últimas 8 semanas → [64 celdas de memoria] → predicción
   Más simple, menos parámetros, menor riesgo de memorizar el dataset.
   Buena para patrones de corto y mediano plazo.

B) LSTM apilado (2 capas):
   Las últimas 8 semanas → [64 celdas] → [32 celdas] → predicción
   La primera capa aprende patrones simples (subidas, bajadas).
   La segunda aprende patrones DE patrones (curvas de brote, mesetas).
   Más capacidad, pero puede sobreajustar con datasets pequeños.

Comparamos ambas para elegir la mejor para este dataset de 720 filas.

PIPELINE (orden de ejecución):
  Paso 1 → Cargar datos y normalizar features y target
  Paso 2 → Construir secuencias temporales (ventanas de 8 semanas)
  Paso 3 → Definir las dos arquitecturas LSTM
  Paso 4 → Entrenar cada arquitectura para cada horizonte (h=0,1,2,3,4)
  Paso 5 → Comparar resultados entre arquitecturas y contra Sprint 4-5
  Paso 6 → Guardar modelos, gráficos y métricas

PREREQUISITO:
  Ejecutar src/features/lags.py antes de este script.
  Requiere TensorFlow 2.x instalado en venv311.

NOTA SOBRE GPU:
  TensorFlow >= 2.11 no usa GPU en Windows nativo.
  El entrenamiento se hace en CPU — con 720 filas tarda ~20-40 minutos total.
  Los warnings de oneDNN y GPU son normales y no afectan los resultados.
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

# Silenciar los mensajes verbosos de TensorFlow que no son errores
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Semillas para reproducibilidad
# Sin esto, dos ejecuciones del mismo script pueden dar resultados distintos
# porque las redes neuronales se inicializan con pesos aleatorios.
tf.random.set_seed(42)
np.random.seed(42)


# =============================================================================
# CONFIGURACIÓN GENERAL
# Todos los parámetros importantes en un solo lugar para facilitar cambios.
# =============================================================================

# Rutas de archivos
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/saved")
FIGURES_DIR   = Path("reports/figures")

# AUGMENTATION: si existe train_augmented.parquet (generado por augmentation.py)
# lo usamos en lugar del train original. Contiene el train original más
# copias escaladas x2 y x3 de las semanas de brote para corregir el
# distribution shift respecto al brote masivo de 2024.
_train_aug = PROCESSED_DIR / "train_augmented.parquet"
TRAIN_FILE  = _train_aug if _train_aug.exists() else PROCESSED_DIR / "train.parquet"
VAL_FILE   = PROCESSED_DIR / "validation.parquet"  # 2024 semanas 1-26 (brote)
TEST_FILE  = PROCESSED_DIR / "test.parquet"        # 2024 S2 + 2025

TARGET_COL = "confirmed_cases"  # variable que queremos predecir
HORIZONTES = [1, 2, 3, 4]      # semanas adelante a evaluar

# VENTANA TEMPORAL: cuántas semanas mira el LSTM hacia atrás para predecir.
#
# VERSIÓN FINAL (v2+v3): ventana de 12 semanas (3 meses).
# Con 8 semanas el modelo no tenía suficiente contexto para entender la
# dinámica completa de un brote (ascenso, pico, descenso). Con 12 semanas
# puede ver el patrón completo y aprender cuándo un brote está acelerando
# o desacelerando — información clave para las alertas tempranas.
# Costo: de 660 → 600 secuencias de train, pérdida mínima.
WINDOW_SIZE = 12


# =============================================================================
# FEATURES PARA LSTM
#
# ¿Por qué usamos las versiones normalizadas (_norm)?
# Las redes neuronales son muy sensibles a la escala de los datos.
# Si una variable vale 1.391 (casos en pico de brote) y otra vale 0.003
# (semana_sin), la red neuronal dará automáticamente mucho más peso a la
# primera simplemente por su magnitud — no porque sea más informativa.
# La normalización elimina este problema llevando todas las variables
# a una escala comparable (media=0, desvío=1).
#
# Las columnas con sufijo _norm fueron generadas en lags.py con un
# StandardScaler ajustado solo con los datos de entrenamiento.
#
# La estacionalidad (semana_sin, semana_cos) ya está entre -1 y 1 por
# construcción matemática (seno y coseno), así que no necesita _norm.
# =============================================================================

FEATURES_LSTM = (
    # Historial de casos de la propia comuna (normalizados)
    # Son los predictores más fuertes a corto plazo
    ["cases_lag1_norm", "cases_lag2_norm", "cases_lag3_norm", "cases_lag4_norm",
     "incidencia_lag1_norm", "incidencia_lag2_norm",
     "incidencia_lag3_norm", "incidencia_lag4_norm"] +

    # Vecindad espacial: promedio de casos de comunas vecinas la semana pasada
    # Captura la dispersión geográfica del dengue entre comunas adyacentes
    ["casos_vecinas_lag1", "incidencia_vecinas_lag1"] +

    # Variables climáticas con lags (normalizadas)
    # El calor y la lluvia de semanas atrás favorecen la reproducción del mosquito
    [f"temp_mean_lag{i}_norm"             for i in range(1, 5)] +
    [f"precipitation_lag{i}_norm"         for i in range(1, 5)] +
    [f"humidity_mean_lag{i}_norm"         for i in range(1, 5)] +
    [f"heat_index_mean_lag{i}_norm"       for i in range(1, 5)] +
    [f"temp_mean_anomaly_lag{i}_norm"     for i in range(1, 5)] +
    [f"precipitation_anomaly_lag{i}_norm" for i in range(1, 5)] +
    [f"humidity_mean_anomaly_lag{i}_norm" for i in range(1, 5)] +

    # Estacionalidad: en qué momento del año estamos
    # semana_sin y semana_cos codifican la semana como un círculo continuo
    # para que el modelo entienda que diciembre y enero son consecutivos
    ["semana_sin", "semana_cos", "is_epidemic_season", "mes_aprox"] +

    # Características de la comuna
    ["es_comuna_1", "poblacion"]
)

# Métricas de referencia de los modelos anteriores para la tabla comparativa
SPRINT_REF = {
    "Persistencia (lag 1)": {"MAE": 16.81, "R2": 0.932},
    "Random Forest":        {"MAE": 24.93, "R2": 0.614},
    "XGBoost semana actual":{"MAE": 24.92, "R2": 0.580},
}

# Colores para los gráficos — consistentes con los scripts anteriores
COLOR_LSTM1 = "#8E44AD"   # violeta — LSTM simple
COLOR_LSTM2 = "#1ABC9C"   # verde   — LSTM apilado
COLOR_RF    = "#3498DB"   # azul    — Random Forest
COLOR_PERS  = "#F39C12"   # naranja — Persistencia


# =============================================================================
# PASO 1: CARGA Y NORMALIZACIÓN
#
# Para las redes neuronales necesitamos normalizar DOS cosas por separado:
#
# A) FEATURES: ya están normalizadas en lags.py (columnas con sufijo _norm).
#    El StandardScaler fue ajustado solo con train para evitar que el modelo
#    "vea" estadísticas del futuro durante el entrenamiento.
#
# B) TARGET (confirmed_cases): lo normalizamos aquí con MinMaxScaler.
#    MinMaxScaler lleva los valores al rango [0, 1]:
#      0 casos → 0.0
#      1391 casos (máximo del brote) → 1.0
#
#    Esto ayuda a la red neuronal a aprender más establo porque la capa
#    de salida (Dense con activación lineal) produce valores en cualquier
#    rango, y es más fácil aprender valores entre 0 y 1 que entre 0 y 1391.
#
#    Al final desnormalizamos las predicciones para obtener casos reales.
#
#    REGLA CRÍTICA: el MinMaxScaler se ajusta SOLO con los datos de train.
#    Usarlo con val o test sería trampa — el modelo vería estadísticas
#    del futuro durante el entrenamiento.
# =============================================================================

def cargar_y_normalizar():
    """
    Carga los tres conjuntos de datos y normaliza el target (confirmed_cases).

    Las features ya vienen normalizadas de lags.py (sufijo _norm).
    El target se normaliza aquí con MinMaxScaler ajustado solo con train.

    Retorna los datasets, la lista de features disponibles, el normalizador
    del target y los valores del target normalizados para cada conjunto.
    """
    logger.info("--- PASO 1: Cargando datos y normalizando el target ---")

    # Verificar que los archivos existen
    for archivo in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not archivo.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {archivo}\n"
                "Solución: ejecutá src/features/lags.py primero."
            )

    df_train = pd.read_parquet(TRAIN_FILE)
    df_val   = pd.read_parquet(VAL_FILE)
    df_test  = pd.read_parquet(TEST_FILE)

    logger.info(
        "  Entrenamiento: %d filas | Validación: %d | Test: %d",
        len(df_train), len(df_val), len(df_test)
    )

    # Identificar qué features de la lista están realmente en el dataset
    # Algunas columnas normalizadas pueden no existir si lags.py no las generó
    features_disponibles = [f for f in FEATURES_LSTM if f in df_train.columns]
    features_faltantes   = [f for f in FEATURES_LSTM if f not in df_train.columns]
    if features_faltantes:
        logger.warning(
            "  %d features no encontradas (primeras 3): %s",
            len(features_faltantes), features_faltantes[:3]
        )
    logger.info(
        "  Features disponibles: %d de %d solicitadas",
        len(features_disponibles), len(FEATURES_LSTM)
    )

    # Normalizar el target con MinMaxScaler (rango [0, 1])
    # fit_transform: aprende los parámetros (min, max) Y transforma — solo en train
    # transform: aplica los mismos parámetros aprendidos — en val y test
    target_scaler = MinMaxScaler()
    y_train_norm  = target_scaler.fit_transform(
        df_train[[TARGET_COL]].values
    ).flatten()
    y_val_norm    = target_scaler.transform(df_val[[TARGET_COL]].values).flatten()
    y_test_norm   = target_scaler.transform(df_test[[TARGET_COL]].values).flatten()

    logger.info(
        "  Target normalizado — train: [%.3f, %.3f] | val: [%.3f, %.3f]",
        y_train_norm.min(), y_train_norm.max(),
        y_val_norm.min(),   y_val_norm.max()
    )

    return (df_train, df_val, df_test,
            features_disponibles, target_scaler,
            y_train_norm, y_val_norm, y_test_norm)


# =============================================================================
# PASO 2: CONSTRUCCIÓN DE SECUENCIAS TEMPORALES
#
# XGBoost recibe: una fila = los datos de UNA semana
# LSTM recibe:    una ventana = los datos de las últimas WINDOW_SIZE semanas
#
# Para cada semana del dataset, construimos una "ventana deslizante":
#
#   Semana 9  (target) ← aprendemos a predecir esto
#   ↑
#   [sem2, sem3, sem4, sem5, sem6, sem7, sem8, sem9] ← esto es la entrada
#    ↑_________________WINDOW_SIZE=8 semanas__________↑
#
# Luego deslizamos la ventana:
#   Semana 10 (target)
#   [sem3, sem4, sem5, sem6, sem7, sem8, sem9, sem10] ← entrada
#
# IMPORTANTE: las ventanas se construyen DENTRO de cada comuna por separado.
# No mezclamos semanas de la Comuna 1 con semanas de la Comuna 5 —
# son series temporales independientes.
#
# Las primeras WINDOW_SIZE semanas de cada comuna no tienen suficiente
# historia y se descartan (igual que los NaN en lags).
#
# La forma del tensor de entrada al LSTM es:
#   (cantidad de muestras, WINDOW_SIZE, cantidad de features)
#   Ejemplo: (640, 8, 42) = 640 ventanas de 8 semanas con 42 variables cada una
# =============================================================================

def construir_secuencias(df, features, y_norm, horizonte=0):
    """
    Convierte el dataset plano en secuencias temporales para LSTM.

    Para cada (comuna, semana_t) construye:
      - Entrada X: datos de las semanas [t-8, t-7, ..., t-1] → forma (8, n_features)
      - Target y: casos en t (horizonte=0) o en t+h (horizonte=h)

    Ejemplo con horizonte=0 (semana actual), WINDOW_SIZE=8:
      Entrada: semanas 1 a 8 de la Comuna 1
      Target: casos de la semana 9 de la Comuna 1

    Ejemplo con horizonte=2 (2 semanas adelante):
      Entrada: semanas 1 a 8 de la Comuna 1
      Target: casos de la semana 11 de la Comuna 1

    Retorna:
      X: array 3D de forma (muestras, WINDOW_SIZE, n_features)
      y: array 1D de forma (muestras,) — valores normalizados
      idx: lista de índices originales (para alinear con el dataset)
    """
    X_list, y_list, idx_list = [], [], []

    for comuna_id in sorted(df["comuna_id"].unique()):

        # Extraer solo las filas de esta comuna, en orden cronológico
        mask    = df["comuna_id"] == comuna_id
        df_c    = df[mask].sort_values(["year", "epi_week"]).copy()
        idx_c   = df_c.index.tolist()

        # Matriz de features de esta comuna: (n_semanas, n_features)
        # Rellenamos NaN con 0 — la red neuronal no puede procesar NaN
        X_c = df_c[features].fillna(0).values

        # Vector de targets normalizados de esta comuna
        y_c = y_norm[mask.values]

        n = len(df_c)

        # Deslizar la ventana por todas las semanas de esta comuna
        for t in range(WINDOW_SIZE, n):

            # El target está en t + horizonte
            t_target = t + horizonte
            if t_target >= n:
                continue  # no hay datos futuros disponibles

            # Ventana de entrada: las WINDOW_SIZE semanas anteriores a t
            ventana = X_c[t - WINDOW_SIZE : t]   # forma: (WINDOW_SIZE, n_features)

            # Valor a predecir (ya normalizado)
            target_val = y_c[t_target]

            if not np.isnan(target_val):
                X_list.append(ventana)
                y_list.append(target_val)
                idx_list.append(idx_c[t])

    if len(X_list) == 0:
        raise ValueError(
            "No se generaron secuencias de entrenamiento.\n"
            f"Verificá que WINDOW_SIZE={WINDOW_SIZE} no sea mayor que "
            "la cantidad de semanas disponibles por comuna."
        )

    X = np.array(X_list, dtype=np.float32)   # (muestras, WINDOW_SIZE, features)
    y = np.array(y_list,  dtype=np.float32)   # (muestras,)

    logger.info(
        "  Secuencias construidas: %d ventanas | forma: %s",
        len(X), str(X.shape)
    )

    return X, y, idx_list


# =============================================================================
# PASO 3: ARQUITECTURAS LSTM
#
# ¿Qué son los componentes de cada arquitectura?
#
# LSTM(units):
#   La capa recurrente principal. 'units' es la cantidad de "celdas de memoria"
#   — como neuronas que pueden recordar información de semanas anteriores.
#   Más unidades = más capacidad, pero más riesgo de memorizar el dataset.
#   Con solo 720 filas usamos 64 y 32 (valores conservadores).
#
# Dropout(rate):
#   Durante el entrenamiento, apaga aleatoriamente 'rate'% de las neuronas.
#   Por ejemplo, Dropout(0.2) apaga el 20% de neuronas en cada paso.
#   Esto FUERZA al modelo a no depender de ninguna neurona individual,
#   haciendo el modelo más robusto y generalizable.
#   Es la técnica más común para evitar sobreajuste en redes neuronales.
#
# Dense(n, activation):
#   Capa completamente conectada (todas las neuronas conectadas con todas).
#   Dense(32, activation="relu"): 32 neuronas que aprenden combinaciones
#     de los patrones del LSTM. relu = max(0, x), la activación más común.
#   Dense(1, activation="linear"): capa de salida, produce 1 número = casos.
#
# return_sequences=True:
#   Por defecto el LSTM devuelve solo el último paso temporal (el resumen).
#   Con return_sequences=True devuelve TODA la secuencia paso a paso.
#   Necesario cuando hay una segunda capa LSTM que necesita ver la secuencia.
# =============================================================================

def construir_lstm_simple(n_features, nombre="LSTM simple"):
    """
    Arquitectura LSTM de una sola capa con transición gradual hacia la salida.

    Flujo de información:
      Entrada (8 semanas × n_features)
        → LSTM 64 celdas     (resume la secuencia en 64 patrones temporales)
        → Dropout 20%        (regularización)
        → Dense 32 + relu    (combina los 64 patrones en 32 representaciones)
        → Dropout 20%        (regularización)
        → Dense 16 + relu    (refina a 16 combinaciones más relevantes)
        → Dropout 20%        (regularización)
        → Dense 1 + linear   (predicción final: 1 número = casos normalizados)

    ¿Por qué la transición gradual 64→32→16→1?
    Bajar directamente de 64 a 1 es un salto muy abrupto — el modelo
    tiene que comprimir demasiada información en un solo paso y puede
    perder patrones importantes. La transición gradual le da espacio
    para aprender combinaciones intermedias antes de la predicción final.

    ¿Por qué relu en las capas intermedias y linear en la salida?
    relu (Rectified Linear Unit): f(x) = max(0, x)
      Introduce no-linealidad — sin esto, apilar capas Dense sería
      equivalente a tener una sola capa (las transformaciones lineales
      se cancelan entre sí). relu es la activación más usada en la práctica.
    linear: no aplica ninguna transformación — el valor sale tal cual.
      Correcto para regresión: los casos de dengue pueden ser cualquier
      número positivo, sin un rango acotado como tendría sigmoid (0-1)
      o tanh (-1 a 1).

    ¿Por qué Dropout en todas las capas intermedias?
    Cada capa puede desarrollar dependencias de neuronas específicas.
    El Dropout en cada capa fuerza a que TODAS las capas sean robustas
    y no dependan de neuronas individuales — no solo la primera.

    Ventajas de esta arquitectura:
      - Menos parámetros que el apilado → menor riesgo de sobreajuste
      - Transición gradual → mejor capacidad de abstracción
      - Más rápida de entrenar
      - Generalmente mejor con datasets pequeños (720 filas)
    """
    # VERSIÓN FINAL (v2+v3): LSTM 32 unidades, Dropout 0.1, transición 32→16→8→1
    # Con 64 unidades y Dropout 0.2 el modelo sobreajustaba (train→0.01, val→0.09).
    # Con 32 unidades y Dropout 0.1 la brecha se reduce considerablemente.
    modelo = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),

        # LSTM 32 celdas — más conservador para dataset de 600 muestras
        LSTM(32, return_sequences=False),
        Dropout(0.1),  # Dropout suave: 10% (antes 20%, demasiado agresivo)

        # Transición gradual: 32 → 16 → 8 → 1
        Dense(16, activation="relu"),
        Dropout(0.1),

        Dense(8, activation="relu"),
        Dropout(0.1),

        # Capa de salida: predicción final en escala normalizada
        Dense(1, activation="linear")

    ], name=nombre.replace(" ", "_"))

    return modelo


def construir_lstm_apilado(n_features, nombre="LSTM apilado"):
    """
    Arquitectura LSTM con dos capas apiladas y transición gradual hacia la salida.

    Flujo de información:
      Entrada (8 semanas × n_features)
        → LSTM 64 celdas     (aprende patrones simples: subidas, bajadas, tendencias)
        → Dropout 20%
        → LSTM 32 celdas     (aprende patrones DE patrones: curvas de brote, mesetas)
        → Dropout 20%
        → Dense 16 + relu    (combina los patrones LSTM en representaciones densas)
        → Dropout 20%
        → Dense 8 + relu     (refina a las 8 combinaciones más relevantes)
        → Dropout 20%
        → Dense 1 + linear   (predicción final)

    ¿Por qué dos capas LSTM?
      Primera capa LSTM (64): procesa la secuencia semana a semana y aprende
        patrones simples — "los casos subieron 3 semanas seguidas",
        "el heat index estuvo alto durante todo el mes".
      Segunda capa LSTM (32): recibe los patrones de la primera capa como
        una nueva secuencia y aprende relaciones entre ellos —
        "cuando el crecimiento fue exponencial + heat index alto → brote inminente".
      Es como leer un texto dos veces: la primera para entender las palabras,
      la segunda para entender el significado del párrafo completo.

    ¿Por qué la transición gradual 32→16→8→1?
      Después de las capas LSTM, tenemos 32 representaciones temporales.
      Bajar directamente de 32 a 1 es demasiado abrupto — el modelo tiene
      que comprimir toda esa información en un solo paso.
      La transición 32→16→8→1 permite aprender combinaciones progresivamente
      más abstractas antes de la predicción final.

    ¿Por qué Dropout en TODAS las capas intermedias?
      Sin Dropout, cada capa puede volverse dependiente de neuronas específicas.
      Con Dropout en todas las capas, el modelo aprende representaciones
      distribuidas y robustas en cada nivel de abstracción.

    Ventajas:
      - Mayor capacidad para capturar dinámicas temporales complejas
      - Transición gradual evita pérdida de información
      - Puede detectar patrones de orden superior (aceleración, inflexión del brote)

    Riesgo:
      - Con 720 filas puede sobreajustar — el Dropout en todas las capas lo mitiga
      - Más lento de entrenar que el simple
    """
    # VERSIÓN FINAL (v2+v3): LSTM 32+16 unidades, Dropout 0.1, transición 16→8→1
    # Reducimos de 64+32 → 32+16 para evitar sobreajuste con 600 muestras.
    modelo = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),

        # Primera capa LSTM: 32 celdas — aprende patrones simples
        # return_sequences=True: pasa la secuencia completa a la segunda capa
        LSTM(32, return_sequences=True),
        Dropout(0.1),

        # Segunda capa LSTM: 16 celdas — aprende patrones de patrones
        # return_sequences=False: devuelve solo el resumen final
        LSTM(16, return_sequences=False),
        Dropout(0.1),

        # Transición gradual: 16 → 8 → 1
        Dense(8, activation="relu"),
        Dropout(0.1),

        # Capa de salida
        Dense(1, activation="linear")

    ], name=nombre.replace(" ", "_"))

    return modelo


# =============================================================================
# CÁLCULO DE MÉTRICAS DE ERROR
# Siempre en escala original (casos reales, ya desnormalizados).
# Las mismas métricas que en los modelos anteriores para comparación directa.
# =============================================================================

def calcular_metricas(y_real, y_pred, modelo, split, horizonte):
    """
    Calcula MAE, RMSE, R² y MAPE.
    Tanto y_real como y_pred deben estar en escala original de casos
    (ya desnormalizados — de vuelta a 0..1391 casos).
    """
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2   = r2_score(y_real, y_pred)

    # MAPE solo donde hay casos reales (evita división por cero)
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
# PASO 4: ENTRENAMIENTO
#
# ¿Qué es una época?
# Una época = el modelo procesa TODOS los datos de entrenamiento una vez.
# Con 200 épocas máximas, el modelo vería los datos 200 veces — pero el
# early stopping lo detiene antes si deja de mejorar.
#
# ¿Qué es el batch_size?
# En lugar de actualizar los pesos después de cada muestra (lento) o después
# de todas las muestras (impreciso), procesamos mini-lotes de 32 muestras
# a la vez. batch_size=32 es el valor estándar para datasets medianos.
#
# Callbacks (acciones automáticas durante el entrenamiento):
#
# EarlyStopping: vigila el error en validación. Si no mejora en 20 épocas
#   consecutivas, detiene el entrenamiento y restaura el mejor modelo.
#   Esto evita sobreajuste y ahorra tiempo — no tiene sentido seguir
#   entrenando si el modelo ya alcanzó su mejor punto.
#
# ReduceLROnPlateau: si el error se estanca (no mejora en 10 épocas),
#   reduce la tasa de aprendizaje a la mitad. Es como ir más despacio
#   cuando el camino se pone difícil — pasos más pequeños para no pasarse.
#   La tasa mínima es 1e-6 para no llegar a pasos tan pequeños que no avancen.
# =============================================================================

def calcular_sample_weights(y_train_norm, scaler):
    """
    Calcula pesos por muestra para corregir el desbalanceo del dataset.

    PROBLEMA: el 60% de las semanas tienen 0 casos. El modelo aprende que
    predecir cero siempre minimiza el error promedio — y técnicamente tiene
    razón para el 60% de los casos. Pero eso es inútil para detectar brotes.

    SOLUCIÓN: darle más peso a las semanas con casos durante el entrenamiento.
    Las 16 filas de brote severo (>200 casos) pesan igual que las 468 de cero
    — eso hay que corregirlo para que el modelo "preste más atención" a los
    brotes, que son el evento que más importa predecir.

    Fórmula: peso = log(1 + casos_reales) + 1
      0 casos   → peso = log(1) + 1 = 1.0  (peso base)
      10 casos  → peso = log(11) + 1 ≈ 3.4
      50 casos  → peso = log(51) + 1 ≈ 4.9
      500 casos → peso = log(501) + 1 ≈ 7.2
      1391 casos→ peso = log(1392)+ 1 ≈ 8.2

    El logaritmo suaviza la diferencia — no queremos que los picos del brote
    dominen completamente el entrenamiento, sino que tengan más presencia.
    """
    # Desnormalizar para obtener los casos reales y calcular los pesos
    y_real = scaler.inverse_transform(
        y_train_norm.reshape(-1, 1)
    ).flatten().clip(min=0)

    pesos = np.log1p(y_real) + 1.0
    return pesos


def entrenar_modelo(modelo, X_train, y_train, X_val, y_val,
                    nombre_arquitectura, horizonte, scaler):
    """
    Compila y entrena un modelo LSTM con:
      - Huber loss (más robusta que MSE para distribuciones desbalanceadas)
      - Sample weights (más peso a semanas de brote)
      - Early stopping y ReduceLROnPlateau

    AJUSTE v3 — DOS MEJORAS PARA EL DESBALANCEO:

    1. HUBER LOSS (reemplaza MSE):
       MSE = (predicción - real)² → penaliza enormemente los errores grandes.
       Con 60% de ceros, el modelo aprende a predecir cerca de cero siempre
       porque eso minimiza el MSE. Los pocos errores grandes del brote no
       alcanzan a compensar.

       Huber loss combina MSE y MAE:
         Si error < delta → usa MSE (sensible a diferencias pequeñas)
         Si error > delta → usa MAE (menos catastrófico con errores grandes)

       delta=0.1 en escala normalizada ≈ 130 casos reales con el scaler actual.
       Esto significa: errores de hasta ~130 casos → MSE (aprendizaje preciso),
       errores mayores → MAE (no colapsa el entrenamiento).

    2. SAMPLE WEIGHTS (pesos por muestra):
       Cada secuencia de entrenamiento recibe un peso proporcional al
       log(1 + casos_reales) + 1. Las semanas de brote tienen más peso
       y el modelo les presta más atención durante el entrenamiento.

       Con 60% de ceros y solo 2.1% de brotes severos:
         Sin weights: el modelo ve 28x más ejemplos sin brote que con brote
         Con weights: la brecha efectiva se reduce significativamente

    Retorna el modelo entrenado y el historial de entrenamiento.
    """
    # Calcular pesos por muestra para corregir el desbalanceo
    sample_weights = calcular_sample_weights(y_train, scaler)

    logger.info(
        "  Pesos — min: %.2f | max: %.2f | media: %.2f",
        sample_weights.min(), sample_weights.max(), sample_weights.mean()
    )

    # Compilar con Huber loss en lugar de MSE
    # delta=0.1 en escala normalizada es el umbral entre MSE y MAE
    modelo.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=tf.keras.losses.Huber(delta=0.1),   # AJUSTE: Huber en lugar de MSE
        metrics=["mae"]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=30,                 # más paciencia con lr bajo
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0
        )
    ]

    logger.info(
        "  Entrenando %s h=%d | Train: %s | Val: %s",
        nombre_arquitectura, horizonte, X_train.shape, X_val.shape
    )

    historia = modelo.fit(
        X_train, y_train,
        sample_weight=sample_weights,    # AJUSTE: pesos por muestra
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=16,
        callbacks=callbacks,
        verbose=0
    )

    n_epocas       = len(historia.history["loss"])
    mejor_val_loss = min(historia.history["val_loss"])
    logger.info(
        "  Completado — épocas: %d | mejor val_loss: %.5f",
        n_epocas, mejor_val_loss
    )

    return modelo, historia


# =============================================================================
# PASO 5: TABLAS DE COMPARACIÓN
# =============================================================================

def tabla_comparacion_general(todas_metricas):
    """
    Tabla comparativa de LSTM vs. todos los modelos anteriores.
    Muestra solo el período de validación (brote 2024 S1) porque es
    el período más relevante y donde la comparación es más justa.
    """
    print("\n" + "=" * 82)
    print("  LSTM — Comparación con modelos anteriores")
    print("  Período: validación (brote 2024, semanas 1-26)")
    print("=" * 82)
    print(f"  {'Modelo':<32} {'Horizonte':>18} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("  " + "-" * 78)

    print("  --- Referencia Sprint 4-5 (h=semana actual) ---")
    for nombre, m in SPRINT_REF.items():
        print(
            f"  {nombre:<32} {'semana actual':>18} "
            f"{m['MAE']:>8.2f} {'---':>8} {m['R2']:>8.3f}"
        )

    print("  --- LSTM Sprint 5 ---")
    for m in todas_metricas:
        if m["Split"] == "Validation":
            h_str = "semana actual" if m["Horizonte"] == 0 else f"{m['Horizonte']} sem. adelante"
            print(
                f"  {m['Modelo']:<32} {h_str:>18} "
                f"{m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['R2']:>8.3f}"
            )

    print("=" * 82)
    print("  MAE = error promedio en casos | R² = varianza explicada (1=perfecto)")
    print("=" * 82 + "\n")


def tabla_degradacion_por_horizonte(todas_metricas):
    """
    Muestra cómo cambian MAE y R² al predecir más lejos en el tiempo.
    Si el LSTM mantiene métricas aceptables en h=3 y h=4 (donde XGBoost
    dependía casi solo de la estacionalidad), confirma su valor para el
    sistema de alertas tempranas.
    """
    print("\n" + "=" * 72)
    print("  LSTM — Degradación de precisión por horizonte")
    print("  ¿Cuánto empeora el modelo al predecir más lejos?")
    print("=" * 72)

    for arquitectura in ["LSTM simple", "LSTM apilado"]:
        m_arq = [m for m in todas_metricas
                 if m["Modelo"] == arquitectura and m["Split"] == "Validation"]
        m_h   = {m["Horizonte"]: m for m in m_arq}

        print(f"\n  {arquitectura}:")
        print(f"  {'Métrica':<10} {'actual':>10} {'h=1':>10} {'h=2':>10} {'h=3':>10} {'h=4':>10}")
        print("  " + "-" * 62)

        fila_mae = f"  {'MAE':<10}"
        fila_r2  = f"  {'R²':<10}"
        for h in [0, 1, 2, 3, 4]:
            if h in m_h:
                fila_mae += f" {m_h[h]['MAE']:>10.2f}"
                fila_r2  += f" {m_h[h]['R2']:>10.3f}"
            else:
                fila_mae += f" {'n/a':>10}"
                fila_r2  += f" {'n/a':>10}"
        print(fila_mae)
        print(fila_r2)

    print("\n" + "=" * 72 + "\n")


# =============================================================================
# PASO 6: GRÁFICOS
# =============================================================================

def graficar_historia_entrenamiento(historias, nombre_arquitectura):
    """
    Grafica cómo evolucionó el error durante el entrenamiento.

    ¿Cómo interpretar este gráfico?
      - Curva azul (Train): error en los datos de entrenamiento
      - Curva roja (Val): error en los datos de validación

    Situaciones posibles:
      ✓ Ambas bajan juntas → el modelo generaliza bien
      ✗ Train baja pero Val sube → sobreajuste (el modelo memoriza el train)
      ✗ Ambas se estancan desde el inicio → el modelo no aprende nada útil

    El punto donde la curva de validación alcanza su mínimo es donde el
    early stopping guarda el modelo (el mejor durante todo el entrenamiento).
    """
    horizontes_label = {0: "semana actual", 1: "h=1", 2: "h=2", 3: "h=3", 4: "h=4"}

    n = len(historias)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (h, historia) in zip(axes, historias.items()):
        epocas = range(1, len(historia.history["loss"]) + 1)
        ax.plot(epocas, historia.history["loss"],     label="Train (entrenamiento)",
                color="#3498DB", linewidth=1.5)
        ax.plot(epocas, historia.history["val_loss"], label="Val (validación)",
                color="#E74C3C", linewidth=1.5)
        ax.set_xlabel("Época (pasada por todos los datos)")
        ax.set_ylabel("Error cuadrático medio (MSE)")
        ax.set_title(
            f"Horizonte: {horizontes_label.get(h, f'h={h}')}\n"
            f"Épocas entrenadas: {len(epocas)}",
            fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(
        f"{nombre_arquitectura} — Curva de aprendizaje\nSprint 5 / HU6",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    nombre_archivo = nombre_arquitectura.lower().replace(" ", "_")
    plt.savefig(
        FIGURES_DIR / f"18_aprendizaje_{nombre_archivo}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()


def graficar_degradacion_lstm(todas_metricas):
    """
    Curva de degradación por horizonte para ambas arquitecturas.
    Incluye líneas de referencia de modelos anteriores.

    ¿Cómo interpretar?
      - Si LSTM simple ≈ LSTM apilado → la complejidad extra no aporta
      - Si LSTM supera a XGBoost en h=3 o h=4 → justifica su uso
      - Si LSTM cae por debajo de R²=0 → no mejor que predecir el promedio
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    estilos = [
        ("LSTM simple",  COLOR_LSTM1, "o-"),
        ("LSTM apilado", COLOR_LSTM2, "s-"),
    ]

    for arquitectura, color, estilo in estilos:
        m_arq = [m for m in todas_metricas
                 if m["Modelo"] == arquitectura
                 and m["Split"] == "Validation"
                 and m["Horizonte"] > 0]
        if not m_arq:
            continue

        m_ord = sorted(m_arq, key=lambda x: x["Horizonte"])
        hs    = [m["Horizonte"] for m in m_ord]
        maes  = [m["MAE"] for m in m_ord]
        r2s   = [m["R2"]  for m in m_ord]

        ax1.plot(hs, maes, estilo, color=color, linewidth=2.5,
                 markersize=9, label=arquitectura, zorder=3)
        ax2.plot(hs, r2s,  estilo, color=color, linewidth=2.5,
                 markersize=9, label=arquitectura, zorder=3)

        # Anotar valores
        for h, mae, r2 in zip(hs, maes, r2s):
            ax1.annotate(f"{mae:.1f}", (h, mae),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9, color=color, fontweight="bold")
            ax2.annotate(f"{r2:.3f}", (h, r2),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9, color=color, fontweight="bold")

    # Líneas de referencia de Sprint 4-5
    for ax, key in [(ax1, "MAE"), (ax2, "R2")]:
        ax.axhline(
            SPRINT_REF["Persistencia (lag 1)"][key],
            color=COLOR_PERS, linestyle="--", linewidth=1.5,
            label="Persistencia Sprint 4"
        )
        ax.axhline(
            SPRINT_REF["Random Forest"][key],
            color=COLOR_RF, linestyle="--", linewidth=1.5,
            label="Random Forest Sprint 4"
        )
        if key == "R2":
            ax.axhline(0, color="gray", linestyle=":", linewidth=1,
                       label="R²=0 (sin mejora)")

    ax1.set_xlabel("Horizonte de predicción (semanas)")
    ax1.set_ylabel("Error promedio (MAE en casos)")
    ax1.set_title("Error por horizonte\n(menor es mejor)", fontweight="bold")
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xticklabels(["1 sem.", "2 sem.", "3 sem.", "4 sem."])
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Horizonte de predicción (semanas)")
    ax2.set_ylabel("R² (varianza explicada)")
    ax2.set_title("Varianza explicada por horizonte\n(mayor es mejor, máximo=1.0)",
                  fontweight="bold")
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xticklabels(["1 sem.", "2 sem.", "3 sem.", "4 sem."])
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle(
        "LSTM — Impacto del horizonte de predicción\nSprint 5 / HU6",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "19_degradacion_lstm.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# PIPELINE PRINCIPAL
# Orquesta todos los pasos en orden para ambas arquitecturas y todos los
# horizontes de predicción.
# =============================================================================

def run_lstm():
    """
    Ejecuta el pipeline completo de LSTM.

    Entrena 10 modelos en total:
      2 arquitecturas (simple, apilado) × 5 horizontes (actual, h=1,2,3,4)

    Cada modelo usa early stopping — se detiene automáticamente cuando
    el error en validación deja de mejorar.

    Mejoras v3: Huber loss + sample weights para corregir desbalanceo (60% ceros).
    Tiempo estimado en CPU: 30-50 minutos.
    """
    print("\n" + "=" * 65)
    print("  SPRINT 5 / HU6 — LSTM")
    print("  Comparando: LSTM simple (1 capa) vs LSTM apilado (2 capas) — v FINAL")
    print(f"  Horizontes: semana actual + h=1, h=2, h=3, h=4")
    print(f"  Ventana temporal: {WINDOW_SIZE} semanas de historia")
    print("=" * 65 + "\n")

    # ── Paso 1: carga y normalización ────────────────────────────────
    (df_train, df_val, df_test,
     features, target_scaler,
     y_train_norm, y_val_norm, y_test_norm) = cargar_y_normalizar()

    n_features = len(features)
    logger.info(
        "  Entrada al LSTM: ventanas de %d semanas × %d features",
        WINDOW_SIZE, n_features
    )

    # Las dos arquitecturas a comparar
    arquitecturas = {
        "LSTM simple":  construir_lstm_simple,
        "LSTM apilado": construir_lstm_apilado,
    }

    todas_metricas    = []
    modelos_guardados = {}
    historias_por_arq = {nombre: {} for nombre in arquitecturas}

    # ── Pasos 2-4: entrenar por horizonte ────────────────────────────
    # Horizonte 0 = semana actual (confirmed_cases, comparable con Sprint 4)
    # Horizontes 1-4 = semanas futuras (target_h1..h4)
    for horizonte in [0] + HORIZONTES:

        if horizonte == 0:
            target_col = TARGET_COL
            label_h    = "semana actual"
            y_tr = y_train_norm
            y_vl = y_val_norm
            y_te = y_test_norm
            h_scaler = target_scaler
        else:
            target_col = f"target_h{horizonte}"
            label_h    = f"{horizonte} semana(s) adelante"
            # Normalizar el target del horizonte correspondiente con MinMaxScaler
            # ajustado SOLO con los datos de train
            col_tr = df_train[target_col].fillna(0).values.reshape(-1, 1)
            col_vl = df_val[target_col].fillna(0).values.reshape(-1, 1)
            col_te = df_test[target_col].fillna(0).values.reshape(-1, 1)
            h_scaler = MinMaxScaler()
            y_tr = h_scaler.fit_transform(col_tr).flatten()
            y_vl = h_scaler.transform(col_vl).flatten()
            y_te = h_scaler.transform(col_te).flatten()

        logger.info("=" * 60)
        logger.info("  HORIZONTE: %s", label_h.upper())
        logger.info("=" * 60)

        # Paso 2: construir secuencias temporales para este horizonte
        logger.info("  Construyendo ventanas de %d semanas...", WINDOW_SIZE)
        X_train_seq, y_train_seq, _ = construir_secuencias(
            df_train, features, y_tr, horizonte=0
        )
        X_val_seq, y_val_seq, _   = construir_secuencias(
            df_val, features, y_vl, horizonte=0
        )
        X_test_seq, y_test_seq, _ = construir_secuencias(
            df_test, features, y_te, horizonte=0
        )

        # Paso 3-4: entrenar cada arquitectura con las mismas secuencias
        for nombre_arq, constructor in arquitecturas.items():

            logger.info("  ── %s ──", nombre_arq)

            # Construir modelo FRESCO para este horizonte
            # (no heredar pesos del horizonte anterior)
            modelo = constructor(n_features)

            # Entrenar con Huber loss + sample weights
            modelo, historia = entrenar_modelo(
                modelo,
                X_train_seq, y_train_seq,
                X_val_seq,   y_val_seq,
                nombre_arq,  horizonte,
                h_scaler     # para calcular pesos en escala real
            )

            # Guardar historia para graficar la curva de aprendizaje
            historias_por_arq[nombre_arq][horizonte] = historia

            # ── Predicciones y desnormalización ──────────────────────
            # El modelo predice valores entre 0 y 1 (normalizados).
            # Desnormalizamos para obtener casos reales (0..1391).
            def desnorm(y_norm_pred, scaler):
                """Revierte la normalización MinMaxScaler → casos reales."""
                return scaler.inverse_transform(
                    y_norm_pred.reshape(-1, 1)
                ).flatten().clip(min=0)

            pred_val_real  = desnorm(modelo.predict(X_val_seq,  verbose=0).flatten(), h_scaler)
            pred_test_real = desnorm(modelo.predict(X_test_seq, verbose=0).flatten(), h_scaler)
            y_val_real     = desnorm(y_val_seq,  h_scaler)
            y_test_real    = desnorm(y_test_seq, h_scaler)

            # Calcular métricas en escala original de casos
            met_val  = calcular_metricas(
                y_val_real,  pred_val_real,  nombre_arq, "Validation", horizonte
            )
            met_test = calcular_metricas(
                y_test_real, pred_test_real, nombre_arq, "Test",       horizonte
            )
            todas_metricas.extend([met_val, met_test])

            logger.info(
                "  VAL  — MAE: %.2f | RMSE: %.2f | R²: %.3f",
                met_val["MAE"], met_val["RMSE"], met_val["R2"]
            )
            logger.info(
                "  TEST — MAE: %.2f | RMSE: %.2f | R²: %.3f",
                met_test["MAE"], met_test["RMSE"], met_test["R2"]
            )

            # Guardar referencia al modelo para guardarlo al final
            clave = f"{nombre_arq.lower().replace(' ', '_')}_h{horizonte}"
            modelos_guardados[clave] = modelo

    # ── Paso 5: tablas de comparación ────────────────────────────────
    tabla_comparacion_general(todas_metricas)
    tabla_degradacion_por_horizonte(todas_metricas)

    # ── Paso 6: gráficos ─────────────────────────────────────────────

    # Curvas de aprendizaje para cada arquitectura (horizontes 0 y 1)
    for nombre_arq, historias_h in historias_por_arq.items():
        historias_sel = {
            h: historias_h[h]
            for h in [0, 1]
            if h in historias_h
        }
        if historias_sel:
            graficar_historia_entrenamiento(historias_sel, nombre_arq)

    # Curva de degradación por horizonte
    graficar_degradacion_lstm(todas_metricas)

    # ── Guardar modelos y resultados ──────────────────────────────────
    logger.info("--- Guardando modelos y resultados ---")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Guardar cada modelo en formato .keras (formato nativo de TensorFlow)
    for clave, modelo in modelos_guardados.items():
        ruta = MODELS_DIR / f"lstm_{clave}.keras"
        modelo.save(ruta)
        logger.info("  Guardado: %s", ruta.name)

    # Guardar el normalizador del target para usar en predicciones futuras
    with open(MODELS_DIR / "lstm_target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)
    logger.info("  Normalizador guardado: lstm_target_scaler.pkl")

    # Guardar métricas en CSV para análisis y comparación posterior
    df_metricas = pd.DataFrame(todas_metricas)
    df_metricas.to_csv(MODELS_DIR / "metricas_sprint5_lstm.csv", index=False)
    logger.info("  Métricas guardadas: metricas_sprint5_lstm.csv")

    print("\n✓ Sprint 5 / HU6 — LSTM completado.")
    print("  LSTM simple  ✓ | LSTM apilado ✓")
    print("  Horizontes evaluados: semana actual + h=1, h=2, h=3, h=4")
    print("  Próximo paso: GRU (Gated Recurrent Unit)")

    return {"modelos": modelos_guardados, "metricas": todas_metricas}


# =============================================================================
# PUNTO DE ENTRADA
# Se ejecuta cuando corrés: python src/models/lstm.py
# =============================================================================

if __name__ == "__main__":
    resultado = run_lstm()
