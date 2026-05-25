"""
================================================================================
Sprint 5 / HU6 — GRU para predicción de brotes de dengue en CABA
================================================================================

¿QUÉ ES GRU?
-------------
GRU significa Gated Recurrent Unit (Unidad Recurrente con Compuertas).
Es un tipo de red neuronal diseñada para aprender de secuencias de datos
en el tiempo, igual que el LSTM — pero con una arquitectura más simple.

Al igual que el LSTM, el GRU procesa las últimas N semanas EN ORDEN y aprende
qué patrones en esa secuencia predicen los casos futuros de dengue.

¿EN QUÉ SE DIFERENCIA DEL LSTM?
---------------------------------
La diferencia está en la cantidad de "compuertas" de memoria que usa
internamente para decidir qué recordar y qué olvidar:

  LSTM: tiene 3 compuertas → olvido, entrada, salida
  GRU:  tiene 2 compuertas → reset, update

  Menos compuertas = menos parámetros que aprender = más eficiente con
  datasets pequeños como el nuestro (600 secuencias de entrenamiento).

Analogía: el LSTM es como un sistema de archivo con tres cajones separados
(qué olvidar, qué guardar nuevo, qué usar ahora). El GRU tiene solo dos
cajones combinados — más simple pero igual de eficiente en la práctica.

¿POR QUÉ GRU PODRÍA SER MEJOR QUE LSTM EN ESTA TESIS?
-------------------------------------------------------
En los resultados del LSTM vimos que el modelo tiene sobreajuste — aprende
bien el dataset de entrenamiento pero no generaliza tan bien a la validación.
Esto es típico cuando hay pocos datos.

El GRU, al tener menos parámetros, tiene menos riesgo de sobreajuste y
generalmente converge más rápido. La literatura científica muestra que GRU
supera al LSTM en series temporales cortas con datos escasos.

Comparación de parámetros (con 32 unidades y 44 features):
  LSTM(32): ~10.000 parámetros por capa
  GRU(32):   ~7.500 parámetros por capa  (25% menos)

¿CÓMO ES LA COMPARACIÓN CON EL LSTM?
-------------------------------------
Para que la comparación sea completamente justa, este script usa:
  - Exactamente las mismas features (44 variables)
  - Exactamente la misma ventana temporal (12 semanas)
  - Exactamente los mismos hiperparámetros de entrenamiento
  - Exactamente la misma corrección de desbalanceo (Huber loss + sample weights)
  - Los mismos horizontes de predicción (actual, h=1, h=2, h=3, h=4)

La única diferencia es GRU() en lugar de LSTM(). Así cualquier diferencia
en los resultados se debe a la arquitectura, no a la configuración.

ARQUITECTURAS COMPARADAS:
  A) GRU simple (1 capa):
     12 semanas → [GRU 32] → [Dense 16] → [Dense 8] → predicción
     Menos parámetros, menor sobreajuste, más rápido de entrenar.

  B) GRU apilado (2 capas):
     12 semanas → [GRU 32] → [GRU 16] → [Dense 8] → predicción
     Mayor capacidad para capturar dinámicas complejas del brote.

PIPELINE:
  Paso 1 → Cargar datos y normalizar el target
  Paso 2 → Construir secuencias temporales (ventanas de 12 semanas)
  Paso 3 → Definir arquitecturas GRU simple y apilado
  Paso 4 → Entrenar con Huber loss + sample weights (corrige desbalanceo)
  Paso 5 → Comparar GRU vs LSTM vs modelos del Sprint 4
  Paso 6 → Guardar modelos, gráficos y métricas

PREREQUISITO:
  Ejecutar src/features/lags.py antes de este script.
  Requiere TensorFlow 2.x instalado en venv311.
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

# Silenciar warnings de TensorFlow que no son errores
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Semillas para reproducibilidad
# Sin esto, dos ejecuciones del mismo script pueden dar resultados diferentes
# porque las redes neuronales se inicializan con pesos aleatorios.
# Usamos las mismas semillas que en lstm_model.py para comparación justa.
tf.random.set_seed(42)
np.random.seed(42)


# =============================================================================
# CONFIGURACIÓN GENERAL
# Todos los parámetros en un solo lugar.
# IMPORTANTE: estos valores son idénticos a lstm_model.py para garantizar
# que la comparación GRU vs LSTM sea completamente justa.
# =============================================================================

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

# VENTANA TEMPORAL: cuántas semanas mira el GRU hacia atrás para predecir.
# 12 semanas (3 meses) — igual que el LSTM final para comparación justa.
# Con 12 semanas el modelo puede ver el patrón completo de un brote:
# inicio lento, aceleración, pico, descenso.
WINDOW_SIZE = 12


# =============================================================================
# FEATURES — exactamente las mismas que el LSTM
#
# Usamos las versiones normalizadas (_norm) porque las redes neuronales
# son sensibles a la escala de los datos. Sin normalización, una variable
# con valores de 0-1391 (casos) dominaría sobre una con valores de -1 a 1
# (semana_sin), sin que eso refleje su importancia real.
#
# Las columnas _norm fueron generadas en lags.py con StandardScaler
# ajustado solo con los datos de entrenamiento (sin ver el futuro).
# =============================================================================

FEATURES_GRU = (
    # Historial de casos de la propia comuna (normalizados)
    # Los lags capturan la memoria epidemiológica: si hubo brote la semana
    # pasada, es probable que haya brote esta semana también.
    ["cases_lag1_norm", "cases_lag2_norm", "cases_lag3_norm", "cases_lag4_norm",
     "incidencia_lag1_norm", "incidencia_lag2_norm",
     "incidencia_lag3_norm", "incidencia_lag4_norm"] +

    # Vecindad espacial: promedio de casos de comunas vecinas la semana pasada
    # Si la Comuna 5 tiene brote, es probable que la Comuna 6 (su vecina)
    # también lo tenga pronto — la enfermedad se dispersa geográficamente.
    ["casos_vecinas_lag1", "incidencia_vecinas_lag1"] +

    # Variables climáticas con lags (normalizadas)
    # El mosquito Aedes aegypti tarda semanas en reproducirse.
    # El calor de hace 4 semanas influye en los casos de hoy.
    [f"temp_mean_lag{i}_norm"             for i in range(1, 5)] +
    [f"precipitation_lag{i}_norm"         for i in range(1, 5)] +
    [f"humidity_mean_lag{i}_norm"         for i in range(1, 5)] +
    [f"heat_index_mean_lag{i}_norm"       for i in range(1, 5)] +
    [f"temp_mean_anomaly_lag{i}_norm"     for i in range(1, 5)] +
    [f"precipitation_anomaly_lag{i}_norm" for i in range(1, 5)] +
    [f"humidity_mean_anomaly_lag{i}_norm" for i in range(1, 5)] +

    # Estacionalidad: en qué momento del año estamos
    # El dengue tiene picos estacionales en enero-abril (verano austral).
    # semana_sin y semana_cos codifican la semana como un círculo continuo
    # para que el modelo entienda que diciembre y enero son consecutivos.
    ["semana_sin", "semana_cos", "is_epidemic_season", "mes_aprox"] +

    # Características de la comuna
    ["es_comuna_1", "poblacion"]
)

# Métricas de referencia de modelos anteriores para la tabla comparativa final
SPRINT_REF = {
    "Persistencia (lag 1)": {"MAE": 16.81, "R2": 0.932},
    "Random Forest":        {"MAE": 24.93, "R2": 0.614},
    "XGBoost semana actual":{"MAE": 24.92, "R2": 0.580},
}

# Resultados del LSTM (versión final) para comparación directa GRU vs LSTM
# Estos valores vienen del informe INFORME_SPRINT5_LSTM.md
LSTM_REF = {
    0: 51.99,   # semana actual
    1: 39.88,   # h=1
    2: 28.49,   # h=2
    3: 18.77,   # h=3
    4: 12.88,   # h=4
}

# Colores para los gráficos
COLOR_GRU1  = "#E74C3C"   # rojo    — GRU simple
COLOR_GRU2  = "#E67E22"   # naranja — GRU apilado
COLOR_LSTM  = "#8E44AD"   # violeta — LSTM (referencia)
COLOR_RF    = "#3498DB"   # azul    — Random Forest
COLOR_PERS  = "#F39C12"   # amarillo— Persistencia


# =============================================================================
# PASO 1: CARGA Y NORMALIZACIÓN
#
# Para redes neuronales normalizamos dos cosas por separado:
#
# A) FEATURES: ya normalizadas en lags.py con StandardScaler (sufijo _norm).
#    El scaler fue ajustado solo con train para evitar ver el futuro.
#
# B) TARGET (confirmed_cases): lo normalizamos aquí con MinMaxScaler.
#    MinMaxScaler lleva los valores al rango [0, 1]:
#      0 casos   → 0.0
#      649 casos (máximo train) → 1.0
#      1391 casos (máximo val)  → 2.14  (puede superar 1 porque val tiene más casos)
#
#    Al final desnormalizamos las predicciones para obtener casos reales.
#    REGLA CRÍTICA: el MinMaxScaler se ajusta SOLO con train.
# =============================================================================

def cargar_y_normalizar():
    """
    Carga los tres conjuntos de datos y normaliza el target confirmed_cases.

    Las features ya vienen normalizadas de lags.py.
    El target se normaliza aquí con MinMaxScaler ajustado solo con train.

    Retorna los datasets, las features disponibles, el normalizador del target
    y los valores del target normalizados para cada conjunto de datos.
    """
    logger.info("--- PASO 1: Cargando datos y normalizando el target ---")

    for archivo in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not archivo.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {archivo}\n"
                "Solución: ejecutá src/features/lags.py primero."
            )

    df_train = pd.read_parquet(TRAIN_FILE)
    df_val   = pd.read_parquet(VAL_FILE)
    df_test  = pd.read_parquet(TEST_FILE)

    logger.info("  Entrenamiento: %d filas | Validación: %d | Test: %d",
                len(df_train), len(df_val), len(df_test))

    # Verificar qué features están disponibles en el dataset
    features_disponibles = [f for f in FEATURES_GRU if f in df_train.columns]
    features_faltantes   = [f for f in FEATURES_GRU if f not in df_train.columns]
    if features_faltantes:
        logger.warning("  %d features no encontradas: %s",
                       len(features_faltantes), features_faltantes[:3])
    logger.info("  Features disponibles: %d de %d",
                len(features_disponibles), len(FEATURES_GRU))

    # Normalizar el target con MinMaxScaler
    # fit_transform: aprende los parámetros (min, max) Y transforma — solo en train
    # transform: aplica los mismos parámetros ya aprendidos — en val y test
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
# El GRU no recibe filas individuales como XGBoost — recibe VENTANAS:
# secuencias de WINDOW_SIZE semanas consecutivas en orden cronológico.
#
# Para cada semana t de cada comuna, construimos una ventana:
#   Entrada: datos de semanas [t-12, t-11, ..., t-1]  → forma (12, n_features)
#   Target:  casos de la semana t (o t+h para horizontes futuros)
#
# Ejemplo con WINDOW_SIZE=12, horizonte=0:
#   Semana 13 de la Comuna 1:
#     Entrada: semanas 1 a 12 de la Comuna 1
#     Target: casos de la semana 13
#
# Las ventanas se construyen DENTRO de cada comuna para no mezclar series.
# Las primeras 12 semanas de cada comuna no tienen suficiente historia
# y se descartan.
# =============================================================================

def construir_secuencias(df, features, y_norm, horizonte=0):
    """
    Convierte el dataset plano en ventanas temporales para el GRU.

    Para cada (comuna, semana_t) construye:
      - X: datos de las semanas [t-12, ..., t-1] → forma (12, n_features)
      - y: casos en t (horizonte=0) o en t+h (horizonte=h), normalizados

    Las ventanas se construyen por comuna para no mezclar series de
    comunas diferentes.

    Retorna:
      X: array 3D (muestras, WINDOW_SIZE, n_features)
      y: array 1D (muestras,) — valores normalizados
      idx: índices originales para trazabilidad
    """
    X_list, y_list, idx_list = [], [], []

    for comuna_id in sorted(df["comuna_id"].unique()):

        # Extraer solo esta comuna en orden cronológico
        mask  = df["comuna_id"] == comuna_id
        df_c  = df[mask].sort_values(["year", "epi_week"]).copy()
        idx_c = df_c.index.tolist()

        # Matriz de features (n_semanas × n_features) — NaN → 0
        X_c = df_c[features].fillna(0).values

        # Vector de targets normalizados de esta comuna
        y_c = y_norm[mask.values]
        n   = len(df_c)

        # Deslizar la ventana por todas las semanas de esta comuna
        for t in range(WINDOW_SIZE, n):
            t_target = t + horizonte
            if t_target >= n:
                continue  # no hay datos futuros disponibles

            ventana    = X_c[t - WINDOW_SIZE : t]  # (WINDOW_SIZE, n_features)
            target_val = y_c[t_target]

            if not np.isnan(target_val):
                X_list.append(ventana)
                y_list.append(target_val)
                idx_list.append(idx_c[t])

    if len(X_list) == 0:
        raise ValueError(
            f"No se generaron secuencias. WINDOW_SIZE={WINDOW_SIZE} "
            "puede ser mayor que las semanas disponibles por comuna."
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.float32)

    logger.info("  Secuencias construidas: %d ventanas | forma: %s",
                len(X), str(X.shape))
    return X, y, idx_list


# =============================================================================
# PASO 3: ARQUITECTURAS GRU
#
# La capa GRU funciona igual que la LSTM pero con menos compuertas internas:
#
# GRU(units):
#   'units' es la cantidad de "celdas de memoria" — neuronas que recuerdan
#   información de semanas anteriores. Con 32 unidades, el GRU produce un
#   vector de 32 números que resume la secuencia de 12 semanas.
#   Menos que LSTM (que usaba 32 también) pero con ~25% menos parámetros.
#
# return_sequences=True/False:
#   False (default): devuelve solo el resumen final (vector de 32 números)
#   True: devuelve la secuencia completa paso a paso
#   Necesario True cuando hay una segunda capa GRU que necesita ver la secuencia.
#
# Dropout(0.1):
#   Apaga aleatoriamente el 10% de las neuronas durante el entrenamiento.
#   Fuerza al modelo a no depender de neuronas específicas, mejorando
#   la generalización. Usamos 0.1 (10%) — más suave que antes (0.2)
#   porque con pocos datos el Dropout agresivo impide el aprendizaje.
#
# Dense(n, activation="relu"):
#   Capa completamente conectada que aprende combinaciones de los patrones
#   del GRU. relu = max(0, x) — introduce no-linealidad necesaria.
#
# Dense(1, activation="linear"):
#   Capa de salida. Un solo número = predicción de casos (normalizado).
#   "linear" = sin transformación, el valor puede ser cualquier número.
# =============================================================================

def construir_gru_simple(n_features, nombre="GRU simple"):
    """
    Arquitectura GRU de una sola capa — espejo exacto del LSTM simple.

    La única diferencia con lstm_model.py es GRU() en lugar de LSTM().
    Esto garantiza que cualquier diferencia en resultados se debe a la
    arquitectura recurrente, no a otros factores.

    Flujo de información:
      Entrada (12 semanas × 44 features)
        → GRU 32 celdas   (resume 12 semanas en 32 patrones temporales)
        → Dropout 10%     (regularización: apaga 10% de neuronas)
        → Dense 16 + relu (combina los 32 patrones en 16 representaciones)
        → Dropout 10%
        → Dense 8 + relu  (refina a 8 combinaciones clave — transición gradual)
        → Dropout 10%
        → Dense 1 + linear (predicción: 1 número = casos normalizados)

    ¿Por qué la transición gradual 32→16→8→1?
    Bajar directamente de 32 a 1 es muy abrupto — el modelo perdería
    información importante. La transición gradual permite aprender
    combinaciones progresivamente más abstractas antes de la predicción.

    Ventajas sobre LSTM simple:
      - ~25% menos parámetros → menor sobreajuste con 600 muestras
      - Converge más rápido → más eficiente en CPU
      - Igual de expresivo para series de longitud media (12 semanas)
    """
    modelo = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),

        # GRU con 32 celdas — reemplaza al LSTM(32) de lstm_model.py
        # return_sequences=False: solo necesitamos el resumen final
        GRU(32, return_sequences=False),
        Dropout(0.1),   # regularización suave

        # Transición gradual hacia la predicción final
        Dense(16, activation="relu"),
        Dropout(0.1),

        Dense(8, activation="relu"),
        Dropout(0.1),

        # Capa de salida: predicción de casos en escala normalizada
        # activation="linear" = sin transformación, permite cualquier valor
        Dense(1, activation="linear")

    ], name=nombre.replace(" ", "_"))

    return modelo


def construir_gru_apilado(n_features, nombre="GRU apilado"):
    """
    Arquitectura GRU con dos capas apiladas — espejo exacto del LSTM apilado.

    Flujo de información:
      Entrada (12 semanas × 44 features)
        → GRU 32 celdas   (primera capa: aprende patrones simples)
        → Dropout 10%
        → GRU 16 celdas   (segunda capa: aprende patrones de patrones)
        → Dropout 10%
        → Dense 8 + relu  (transición gradual 16→8→1)
        → Dropout 10%
        → Dense 1 + linear

    ¿Por qué dos capas GRU?
      Primera capa GRU (32 celdas): procesa la secuencia semana a semana y
        aprende patrones simples — "los casos subieron 3 semanas seguidas",
        "el heat index estuvo alto las últimas 2 semanas".
      Segunda capa GRU (16 celdas): recibe la secuencia de patrones de la
        primera capa y aprende relaciones entre ellos — "cuando el crecimiento
        fue exponencial y el calor fue alto, viene un pico".

    Con GRU apilado usamos 32+16=48 unidades totales vs 32+16=48 del LSTM
    apilado — misma cantidad de unidades pero con menos parámetros por capa.

    Riesgo: con 600 muestras puede sobreajustar — el Dropout mitiga esto.
    """
    modelo = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),

        # Primera capa GRU: 32 celdas, aprende patrones simples
        # return_sequences=True: pasa TODA la secuencia a la segunda capa
        # (la segunda necesita ver los patrones de cada semana, no solo el resumen)
        GRU(32, return_sequences=True),
        Dropout(0.1),

        # Segunda capa GRU: 16 celdas, aprende patrones de patrones
        # return_sequences=False: devuelve solo el resumen final
        GRU(16, return_sequences=False),
        Dropout(0.1),

        # Transición gradual: 16 → 8 → 1
        Dense(8, activation="relu"),
        Dropout(0.1),

        Dense(1, activation="linear")

    ], name=nombre.replace(" ", "_"))

    return modelo


# =============================================================================
# CÁLCULO DE MÉTRICAS DE ERROR
#
# Cuatro métricas complementarias, siempre en escala original de casos:
#
# MAE  (Mean Absolute Error): error promedio en casos absolutos.
#   "En promedio me equivoco X casos por semana."
#   Fácil de interpretar, no penaliza extra los errores grandes.
#
# RMSE (Root Mean Squared Error): similar al MAE pero penaliza más los
#   errores grandes. Útil para detectar semanas muy mal predichas.
#
# R²   (Coeficiente de determinación): fracción de la variación real que
#   explica el modelo. 1.0=perfecto, 0.0=no mejor que el promedio,
#   negativo=peor que el promedio.
#
# MAPE (Mean Absolute Percentage Error): error en porcentaje.
#   Solo se calcula donde hay casos > 0 para evitar división por cero.
# =============================================================================

def calcular_metricas(y_real, y_pred, modelo, split, horizonte):
    """Calcula MAE, RMSE, R² y MAPE en escala original de casos."""
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
# PASO 4A: PESOS POR MUESTRA (SAMPLE WEIGHTS)
#
# El dataset tiene un desbalanceo severo: el 60% de las semanas tienen 0 casos
# y solo el 2.1% tienen más de 200 casos (brote severo).
#
# Sin corrección, el GRU aprende que predecir cero siempre minimiza el error
# promedio — correcto para el 60% de los casos pero inútil para detectar brotes.
#
# SOLUCIÓN: darle más peso a las semanas con casos durante el entrenamiento.
# Las semanas de brote "valen más" en el cálculo del error.
#
# Fórmula: peso = log(1 + casos_reales) + 1.0
#   0 casos   → peso = 1.0  (peso mínimo)
#   10 casos  → peso ≈ 3.4
#   50 casos  → peso ≈ 4.9
#   500 casos → peso ≈ 7.2  (7x más peso que una semana sin casos)
#   1391 casos→ peso ≈ 8.2
#
# El logaritmo suaviza la diferencia — no queremos que los picos dominen
# completamente el entrenamiento, sino que tengan más presencia.
# =============================================================================

def calcular_sample_weights(y_train_norm, scaler):
    """
    Calcula pesos por muestra para corregir el desbalanceo del 60% de ceros.

    Desnormaliza el target para obtener los casos reales y calcula el peso
    proporcional al logaritmo de los casos + 1 (para evitar log(0)).
    """
    y_real = scaler.inverse_transform(
        y_train_norm.reshape(-1, 1)
    ).flatten().clip(min=0)

    pesos = np.log1p(y_real) + 1.0
    return pesos


# =============================================================================
# PASO 4B: ENTRENAMIENTO CON HUBER LOSS + SAMPLE WEIGHTS
#
# DOS AJUSTES PARA EL DESBALANCEO (idénticos al LSTM para comparación justa):
#
# 1. HUBER LOSS (reemplaza MSE):
#    MSE = (predicción - real)² → los errores grandes pesan enormemente.
#    Con 60% de ceros, el modelo aprende a predecir cerca de cero siempre.
#
#    Huber loss combina lo mejor de MSE y MAE:
#      Error < delta → usa MSE (preciso para errores pequeños)
#      Error > delta → usa MAE (robusto ante errores grandes)
#    delta=0.1 en escala normalizada ≈ 130 casos reales.
#
# 2. SAMPLE WEIGHTS: ya calculados en calcular_sample_weights().
#
# PARÁMETROS DE ENTRENAMIENTO:
#   learning_rate=0.0005: aprendizaje lento y estable
#   batch_size=16: más actualizaciones por época con dataset pequeño
#   patience=30: espera 30 épocas sin mejora antes de detener
#   epochs=300: máximo (el early stopping detiene antes)
# =============================================================================

def entrenar_modelo(modelo, X_train, y_train, X_val, y_val,
                    nombre_arquitectura, horizonte, scaler):
    """
    Compila y entrena un modelo GRU con Huber loss y sample weights.

    El proceso:
    1. Calcula los pesos por muestra (más peso a semanas de brote)
    2. Compila el modelo con Huber loss y optimizer Adam
    3. Entrena con early stopping (detiene si val_loss no mejora en 30 épocas)
    4. Reduce el learning rate si el entrenamiento se estanca (ReduceLROnPlateau)
    5. Restaura automáticamente el mejor modelo encontrado durante el entrenamiento

    Retorna el modelo entrenado y el historial de entrenamiento.
    """
    # Calcular pesos — semanas de brote pesan más
    sample_weights = calcular_sample_weights(y_train, scaler)

    logger.info("  Pesos — min: %.2f | max: %.2f | media: %.2f",
                sample_weights.min(), sample_weights.max(), sample_weights.mean())

    # Compilar: definir cómo aprende el modelo
    modelo.compile(
        optimizer=Adam(learning_rate=0.0005),   # Adam: adapta la velocidad de aprendizaje
        loss=tf.keras.losses.Huber(delta=0.1),  # Huber: más robusta que MSE
        metrics=["mae"]                          # también muestra MAE durante el entrenamiento
    )

    # Callbacks: acciones automáticas durante el entrenamiento
    callbacks = [
        EarlyStopping(
            monitor="val_loss",          # vigila el error en validación
            patience=30,                 # para si no mejora en 30 épocas consecutivas
            restore_best_weights=True,   # guarda el mejor modelo automáticamente
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,                  # reduce el learning rate a la mitad
            patience=10,                 # si no mejora en 10 épocas
            min_lr=1e-6,                 # learning rate mínimo permitido
            verbose=0
        )
    ]

    logger.info("  Entrenando %s h=%d | Train: %s | Val: %s",
                nombre_arquitectura, horizonte, X_train.shape, X_val.shape)

    historia = modelo.fit(
        X_train, y_train,
        sample_weight=sample_weights,    # más peso a semanas de brote
        validation_data=(X_val, y_val),
        epochs=300,                      # máximo (early stopping detiene antes)
        batch_size=16,                   # 16 muestras por actualización de pesos
        callbacks=callbacks,
        verbose=0                        # silencioso — logger maneja los mensajes
    )

    n_epocas       = len(historia.history["loss"])
    mejor_val_loss = min(historia.history["val_loss"])
    logger.info("  Completado — épocas: %d | mejor val_loss: %.5f",
                n_epocas, mejor_val_loss)

    return modelo, historia


# =============================================================================
# PASO 5: TABLAS DE COMPARACIÓN
# =============================================================================

def tabla_gru_vs_lstm_vs_sprint4(todas_metricas):
    """
    Tabla comparativa completa: GRU vs LSTM vs modelos anteriores.

    Esta es la tabla más importante del informe de GRU. Permite ver:
    1. Si GRU mejora al LSTM en los mismos horizontes
    2. Cómo ambas redes neuronales se comparan con RF, XGBoost y persistencia
    3. En qué horizontes las redes neuronales aportan valor real
    """
    print("\n" + "=" * 90)
    print("  GRU vs LSTM vs Sprint 4 — Validación (brote 2024, semanas 1-26)")
    print("=" * 90)
    print(f"  {'Modelo':<32} {'Horizonte':>20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("  " + "-" * 84)

    print("  --- Sprint 4 (referencia) ---")
    for nombre, m in SPRINT_REF.items():
        print(f"  {nombre:<32} {'semana actual':>20} "
              f"{m['MAE']:>8.2f} {'---':>8} {m['R2']:>8.3f}")

    print("  --- LSTM Sprint 5 (referencia para comparar) ---")
    lstm_labels = {0: "semana actual", 1: "1 sem. adelante",
                   2: "2 sem. adelante", 3: "3 sem. adelante", 4: "4 sem. adelante"}
    for h, mae in LSTM_REF.items():
        print(f"  {'LSTM simple':<32} {lstm_labels[h]:>20} "
              f"{mae:>8.2f} {'---':>8} {'---':>8}")

    print("  --- GRU Sprint 5 ---")
    for m in todas_metricas:
        if m["Split"] == "Validation":
            h_str = "semana actual" if m["Horizonte"] == 0 else f"{m['Horizonte']} sem. adelante"
            print(f"  {m['Modelo']:<32} {h_str:>20} "
                  f"{m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['R2']:>8.3f}")

    print("=" * 90)
    print("  MAE = error promedio en casos | R² = varianza explicada (1=perfecto)")
    print("=" * 90 + "\n")


def tabla_gru_vs_lstm_directo(todas_metricas):
    """
    Comparación directa GRU vs LSTM por horizonte.
    Muestra con * los horizontes donde GRU supera al LSTM.
    Esta tabla responde la pregunta principal: ¿vale la pena usar GRU?
    """
    print("\n" + "=" * 80)
    print("  GRU vs LSTM — Comparación directa MAE por horizonte (Validation)")
    print("  * indica que GRU supera al LSTM en ese horizonte")
    print("=" * 80)
    print(f"  {'Modelo':<22} {'actual':>10} {'h=1':>10} {'h=2':>10} {'h=3':>10} {'h=4':>10}")
    print("  " + "-" * 74)

    # Fila LSTM (referencia)
    fila = f"  {'LSTM simple':<22}"
    for h in [0, 1, 2, 3, 4]:
        fila += f" {LSTM_REF[h]:>10.2f}"
    print(fila + "  ← referencia")

    # Filas GRU
    for arquitectura in ["GRU simple", "GRU apilado"]:
        m_arq = {m["Horizonte"]: m for m in todas_metricas
                 if m["Modelo"] == arquitectura and m["Split"] == "Validation"}
        fila = f"  {arquitectura:<22}"
        for h in [0, 1, 2, 3, 4]:
            if h in m_arq:
                mae   = m_arq[h]["MAE"]
                mejor = mae < LSTM_REF.get(h, 999)
                fila += f" {mae:>9.2f}{'*' if mejor else ' '}"
            else:
                fila += f" {'n/a':>10}"
        print(fila)

    print("=" * 80 + "\n")


def tabla_degradacion_gru(todas_metricas):
    """
    Muestra cómo cambia el MAE al predecir más semanas adelante.
    Si la degradación del GRU es más suave que la del LSTM,
    confirma que GRU generaliza mejor a largo plazo.
    """
    print("\n" + "=" * 70)
    print("  GRU — Degradación de precisión por horizonte")
    print("  ¿Cuánto empeora el modelo al predecir más lejos?")
    print("=" * 70)

    for arquitectura in ["GRU simple", "GRU apilado"]:
        m_arq = {m["Horizonte"]: m for m in todas_metricas
                 if m["Modelo"] == arquitectura and m["Split"] == "Validation"}
        print(f"\n  {arquitectura}:")
        print(f"  {'Métrica':<10} {'actual':>10} {'h=1':>10} {'h=2':>10} {'h=3':>10} {'h=4':>10}")
        print("  " + "-" * 62)
        fila_mae = f"  {'MAE':<10}"
        fila_r2  = f"  {'R²':<10}"
        for h in [0, 1, 2, 3, 4]:
            if h in m_arq:
                fila_mae += f" {m_arq[h]['MAE']:>10.2f}"
                fila_r2  += f" {m_arq[h]['R2']:>10.3f}"
            else:
                fila_mae += f" {'n/a':>10}"
                fila_r2  += f" {'n/a':>10}"
        print(fila_mae)
        print(fila_r2)

    print("\n" + "=" * 70 + "\n")


# =============================================================================
# PASO 6: GRÁFICOS
# =============================================================================

def graficar_historia_entrenamiento(historias, nombre_arquitectura):
    """
    Grafica cómo evolucionó el error durante el entrenamiento.

    ¿Cómo interpretar?
      Curva roja (Train): error en datos de entrenamiento — siempre baja
      Curva azul (Val):   error en datos de validación — lo que importa

      ✓ Ambas bajas y cercanas → el modelo generaliza bien
      ✗ Train baja, Val se queda plana → sobreajuste o distribution shift
      ✗ Ambas altas → el modelo no aprende

    Si el GRU tiene curvas más equilibradas que el LSTM, confirma que
    sus menos parámetros reducen el sobreajuste.
    """
    horizontes_label = {0: "semana actual", 1: "h=1", 2: "h=2", 3: "h=3", 4: "h=4"}
    n    = len(historias)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (h, historia) in zip(axes, historias.items()):
        epocas = range(1, len(historia.history["loss"]) + 1)
        ax.plot(epocas, historia.history["loss"],     label="Train (entrenamiento)",
                color=COLOR_GRU1, linewidth=1.5)
        ax.plot(epocas, historia.history["val_loss"], label="Val (validación)",
                color=COLOR_LSTM, linewidth=1.5)
        ax.set_xlabel("Época (pasada por todos los datos)")
        ax.set_ylabel("Error (Huber loss)")
        ax.set_title(
            f"Horizonte: {horizontes_label.get(h, f'h={h}')}\n"
            f"Épocas entrenadas: {len(epocas)}",
            fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f"{nombre_arquitectura} — Curva de aprendizaje\nSprint 5 / HU6",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    nombre_archivo = nombre_arquitectura.lower().replace(" ", "_")
    plt.savefig(FIGURES_DIR / f"20_aprendizaje_{nombre_archivo}.png",
                dpi=150, bbox_inches="tight")
    plt.show()


def graficar_gru_vs_lstm(todas_metricas):
    """
    Comparación visual GRU vs LSTM por horizonte.

    Responde la pregunta: ¿GRU supera al LSTM con este dataset?
    Si las curvas de GRU están por debajo del LSTM → GRU es mejor.
    Si están por encima → LSTM es mejor (menos probable con datos escasos).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    estilos = [
        ("GRU simple",  COLOR_GRU1, "o-"),
        ("GRU apilado", COLOR_GRU2, "s-"),
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

        for h, mae, r2 in zip(hs, maes, r2s):
            ax1.annotate(f"{mae:.1f}", (h, mae), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9,
                         color=color, fontweight="bold")
            ax2.annotate(f"{r2:.3f}", (h, r2), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9,
                         color=color, fontweight="bold")

    # LSTM como línea de referencia
    hs_lstm   = sorted(k for k in LSTM_REF if k > 0)
    maes_lstm = [LSTM_REF[h] for h in hs_lstm]
    ax1.plot(hs_lstm, maes_lstm, "D--", color=COLOR_LSTM, linewidth=1.5,
             markersize=7, label="LSTM simple (ref)", zorder=2)

    # Líneas de Sprint 4
    for ax, key in [(ax1, "MAE"), (ax2, "R2")]:
        ax.axhline(SPRINT_REF["Persistencia (lag 1)"][key],
                   color=COLOR_PERS, linestyle=":", linewidth=1.5,
                   label="Persistencia Sprint 4")
        ax.axhline(SPRINT_REF["Random Forest"][key],
                   color=COLOR_RF, linestyle=":", linewidth=1.5,
                   label="Random Forest Sprint 4")
        if key == "R2":
            ax.axhline(0, color="gray", linestyle=":", linewidth=1,
                       label="R²=0 (sin mejora)")

    ax1.set_xlabel("Horizonte de predicción (semanas)")
    ax1.set_ylabel("Error promedio (MAE en casos)")
    ax1.set_title("Error por horizonte\n(menor es mejor)", fontweight="bold")
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xticklabels(["1 sem.", "2 sem.", "3 sem.", "4 sem."])
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Horizonte de predicción (semanas)")
    ax2.set_ylabel("R² (varianza explicada)")
    ax2.set_title("Varianza explicada por horizonte\n(mayor es mejor, máximo=1.0)",
                  fontweight="bold")
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xticklabels(["1 sem.", "2 sem.", "3 sem.", "4 sem."])
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.suptitle("GRU vs LSTM — Comparación por horizonte de predicción\nSprint 5 / HU6",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "21_gru_vs_lstm_horizontes.png",
                dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_gru():
    """
    Pipeline completo del GRU.

    Entrena 10 modelos en total:
      2 arquitecturas (simple, apilado) × 5 horizontes (actual + h=1,2,3,4)

    Cada modelo usa la misma configuración que el LSTM para comparación justa.
    La única diferencia es la capa recurrente: GRU() en lugar de LSTM().
    """
    print("\n" + "=" * 65)
    print("  SPRINT 5 / HU6 — GRU")
    print("  GRU simple (1 capa) vs GRU apilado (2 capas)")
    print("  Comparación directa con LSTM del sprint anterior")
    print(f"  Ventana temporal: {WINDOW_SIZE} semanas (igual que LSTM para comparación justa)")
    print("=" * 65 + "\n")

    # ── Paso 1: carga y normalización ────────────────────────────────
    (df_train, df_val, df_test,
     features, target_scaler,
     y_train_norm, y_val_norm, y_test_norm) = cargar_y_normalizar()

    n_features = len(features)
    logger.info("  Entrada: ventanas de %d semanas × %d features", WINDOW_SIZE, n_features)

    # Las dos arquitecturas a comparar
    arquitecturas = {
        "GRU simple":  construir_gru_simple,
        "GRU apilado": construir_gru_apilado,
    }

    todas_metricas    = []
    modelos_guardados = {}
    historias_por_arq = {nombre: {} for nombre in arquitecturas}

    # ── Pasos 2-4: entrenar por horizonte ────────────────────────────
    for horizonte in [0] + HORIZONTES:

        if horizonte == 0:
            target_col = TARGET_COL
            label_h    = "semana actual"
            y_tr       = y_train_norm
            y_vl       = y_val_norm
            y_te       = y_test_norm
            h_scaler   = target_scaler
        else:
            target_col = f"target_h{horizonte}"
            label_h    = f"{horizonte} semana(s) adelante"
            # Normalizar el target del horizonte con MinMaxScaler ajustado en train
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

        # Paso 2: construir secuencias temporales
        logger.info("  Construyendo ventanas de %d semanas...", WINDOW_SIZE)
        X_train_seq, y_train_seq, _ = construir_secuencias(df_train, features, y_tr)
        X_val_seq,   y_val_seq,   _ = construir_secuencias(df_val,   features, y_vl)
        X_test_seq,  y_test_seq,  _ = construir_secuencias(df_test,  features, y_te)

        # Pasos 3-4: entrenar cada arquitectura
        for nombre_arq, constructor in arquitecturas.items():

            logger.info("  ── %s ──", nombre_arq)

            # Modelo fresco para cada horizonte (sin heredar pesos anteriores)
            modelo = constructor(n_features)

            # Entrenar con Huber loss + sample weights
            modelo, historia = entrenar_modelo(
                modelo, X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                nombre_arq, horizonte, h_scaler
            )

            historias_por_arq[nombre_arq][horizonte] = historia

            # ── Predicciones y desnormalización ──────────────────────
            # El modelo predice en escala normalizada [0,1].
            # Desnormalizamos para obtener casos reales (0..1391).
            def desnorm(y_pred_norm, scaler):
                """Revierte la normalización MinMaxScaler → casos reales."""
                return scaler.inverse_transform(
                    y_pred_norm.reshape(-1, 1)
                ).flatten().clip(min=0)

            pred_val_real  = desnorm(modelo.predict(X_val_seq,  verbose=0).flatten(), h_scaler)
            pred_test_real = desnorm(modelo.predict(X_test_seq, verbose=0).flatten(), h_scaler)
            y_val_real     = desnorm(y_val_seq,  h_scaler)
            y_test_real    = desnorm(y_test_seq, h_scaler)

            # Calcular métricas en escala original de casos
            met_val  = calcular_metricas(y_val_real,  pred_val_real,  nombre_arq, "Validation", horizonte)
            met_test = calcular_metricas(y_test_real, pred_test_real, nombre_arq, "Test",       horizonte)
            todas_metricas.extend([met_val, met_test])

            logger.info("  VAL  — MAE: %.2f | RMSE: %.2f | R²: %.3f",
                        met_val["MAE"], met_val["RMSE"], met_val["R2"])
            logger.info("  TEST — MAE: %.2f | RMSE: %.2f | R²: %.3f",
                        met_test["MAE"], met_test["RMSE"], met_test["R2"])

            clave = f"{nombre_arq.lower().replace(' ', '_')}_h{horizonte}"
            modelos_guardados[clave] = modelo

    # ── Paso 5: tablas de comparación ────────────────────────────────
    tabla_gru_vs_lstm_vs_sprint4(todas_metricas)
    tabla_gru_vs_lstm_directo(todas_metricas)
    tabla_degradacion_gru(todas_metricas)

    # ── Paso 6: gráficos ─────────────────────────────────────────────
    for nombre_arq, historias_h in historias_por_arq.items():
        historias_sel = {h: historias_h[h] for h in [0, 1] if h in historias_h}
        if historias_sel:
            graficar_historia_entrenamiento(historias_sel, nombre_arq)

    graficar_gru_vs_lstm(todas_metricas)

    # ── Guardar modelos y resultados ──────────────────────────────────
    logger.info("--- Guardando modelos y resultados ---")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for clave, modelo in modelos_guardados.items():
        ruta = MODELS_DIR / f"gru_{clave}.keras"
        modelo.save(ruta)
        logger.info("  Guardado: %s", ruta.name)

    with open(MODELS_DIR / "gru_target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)
    logger.info("  Normalizador guardado: gru_target_scaler.pkl")

    pd.DataFrame(todas_metricas).to_csv(
        MODELS_DIR / "metricas_sprint5_gru.csv", index=False
    )
    logger.info("  Métricas guardadas: metricas_sprint5_gru.csv")

    print("\n✓ Sprint 5 / HU6 — GRU completado.")
    print("  GRU simple ✓ | GRU apilado ✓")
    print("  Horizontes: semana actual + h=1, h=2, h=3, h=4")
    print("  Próximo paso: comparación final de todos los modelos del sprint")

    return {"modelos": modelos_guardados, "metricas": todas_metricas}


# =============================================================================
# PUNTO DE ENTRADA
# Se ejecuta cuando corrés: python src/models/gru_model.py
# =============================================================================

if __name__ == "__main__":
    resultado = run_gru()
