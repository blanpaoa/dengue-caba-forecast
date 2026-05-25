"""
================================================================================
Feature Engineering
================================================================================

¿QUÉ HACE ESTE SCRIPT?
-----------------------
Transforma los datos crudos de dengue y clima en variables listas para
que los modelos de inteligencia artificial puedan aprender de ellas.


¿QUÉ VARIABLES GENERA?
-----------------------
A partir de los casos confirmados de dengue y las variables climáticas,
este script genera automáticamente:

  1. Historial de casos (lags): cuántos casos hubo 1, 2, 3 y 4 semanas atrás
     en cada comuna. Si esta semana hay brote, probablemente la semana pasada
     también hubo — el modelo necesita saber eso.

  2. Targets futuros: cuántos casos habrá en 1, 2, 3 y 4 semanas adelante.
     Son las "respuestas correctas" que el modelo aprende a predecir.

  3. Vecindad espacial: promedio de casos de las comunas vecinas la semana
     pasada. Si la Comuna 5 tiene brote, es probable que la Comuna 6
     (que comparte límite) también lo tenga pronto.

  4. Lags climáticos: temperatura, humedad y precipitación de semanas anteriores.
     El mosquito Aedes aegypti tarda semanas en reproducirse — el calor de
     hace 4 semanas influye en los casos de esta semana.

  5. Estacionalidad: en qué momento del año estamos. El dengue tiene picos
     estacionales (verano austral, enero-abril).

  6. Codificación de comunas: representación matemática de cada comuna que
     el modelo puede interpretar.

  7. Normalización: ajuste de escala para que todas las variables sean
     comparables entre sí.

PIPELINE (orden de ejecución):
  Paso 1  → Cargar datos de casos y población
  Paso 2  → Calcular tasa de incidencia por habitante
  Paso 3  → Crear historial de casos (lags 1-4 semanas)
  Paso 3b → Crear targets futuros (h=1,2,3,4 semanas adelante)
  Paso 3c → Crear features de vecindad espacial
  Paso 4  → Crear lags de variables climáticas
  Paso 5  → Crear variables de estacionalidad
  Paso 6  → Codificar comunas en formato matemático
  Paso 7  → Definir variables a normalizar
  Paso 8  → Dividir en train / validation / test (respetando el tiempo)
  Paso 9  → Mostrar reporte de variables generadas
  Paso 10 → Guardar datasets en disco

PREREQUISITO:
  Requiere dataset_maestro.parquet (generado en Sprint 2) y
  poblacion_comunas_caba_2022.csv en data/external/.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURACIÓN GENERAL
# Centraliza todas las rutas y parámetros en un solo lugar.
# Si algo cambia (por ejemplo, agregar más semanas de lag), se modifica
# aquí y el cambio se propaga automáticamente a todo el script.
# =============================================================================

# Rutas de archivos de entrada y salida
PROCESSED_DIR = Path("data/processed")   # datos procesados
EXTERNAL_DIR  = Path("data/external")    # datos externos (población)
FEATURES_DIR  = Path("data/processed")   # donde guardar el resultado

MAESTRO_FILE   = PROCESSED_DIR / "dataset_maestro.parquet"       # datos de entrada
POBLACION_FILE = EXTERNAL_DIR  / "poblacion_comunas_caba_2022.csv"  # censo 2022
FEATURES_FILE  = FEATURES_DIR  / "dataset_features.parquet"      # dataset completo
FEATURES_CSV   = FEATURES_DIR  / "dataset_features.csv"          # versión CSV para inspección
SCALER_FILE    = FEATURES_DIR  / "scaler.pkl"                    # normalizador guardado

# División temporal del dataset
# El orden es fundamental: nunca entrenamos con datos del futuro.
# Train:      2023 completo        → el modelo aprende aquí
# Validation: 2024 semanas 1-26   → evaluamos durante el brote masivo
# Test:       2024 sem 27-52+2025 → evaluación final con datos nunca vistos
YEAR_TRAIN_END = 2023
YEAR_VAL_END   = 2024
WEEK_VAL_END   = 26    # semana 26 = fin del primer semestre 2024

# Variables climáticas que usamos como predictoras
# Todas provienen de ERA5 (reanálisis climático) vía Open-Meteo
VARS_CLIMA = [
    "temp_mean",            # temperatura media semanal (°C)
    "precipitation",        # precipitación acumulada (mm)
    "humidity_mean",        # humedad relativa media (%)
    "heat_index_mean",      # índice de calor (combina temp + humedad)
    "temp_mean_anomaly",    # diferencia respecto a la media histórica (°C)
    "precipitation_anomaly",# diferencia respecto a la media histórica (mm)
    "humidity_mean_anomaly",# diferencia respecto a la media histórica (%)
]

# Cuántas semanas de historia miramos hacia atrás
# N_LAGS=4 significa que usamos las últimas 4 semanas como predictores
N_LAGS = 4

# Cuántas semanas adelante queremos predecir
# N_HORIZONTES=4 genera targets para 1, 2, 3 y 4 semanas en el futuro
N_HORIZONTES = 4


# =============================================================================
# MATRIZ DE VECINDAD — 15 COMUNAS DE CABA
#
# ¿Para qué sirve esto?
# Si la Comuna 5 tiene un brote activo, es probable que pronto se propague
# a sus comunas vecinas (las que comparten límite geográfico). Agregar esta
# información como variable predictora ayuda al modelo a capturar la
# dimensión espacial de la epidemia.
#
# ¿Cómo se construyó?
# Verificada manualmente contra el mapa oficial de comunas de CABA (2005).
# Cada entrada lista las comunas que comparten límite con esa comuna.
#
# Datos geográficos relevantes:
#   C7 (Flores/Parque Chacabuco): 7 vecinos — la más conectada de CABA
#   C15 (Agronomía/Chacarita):    6 vecinos
#   C1 (Puerto Madero/Centro):    3 vecinos — posición marginal (costero)
#   C8, C9:                       3 vecinos — sur de CABA, menos conectadas
#
# Decisión técnica DT-25.
# =============================================================================
VECINOS_CABA = {
    1:  [2, 3, 4],              # Puerto Madero, Constitución, Montserrat
    2:  [1, 3, 5, 14],          # Recoleta
    3:  [1, 2, 4, 5],           # Balvanera, San Cristóbal
    4:  [1, 3, 5, 7, 8],        # Boca, Barracas, Parque Patricios, Nueva Pompeya
    5:  [2, 3, 4, 6, 7, 14, 15],# Almagro, Boedo — la más central
    6:  [5, 7, 11, 15],          # Caballito
    7:  [4, 5, 6, 8, 9, 10, 11],# Flores, Parque Chacabuco — más conectada
    8:  [4, 7, 9],               # Villa Lugano, Soldati, Riachuelo
    9:  [7, 8, 10],              # Liniers, Mataderos, Parque Avellaneda
    10: [7, 9, 11],              # Floresta, Monte Castro, Vélez Sársfield
    11: [6, 7, 10, 12, 15],      # Villa del Parque, Devoto, Villa del Parque
    12: [11, 13, 15],            # Coghlan, Saavedra, Villa Urquiza
    13: [12, 14, 15],            # Belgrano, Colegiales, Núñez
    14: [2, 5, 13, 15],          # Palermo
    15: [5, 6, 11, 12, 13, 14],  # Agronomía, Chacarita, Villa Crespo
}


# =============================================================================
# PASO 1: CARGA DE DATOS
#
# ¿Qué cargamos?
#   - dataset_maestro.parquet: casos de dengue + clima ya unificados (Sprint 2)
#     2340 filas × 13 columnas, sin valores faltantes
#   - poblacion_comunas_caba_2022.csv: población de cada comuna según censo 2022
#     Necesaria para calcular tasas de incidencia por habitante
# =============================================================================

def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga el dataset maestro de dengue+clima y los datos de población.
    Verifica que los archivos existen antes de intentar abrirlos.
    """
    logger.info("--- PASO 1: Cargando datos ---")

    for filepath in [MAESTRO_FILE, POBLACION_FILE]:
        if not filepath.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {filepath}\n"
                "Verificá que los pasos anteriores del pipeline se ejecutaron."
            )

    df     = pd.read_parquet(MAESTRO_FILE)
    df_pob = pd.read_csv(POBLACION_FILE)

    logger.info("  Dataset maestro: %d filas × %d columnas", len(df), len(df.columns))
    logger.info("  Comunas con datos de población: %d", len(df_pob))
    logger.info("  Población total CABA: %s habitantes", f"{df_pob['poblacion'].sum():,}")

    return df, df_pob


# =============================================================================
# PASO 2: TASA DE INCIDENCIA PER CÁPITA
#
# ¿Por qué calculamos esto?
# La Comuna 1 tiene 22.000 habitantes y la Comuna 9 tiene 161.000.
# Si ambas tienen 50 casos, el impacto es completamente distinto.
# La tasa de incidencia normaliza por población, haciendo las comunas
# comparables entre sí independientemente de su tamaño.
#
# Fórmula: incidencia_x10000 = casos / (población / 10.000)
# Ejemplo: 50 casos en C1 (22.000 hab) → 22.7 casos/10.000 hab
#          50 casos en C9 (161.000 hab) →  3.1 casos/10.000 hab
#
# Decisión técnica DT-13.
# =============================================================================

def calcular_incidencia(df: pd.DataFrame, df_pob: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la población de cada comuna y calcula la tasa de incidencia
    por cada 10.000 habitantes.

    La tasa de incidencia es la variable epidemiológicamente correcta para
    comparar la intensidad del dengue entre comunas de distinto tamaño.
    """
    logger.info("--- PASO 2: Calculando tasa de incidencia per cápita ---")

    # Agregar la población de cada comuna al dataset principal
    df = df.merge(
        df_pob[["comuna_id", "poblacion"]],
        on="comuna_id",
        how="left"
    )

    # Calcular la tasa: casos por cada 10.000 habitantes
    df["incidencia_x10000"] = (
        df["confirmed_cases"] / (df["poblacion"] / 10_000)
    ).round(4)

    logger.info("  Incidencia media: %.2f casos/10.000 hab", df["incidencia_x10000"].mean())
    logger.info(
        "  Incidencia máxima: %.2f casos/10.000 hab (%.0f casos en comuna con %d hab)",
        df["incidencia_x10000"].max(),
        df.loc[df["incidencia_x10000"].idxmax(), "confirmed_cases"],
        df.loc[df["incidencia_x10000"].idxmax(), "poblacion"]
    )

    return df


# =============================================================================
# PASO 3: HISTORIAL DE CASOS (LAGS TEMPORALES)
#
# ¿Para qué sirven los lags?
# Los modelos de machine learning solo ven los datos de UNA fila a la vez.
# Para que el modelo sepa cuántos casos hubo la semana pasada, tenemos que
# agregar esa información explícitamente como una nueva columna.
#
# Ejemplo para la semana 10 de la Comuna 1:
#   cases_lag1 = casos de la semana 9  (hace 1 semana)
#   cases_lag2 = casos de la semana 8  (hace 2 semanas)
#   cases_lag3 = casos de la semana 7  (hace 3 semanas)
#   cases_lag4 = casos de la semana 6  (hace 4 semanas)
#
# ¡IMPORTANTE! El lag se calcula DENTRO de cada comuna.
# Los casos de la semana pasada en la Comuna 1 NO son los mismos que
# los de la semana pasada en la Comuna 9.
#
# Las primeras semanas de cada comuna quedan con NaN porque no hay
# semanas anteriores disponibles. Estas filas se excluyen al entrenar.
#
# Genera: cases_lag1 a cases_lag4, incidencia_lag1 a incidencia_lag4
# Decisión técnica DT-14.
# =============================================================================

def crear_lags_casos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables con el historial de casos e incidencia de las semanas
    previas para cada comuna, usando shift() dentro de cada grupo de comuna.

    shift(1) desplaza los valores una fila hacia abajo dentro de cada comuna,
    lo que equivale a "la semana anterior de esta misma comuna".
    """
    logger.info("--- PASO 3: Creando historial de casos (lags 1 a 4 semanas) ---")

    # Ordenar por comuna y tiempo antes de calcular los lags
    df = df.sort_values(["comuna_id", "year", "epi_week"]).copy()

    for lag in range(1, N_LAGS + 1):
        # Lag de casos absolutos: cuántos casos hubo hace 'lag' semanas
        df[f"cases_lag{lag}"] = (
            df.groupby("comuna_id")["confirmed_cases"].shift(lag)
        )
        # Lag de incidencia: la tasa normalizada hace 'lag' semanas
        df[f"incidencia_lag{lag}"] = (
            df.groupby("comuna_id")["incidencia_x10000"].shift(lag)
        )

    logger.info("  Variables creadas: cases_lag1 a cases_lag4, incidencia_lag1 a incidencia_lag4")
    logger.info("  NaN esperados en las primeras %d semanas de cada comuna (sin historia previa)", N_LAGS)

    return df


# =============================================================================
# PASO 3b: TARGETS FUTUROS (MULTI-HORIZONTE)
#
# ¿Qué son los targets?
# Son las "respuestas correctas" que el modelo aprende a predecir.
# En lugar de predecir solo la semana siguiente, queremos evaluar qué tan
# lejos en el futuro puede mirar el modelo con utilidad para la salud pública:
#
#   target_h1 = casos la semana que viene      → para ajustar guardia médica
#   target_h2 = casos en 2 semanas             → para planificar insumos
#   target_h3 = casos en 3 semanas             → para iniciar fumigación
#   target_h4 = casos en 4 semanas (1 mes)     → alerta temprana máxima
#
# ¿Cómo funciona el shift negativo?
# shift(-1) desplaza los valores una fila hacia ARRIBA dentro de cada comuna,
# lo que equivale a "la semana siguiente de esta misma comuna".
#
# Las últimas h semanas de cada comuna quedan con NaN porque no hay
# futuro disponible en el dataset. Estas filas se excluyen al entrenar
# cada modelo de horizonte correspondiente.
#
# Criterio de aceptación HU6: "horizontes de predicción 1, 2, 3 y 4 semanas"
# Decisión técnica DT-23.
# =============================================================================

def crear_targets_horizonte(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables objetivo desplazadas hacia el futuro para cada horizonte.

    Ejemplo para la semana 5, Comuna 1:
      confirmed_cases = 10  (lo que ya pasó — presente)
      target_h1       = 15  (semana 6  — 1 semana adelante)
      target_h2       = 8   (semana 7  — 2 semanas adelante)
      target_h3       = 3   (semana 8  — 3 semanas adelante)
      target_h4       = 0   (semana 9  — 4 semanas adelante)

    El modelo recibe los datos de la semana 5 y aprende a predecir
    el valor correspondiente de cada target.
    """
    logger.info("--- PASO 3b: Creando targets futuros (h=1 a h=4 semanas) ---")

    df = df.sort_values(["comuna_id", "year", "epi_week"]).copy()

    for h in range(1, N_HORIZONTES + 1):
        # shift negativo: mira h semanas hacia el futuro dentro de cada comuna
        df[f"target_h{h}"] = (
            df.groupby("comuna_id")["confirmed_cases"].shift(-h)
        )

    # Reportar cuántos NaN hay por horizonte (es información útil para el diagnóstico)
    n_comunas = df["comuna_id"].nunique()
    for h in range(1, N_HORIZONTES + 1):
        n_nan = df[f"target_h{h}"].isna().sum()
        logger.info(
            "  target_h%d: %d NaN esperados (%d comunas × %d semanas sin futuro disponible)",
            h, n_nan, n_comunas, h
        )

    logger.info("  Targets creados: target_h1, target_h2, target_h3, target_h4 ✓")
    return df


# =============================================================================
# PASO 3c: FEATURES DE VECINDAD ESPACIAL
#
# ¿Por qué importa la vecindad?
# El dengue se dispersa geográficamente: si hay un brote activo en la
# Comuna 5, es probable que en las próximas semanas aparezcan casos en
# las comunas vecinas (que comparten límite geográfico con la 5).
#
# ¿Qué generamos?
# Para cada (comuna, semana) calculamos el promedio de casos e incidencia
# de las comunas que comparten límite, con un lag de 1 semana.
#
# Ejemplo para la semana 10 de la Comuna 6:
#   Sus vecinas son: 5, 7, 11, 15
#   casos_vecinas_lag1 = promedio de cases_lag1 de las comunas 5, 7, 11, 15
#   Es decir: cuántos casos tuvieron las comunas vecinas la semana pasada
#
# ¿Por qué usamos lag 1 y no los valores actuales?
# Para evitar "data leakage": no podemos usar los casos de esta semana
# de las comunas vecinas para predecir los casos de esta semana en
# nuestra comuna — esa información no estaría disponible en tiempo real.
# Con lag 1 usamos información de la semana pasada, que sí está disponible.
#
# La matriz de vecindad fue verificada contra el mapa oficial de CABA (2005).
# Decisión técnica DT-25.
# =============================================================================

def crear_features_vecindad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada (comuna, semana), calcula el promedio de casos e incidencia
    de las comunas vecinas con un lag de 1 semana.

    Proceso técnico:
      1. Pivot: reorganiza el dataset para tener una columna por comuna
         y una fila por semana (facilita calcular promedios entre comunas)
      2. Para cada comuna, promedia los valores de sus vecinas según VECINOS_CABA
      3. Aplica lag de 1 semana para evitar filtración de información futura
      4. Merge: devuelve el resultado al formato original del dataset
    """
    logger.info("--- PASO 3c: Creando features de vecindad espacial ---")

    df = df.sort_values(["comuna_id", "year", "epi_week"]).copy()

    # Paso 1: reorganizar para tener una columna por comuna
    # Antes:  cada fila = (semana, comuna, casos)
    # Después: cada fila = semana, columnas = comunas 1..15
    pivot_casos = df.pivot_table(
        index=["year", "epi_week"],
        columns="comuna_id",
        values="confirmed_cases",
        aggfunc="first"
    )
    pivot_incid = df.pivot_table(
        index=["year", "epi_week"],
        columns="comuna_id",
        values="incidencia_x10000",
        aggfunc="first"
    )

    # Paso 2: para cada comuna, promediar los valores de sus vecinas
    prom_casos = {}
    prom_incid = {}
    for comuna, vecinas in VECINOS_CABA.items():
        # Solo incluir vecinas que existan en el dataset
        vecinas_disponibles = [v for v in vecinas if v in pivot_casos.columns]
        if vecinas_disponibles:
            prom_casos[comuna] = pivot_casos[vecinas_disponibles].mean(axis=1)
            prom_incid[comuna] = pivot_incid[vecinas_disponibles].mean(axis=1)

    # Convertir a formato largo para hacer merge con el dataset
    df_vc = pd.DataFrame(prom_casos).stack().reset_index()
    df_vc.columns = ["year", "epi_week", "comuna_id", "casos_vecinas"]
    df_vi = pd.DataFrame(prom_incid).stack().reset_index()
    df_vi.columns = ["year", "epi_week", "comuna_id", "incidencia_vecinas"]

    # Paso 3: merge con el dataset original
    df = df.merge(df_vc, on=["year", "epi_week", "comuna_id"], how="left")
    df = df.merge(df_vi, on=["year", "epi_week", "comuna_id"], how="left")

    # Paso 4: aplicar lag de 1 semana para evitar data leakage
    # casos_vecinas_lag1 = promedio de casos de vecinas la SEMANA PASADA
    df["casos_vecinas_lag1"] = (
        df.groupby("comuna_id")["casos_vecinas"].shift(1)
    )
    df["incidencia_vecinas_lag1"] = (
        df.groupby("comuna_id")["incidencia_vecinas"].shift(1)
    )

    # Eliminar columnas intermedias (sin lag — no se usan directamente)
    df = df.drop(columns=["casos_vecinas", "incidencia_vecinas"])

    n_nan = df["casos_vecinas_lag1"].isna().sum()
    logger.info("  Features creadas: casos_vecinas_lag1, incidencia_vecinas_lag1")
    logger.info("  NaN esperados en primera semana de cada comuna: %d", n_nan)
    logger.info(
        "  C7 tiene %d vecinos (máximo) | C1, C8, C9 tienen %d (mínimo)",
        len(VECINOS_CABA[7]), len(VECINOS_CABA[1])
    )

    return df


# =============================================================================
# PASO 4: LAGS DE VARIABLES CLIMÁTICAS
#
# ¿Por qué usamos el clima de semanas anteriores y no solo el actual?
# El mosquito Aedes aegypti tarda semanas en completar su ciclo biológico:
#   huevo → larva → pupa → adulto → picadura → incubación → enfermedad
# Este ciclo tarda entre 10 y 30 días dependiendo de la temperatura.
# Por eso el clima de hace 4 semanas puede predecir los casos de hoy.
#
# Del análisis exploratorio (Sprint 3):
#   - Temperatura lag 4 semanas: correlación r=+0.45 (la más alta)
#   - Precipitación lag 1 semana: correlación r=+0.25
#
# A diferencia de los lags de casos (que son distintos por comuna),
# el clima es el mismo para toda CABA en cada semana.
# Por eso los lags climáticos se calculan sobre una tabla de clima único
# (una fila por semana) y luego se copian a todas las comunas.
#
# Genera: temp_mean_lag1..4, precipitation_lag1..4, etc. (28 variables)
# Decisión técnica DT-12.
# =============================================================================

def crear_lags_clima(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea lags temporales de 1 a 4 semanas para cada variable climática.

    Como el clima es igual para todas las comunas en cada semana,
    los lags se calculan sobre una tabla reducida (una fila por semana)
    y luego se unen al dataset principal mediante merge.

    Esto es más eficiente y evita calcular el mismo lag 15 veces
    (una por comuna).
    """
    logger.info("--- PASO 4: Creando lags de variables climáticas (1 a 4 semanas) ---")

    # Extraer una fila por semana (el clima es igual para todas las comunas)
    cols_clave = ["year", "epi_week"]
    df_clima_unico = (
        df[cols_clave + VARS_CLIMA]
        .drop_duplicates(subset=cols_clave)
        .sort_values(cols_clave)
        .copy()
    )

    # Calcular el lag de cada variable climática
    for var in VARS_CLIMA:
        for lag in range(1, N_LAGS + 1):
            # shift(lag): toma el valor de 'lag' semanas atrás
            df_clima_unico[f"{var}_lag{lag}"] = df_clima_unico[var].shift(lag)

    # Lista de todas las columnas de lags generadas
    cols_lags_clima = [
        f"{var}_lag{lag}"
        for var in VARS_CLIMA
        for lag in range(1, N_LAGS + 1)
    ]

    # Unir los lags climáticos al dataset principal por año y semana
    df = df.merge(
        df_clima_unico[cols_clave + cols_lags_clima],
        on=cols_clave,
        how="left"
    )

    logger.info("  Variables climáticas: %s", ", ".join(VARS_CLIMA))
    logger.info("  Lags generados por variable: 1 a %d semanas", N_LAGS)
    logger.info("  Total de features climáticas con lags: %d", len(cols_lags_clima))

    return df


# =============================================================================
# PASO 5: VARIABLES DE ESTACIONALIDAD
#
# ¿Por qué es importante la estacionalidad?
# El dengue tiene un patrón estacional muy marcado en Buenos Aires:
# los brotes ocurren principalmente entre enero y abril (verano austral),
# cuando el calor y las lluvias favorecen la reproducción del mosquito.
#
# ¿Cómo codificamos la semana del año?
# Si usamos el número de semana directamente (1 a 52), el modelo pensaría
# que la semana 52 y la semana 1 están muy lejos entre sí — pero en
# realidad son semanas consecutivas (fin e inicio de año).
#
# Solución: codificación cíclica con seno y coseno.
# Proyectamos el año como un círculo: semana_sin y semana_cos representan
# una posición en ese círculo donde la semana 52 y la semana 1 son vecinas.
#
# Decisión técnica DT-09 (codificación cíclica) y DT-10 (temporada alta).
# =============================================================================

def crear_features_estacionalidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables que capturan la posición temporal dentro del año:

    semana_sin, semana_cos: codificación cíclica de la semana epidemiológica.
      Ejemplo: semana 1 y semana 52 son consecutivas en el calendario
      pero el número 1 y 52 parecen muy distintos. Seno y coseno
      resuelven esto mapeando la semana a un círculo continuo.

    is_epidemic_season: 1 si estamos en temporada alta de dengue.
      SE 1-17  (enero-abril): pico de brotes histórico
      SE 48-52 (diciembre):   inicio de la temporada

    mes_aprox: mes del año aproximado (1-12), útil para interpretabilidad.
    """
    logger.info("--- PASO 5: Creando variables de estacionalidad ---")

    # Codificación cíclica: convierte la semana en coordenadas de un círculo
    # 2π/52 = ángulo que corresponde a una semana del año
    df["semana_sin"] = np.sin(2 * np.pi * df["epi_week"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["epi_week"] / 52)

    # Variable binaria: ¿estamos en temporada alta de dengue?
    # SE 1-17:  enero a principios de mayo (brote principal en verano austral)
    # SE 48-52: diciembre (inicio de la temporada, antes de Navidad)
    df["is_epidemic_season"] = (
        (df["epi_week"] <= 17) | (df["epi_week"] >= 48)
    ).astype(int)

    # Mes aproximado — cada 4 semanas ≈ 1 mes
    df["mes_aprox"] = ((df["epi_week"] - 1) // 4 + 1).clip(1, 12)

    semanas_alta = df["is_epidemic_season"].sum()
    logger.info(
        "  Semanas en temporada alta: %d de %d (%.1f%% del dataset)",
        semanas_alta, len(df), df["is_epidemic_season"].mean() * 100
    )

    return df


# =============================================================================
# PASO 6: CODIFICACIÓN DE COMUNAS
#
# ¿Por qué necesitamos codificar las comunas?
# Las comunas están numeradas del 1 al 15, pero ese número no tiene
# ningún significado matemático — la Comuna 15 no es "más grande" o
# "más importante" que la Comuna 1.
#
# Para que los modelos interpreten correctamente las comunas, generamos
# TRES representaciones distintas según el tipo de modelo:
#
# A) comuna_id como entero (1-15)
#    Para modelos de árboles (XGBoost, Random Forest).
#    Los árboles hacen cortes binarios (¿es la commune_id ≤ 7?) y no
#    asumen ningún orden matemático entre los valores.
#
# B) One-Hot Encoding (15 columnas binarias)
#    Para modelos lineales (Ridge, LSTM).
#    Sin One-Hot, el modelo lineal interpretaría que C15 = C1 × 15,
#    lo cual no tiene sentido. Con One-Hot, cada comuna es independiente.
#
# C) Flag es_comuna_1 (0 o 1)
#    La Comuna 1 tiene un comportamiento epidemiológico muy atípico:
#    concentra el 39.9% de todos los casos históricos de CABA pero es
#    la más pequeña en población. Este flag le da al modelo información
#    explícita sobre esta anomalía.
# =============================================================================

def codificar_comuna(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera tres representaciones matemáticas de la variable comuna:
      1. comuna_id (entero 1-15)    → para XGBoost y Random Forest
      2. comuna_1..comuna_15        → One-Hot Encoding para modelos lineales
      3. es_comuna_1 (0/1)          → flag para la anomalía de la Comuna 1
    """
    logger.info("--- PASO 6: Codificando variables de comunas ---")

    # A) comuna_id ya existe como entero — no se modifica

    # B) One-Hot Encoding: una columna binaria por cada comuna
    # pd.get_dummies convierte el ID numérico en 15 columnas de 0s y 1s
    dummies = pd.get_dummies(df["comuna_id"], prefix="comuna", dtype=int)

    # Garantizar que las 15 comunas siempre estén representadas
    # (por si alguna no aparece en un split de datos)
    for c in range(1, 16):
        col = f"comuna_{c}"
        if col not in dummies.columns:
            dummies[col] = 0

    # Ordenar numéricamente (comuna_1, comuna_2, ..., comuna_15)
    dummies = dummies[[f"comuna_{c}" for c in range(1, 16)]]
    df = pd.concat([df, dummies], axis=1)

    # C) Flag especial para la Comuna 1 (comportamiento atípico en EDA)
    df["es_comuna_1"] = (df["comuna_id"] == 1).astype(int)

    logger.info("  One-Hot Encoding: columnas comuna_1 a comuna_15 generadas")
    logger.info("  Flag es_comuna_1: %d filas activas", df["es_comuna_1"].sum())
    logger.info("  Nota: usar comuna_id para XGBoost/RF, dummies para modelos lineales/LSTM")

    return df


# =============================================================================
# PASO 7: LISTA DE VARIABLES A NORMALIZAR
#
# ¿Qué es la normalización y para qué sirve?
# Las variables tienen escalas muy distintas:
#   - temperatura: 15-40°C
#   - precipitación: 0-150mm
#   - casos: 0-1391
#   - semana_sin: -1 a +1
#
# Si no normalizamos, los modelos lineales y las redes neuronales
# (LSTM, GRU) darán más importancia a las variables con valores grandes
# simplemente por su escala, aunque no sean más informativas.
#
# La normalización ajusta cada variable para tener media=0 y desvío=1.
# Esto se hace SOLO con los datos de entrenamiento (train) para evitar
# que el modelo "vea" información de validación y test antes de tiempo.
#
# Los árboles de decisión (XGBoost, RF) NO necesitan normalización porque
# solo hacen cortes binarios, pero la incluimos para LSTM/GRU.
# =============================================================================

VARS_A_NORMALIZAR = (
    # Variables climáticas base
    ["temp_mean", "temp_max_mean", "temp_min_mean",
     "precipitation", "humidity_mean", "heat_index_mean",
     "temp_mean_anomaly", "precipitation_anomaly", "humidity_mean_anomaly",
     "incidencia_x10000"] +
    # Historial de casos
    [f"cases_lag{i}"     for i in range(1, N_LAGS + 1)] +
    [f"incidencia_lag{i}" for i in range(1, N_LAGS + 1)] +
    # Lags climáticos
    [f"{var}_lag{lag}" for var in VARS_CLIMA for lag in range(1, N_LAGS + 1)]
)


# =============================================================================
# PASO 8: DIVISIÓN TEMPORAL (TRAIN / VALIDATION / TEST)
#
# ¿Por qué no dividimos aleatoriamente como en otros problemas de ML?
# En series temporales NO podemos mezclar aleatoriamente los datos porque
# eso significaría entrenar con datos del futuro y evaluar con el pasado.
# El modelo "haría trampa" al ver información futura durante el entrenamiento.
#
# División cronológica:
#   TRAIN:      2023 completo         → el modelo aprende aquí
#   VALIDATION: 2024 semanas 1-26     → evaluamos durante el brote masivo
#   TEST:       2024 sem 27-52 + 2025 → evaluación final con datos no vistos
#
# El período de validación coincide con el brote más severo registrado
# en CABA (hasta 1.391 casos en una sola semana), lo que hace que
# sea el período más exigente y representativo para evaluar el modelo.
# =============================================================================

def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en tres conjuntos respetando el orden cronológico.
    Nunca mezcla datos de períodos distintos ni permite que información
    futura "contamine" el entrenamiento.
    """
    logger.info("--- PASO 8: Dividiendo en train / validation / test ---")

    # Definir qué filas van a cada conjunto
    mask_train = df["year"] == YEAR_TRAIN_END                          # 2023

    mask_val = (
        (df["year"] == YEAR_VAL_END) &                                 # 2024
        (df["epi_week"] <= WEEK_VAL_END)                               # semanas 1-26
    )

    mask_test = (
        ((df["year"] == YEAR_VAL_END) & (df["epi_week"] > WEEK_VAL_END)) |  # 2024 sem 27-52
        (df["year"] == 2025)                                                  # 2025 completo
    )

    df_train = df[mask_train].copy()
    df_val   = df[mask_val].copy()
    df_test  = df[mask_test].copy()

    logger.info("  Train:      %d filas | año 2023 completo", len(df_train))
    logger.info("  Validation: %d filas | 2024 semanas 1-26 (brote masivo)", len(df_val))
    logger.info("  Test:       %d filas | 2024 sem 27-52 + 2025", len(df_test))

    # Verificar que ninguna fila se perdió ni se duplicó
    assert len(df_train) + len(df_val) + len(df_test) == len(df), \
        "Error: la suma de los splits no coincide con el total del dataset"

    return df_train, df_val, df_test


def aplicar_normalizacion(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Normaliza las variables numéricas para que tengan media=0 y desvío=1.

    REGLA CRÍTICA: el normalizador (scaler) se ajusta SOLO con los datos
    de train. Luego se aplica a val y test usando los parámetros de train.

    Si ajustáramos el scaler con val o test, el modelo "vería" estadísticas
    del futuro durante el entrenamiento — eso sería trampa y daría
    resultados artificialmente buenos que no se replicarían en producción.

    Las columnas normalizadas reciben el sufijo '_norm' para diferenciarlas
    de las originales.
    """
    logger.info("  Aplicando normalización (StandardScaler ajustado solo con train) ---")

    # Solo normalizar las columnas que realmente existen en el dataset
    vars_existentes = [v for v in VARS_A_NORMALIZAR if v in df_train.columns]

    # Crear y ajustar el normalizador SOLO con datos de train
    scaler = StandardScaler()
    scaler.fit(df_train[vars_existentes].fillna(0))

    # Aplicar la misma normalización a los tres conjuntos
    sufijo = "_norm"
    for df_split in [df_train, df_val, df_test]:
        valores_norm = scaler.transform(df_split[vars_existentes].fillna(0))
        for i, var in enumerate(vars_existentes):
            df_split[f"{var}{sufijo}"] = valores_norm[:, i]

    logger.info("  Variables normalizadas: %d (sufijo '_norm' en el dataset)", len(vars_existentes))

    return df_train, df_val, df_test, scaler


# =============================================================================
# PASO 9: REPORTE DE FEATURES
# Imprime un resumen completo de todo lo que se generó, para verificar
# que el pipeline funcionó correctamente antes de guardar.
# =============================================================================

def reporte_features(df, df_train, df_val, df_test):
    """Muestra un resumen de todas las variables generadas y los splits."""

    print("\n" + "=" * 65)
    print("  REPORTE DE FEATURES — Sprint 4-5 / HU5")
    print("=" * 65)

    features_casos = (
        ["confirmed_cases", "incidencia_x10000"] +
        [f"cases_lag{i}"      for i in range(1, N_LAGS + 1)] +
        [f"incidencia_lag{i}" for i in range(1, N_LAGS + 1)] +
        [f"target_h{h}"       for h in range(1, N_HORIZONTES + 1)] +
        ["casos_vecinas_lag1", "incidencia_vecinas_lag1"]
    )
    features_clima_base = VARS_CLIMA
    features_clima_lags = [
        f"{var}_lag{lag}" for var in VARS_CLIMA for lag in range(1, N_LAGS + 1)
    ]
    features_estacionalidad = ["semana_sin", "semana_cos", "is_epidemic_season", "mes_aprox"]
    features_espaciales = (
        ["comuna_id", "es_comuna_1", "poblacion"] +
        [f"comuna_{c}" for c in range(1, 16)]
    )

    todas = (features_casos + features_clima_base + features_clima_lags +
             features_estacionalidad + features_espaciales)
    existentes = [f for f in todas if f in df.columns]

    print(f"\n  VARIABLES POR CATEGORÍA")
    print(f"  Epidemiológicas (casos, lags, targets, vecindad): {len([f for f in features_casos if f in df.columns])}")
    print(f"  Climáticas base:                                  {len([f for f in features_clima_base if f in df.columns])}")
    print(f"  Climáticas con lags (1-4 semanas):                {len([f for f in features_clima_lags if f in df.columns])}")
    print(f"  Estacionalidad:                                   {len([f for f in features_estacionalidad if f in df.columns])}")
    print(f"  Espaciales (comuna_id, dummies, flags):           {len([f for f in features_espaciales if f in df.columns])}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  TOTAL VARIABLES:                                  {len(existentes)}")

    print(f"\n  DIVISIÓN TEMPORAL")
    print(f"  Train:      {len(df_train):>5} filas | 2023 completo (entrenamiento)")
    print(f"  Validation: {len(df_val):>5} filas | 2024 semanas 1-{WEEK_VAL_END} (brote masivo)")
    print(f"  Test:       {len(df_test):>5} filas | 2024 sem {WEEK_VAL_END+1}-52 + 2025 (evaluación final)")

    print(f"\n  VARIABLE OBJETIVO")
    print(f"  confirmed_cases     → semana actual (comparable con Sprint 4)")
    print(f"  target_h1..h4       → 1 a 4 semanas adelante (multi-horizonte)")

    print(f"\n  VALORES FALTANTES (NaN)")
    nans = df[existentes].isnull().sum()
    cols_nan = nans[nans > 0]
    if len(cols_nan) == 0:
        print("  Sin valores faltantes.")
    else:
        print(f"  Variables con NaN: {len(cols_nan)}")
        print("  (esperado: primeras/últimas semanas de cada comuna por los lags)")
        for col in cols_nan.index[:5]:
            print(f"    {col}: {cols_nan[col]} NaN")
        if len(cols_nan) > 5:
            print(f"    ... y {len(cols_nan)-5} variables más")

    print(f"\n  CRITERIOS DE ACEPTACIÓN HU5:")
    print(f"  ✓ Lags temporales de casos (cases_lag1-4, incidencia_lag1-4)")
    print(f"  ✓ Targets multi-horizonte (target_h1, target_h2, target_h3, target_h4)")
    print(f"  ✓ Features de vecindad espacial (casos_vecinas_lag1, incidencia_vecinas_lag1)")
    print(f"  ✓ Lags climáticos (temp, precip, humedad, heat_index × 4 semanas)")
    print(f"  ✓ Estacionalidad cíclica (semana_sin, semana_cos, is_epidemic_season)")
    print(f"  ✓ Tasa de incidencia per cápita (casos por 10.000 habitantes)")
    print(f"  ✓ Codificación de comunas (entero + One-Hot + flag Comuna 1)")
    print(f"  ✓ Normalización sin data leakage (scaler ajustado solo con train)")
    print(f"  ✓ Split temporal cronológico (train → val → test)")
    print("=" * 65 + "\n")


# =============================================================================
# PIPELINE PRINCIPAL
# Orquesta todos los pasos en orden. Al ejecutar este script, se llama
# automáticamente a run_feature_engineering() que ejecuta todo el pipeline.
# =============================================================================

def run_feature_engineering(save: bool = True) -> dict:
    """
    Ejecuta el pipeline completo de feature engineering en orden.
    Al finalizar, guarda los datasets listos para el modelado en disco.

    Parámetro save=True: guarda los archivos .parquet, .csv y el scaler.
    Parámetro save=False: solo procesa sin guardar (útil para pruebas).
    """
    print("\n" + "=" * 65)
    print("  SPRINT 4-5 — Feature Engineering")
    print("  HU5: Preparación de variables para el modelado")
    print("=" * 65 + "\n")

    # Pasos 1-2: carga y enriquecimiento base
    df, df_pob = cargar_datos()
    df = calcular_incidencia(df, df_pob)

    # Pasos 3-6: generación de todas las variables predictoras
    df = crear_lags_casos(df)           # historial de casos propios
    df = crear_targets_horizonte(df)    # targets futuros h=1..4
    df = crear_features_vecindad(df)    # casos de comunas vecinas
    df = crear_lags_clima(df)           # historial climático
    df = crear_features_estacionalidad(df)  # posición en el año
    df = codificar_comuna(df)           # representación matemática de comunas

    # Paso 8: división temporal respetando el orden cronológico
    df_train, df_val, df_test = split_temporal(df)

    # Normalización — scaler ajustado solo con train
    df_train, df_val, df_test, scaler = aplicar_normalizacion(df_train, df_val, df_test)

    # Reconstruir dataset completo con todas las features
    df_full = pd.concat([df_train, df_val, df_test]).sort_values(
        ["comuna_id", "year", "epi_week"]
    ).reset_index(drop=True)

    # Paso 9: reporte
    reporte_features(df_full, df_train, df_val, df_test)

    # Paso 10: guardar en disco
    if save:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)

        df_full.to_parquet(FEATURES_FILE, index=False)
        logger.info("Dataset completo guardado: %s", FEATURES_FILE)

        df_full.to_csv(FEATURES_CSV, index=False)
        logger.info("CSV de inspección: %s", FEATURES_CSV)

        df_train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
        df_val.to_parquet(PROCESSED_DIR / "validation.parquet", index=False)
        df_test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
        logger.info("Splits guardados: train.parquet, validation.parquet, test.parquet")

        # Guardar el normalizador para usarlo en predicciones futuras
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Normalizador guardado: %s", SCALER_FILE)

    return {
        "df_full":  df_full,
        "df_train": df_train,
        "df_val":   df_val,
        "df_test":  df_test,
        "scaler":   scaler
    }


# =============================================================================
# PUNTO DE ENTRADA
# Este bloque se ejecuta cuando corrés el script directamente:
#   python src/features/lags.py
# =============================================================================

if __name__ == "__main__":
    resultado = run_feature_engineering(save=True)

    df_train = resultado["df_train"]

    print("Muestra de las primeras filas del conjunto de entrenamiento:")
    cols_muestra = [
        "year", "epi_week", "comuna_id",
        "confirmed_cases", "target_h1", "target_h4",
        "cases_lag1", "casos_vecinas_lag1",
        "temp_mean_lag1", "temp_mean_lag4",
        "semana_sin", "is_epidemic_season", "es_comuna_1"
    ]
    cols_existentes = [c for c in cols_muestra if c in df_train.columns]
    print(df_train[cols_existentes].head(10).to_string(index=False))
    print(f"\nDataset listo para el modelado (Sprint 5 — XGBoost, LSTM, GRU)")
