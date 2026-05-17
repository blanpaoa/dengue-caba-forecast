"""
Feature Engineering — Sprint 4 / HU5

Genera todas las variables predictoras a partir del dataset maestro.
El output es dataset_features.parquet listo para el modelado.

Pipeline:
    Paso 1 → Cargar dataset maestro y población por comuna
    Paso 2 → Calcular tasa de incidencia per cápita
    Paso 3 → Lags temporales de casos por comuna (t-1 a t-4)
    Paso 4 → Lags temporales de variables climáticas (1 a 4 semanas)
    Paso 5 → Variables de estacionalidad (seno, coseno, is_epidemic_season)
    Paso 6 → Codificación de comuna_id (entero para árboles + One-Hot para modelos lineales)
    Paso 7 → Normalización de variables numéricas
    Paso 8 → Split temporal train / validation / test
    Paso 9 → Reporte de features generadas
    Paso 10 → Guardar datasets
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
# CONSTANTES
# =============================================================================

PROCESSED_DIR = Path("data/processed")
EXTERNAL_DIR  = Path("data/external")
FEATURES_DIR  = Path("data/processed")

MAESTRO_FILE     = PROCESSED_DIR / "dataset_maestro.parquet"
POBLACION_FILE   = EXTERNAL_DIR  / "poblacion_comunas_caba_2022.csv"
FEATURES_FILE    = FEATURES_DIR  / "dataset_features.parquet"
FEATURES_CSV     = FEATURES_DIR  / "dataset_features.csv"
SCALER_FILE      = FEATURES_DIR  / "scaler.pkl"

# Años de split temporal
# Train: 2023
# Validation: 2024 semanas 1-26 (primer semestre)
# Test: 2024 semanas 27-52 + 2025
YEAR_TRAIN_END      = 2023
YEAR_VAL_END        = 2024
WEEK_VAL_END        = 26

# Variables climáticas base para crear lags
VARS_CLIMA = [
    "temp_mean",
    "precipitation",
    "humidity_mean",
    "heat_index_mean",
    "temp_mean_anomaly",
    "precipitation_anomaly",
    "humidity_mean_anomaly",
]

# Número de semanas de rezago
N_LAGS = 4


# =============================================================================
# PASO 1: CARGA DE DATOS
# =============================================================================

def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga el dataset maestro y los datos de población por comuna."""
    logger.info("--- PASO 1: Cargando datos ---")

    for filepath in [MAESTRO_FILE, POBLACION_FILE]:
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    df = pd.read_parquet(MAESTRO_FILE)
    df_pob = pd.read_csv(POBLACION_FILE)

    logger.info("  Dataset maestro: %d filas × %d columnas", len(df), len(df.columns))
    logger.info("  Población comunas: %d comunas", len(df_pob))
    logger.info("  Población total CABA: %s habitantes",
                f"{df_pob['poblacion'].sum():,}")

    return df, df_pob


# =============================================================================
# PASO 2: TASA DE INCIDENCIA PER CÁPITA
# Calculamos casos por cada 10,000 habitantes por comuna.
# Permite comparar comunas de diferente tamaño de forma justa.
# Decisión técnica DT-13.
# =============================================================================

def calcular_incidencia(
    df: pd.DataFrame,
    df_pob: pd.DataFrame
) -> pd.DataFrame:
    """
    Agrega la población de cada comuna y calcula la tasa de incidencia
    por cada 10,000 habitantes.

    incidencia_x10000 = confirmed_cases / (poblacion / 10000)
    """
    logger.info("--- PASO 2: Calculando tasa de incidencia per cápita ---")

    df = df.merge(
        df_pob[["comuna_id", "poblacion"]],
        on="comuna_id",
        how="left"
    )

    df["incidencia_x10000"] = (
        df["confirmed_cases"] / (df["poblacion"] / 10_000)
    ).round(4)

    logger.info(
        "  Incidencia media: %.2f casos/10,000 hab",
        df["incidencia_x10000"].mean()
    )
    logger.info(
        "  Incidencia máxima: %.2f casos/10,000 hab (%.0f casos, %d hab)",
        df["incidencia_x10000"].max(),
        df.loc[df["incidencia_x10000"].idxmax(), "confirmed_cases"],
        df.loc[df["incidencia_x10000"].idxmax(), "poblacion"]
    )

    return df


# =============================================================================
# PASO 3: LAGS TEMPORALES DE CASOS POR COMUNA
# Para cada comuna creamos variables con los casos de las semanas previas.
# El lag se calcula DENTRO de cada comuna — los casos de la semana pasada
# en la Comuna 1 predicen los casos de esta semana en la Comuna 1,
# no en la Comuna 9.
# Decisión técnica DT-14.
# =============================================================================

def crear_lags_casos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea lags temporales de confirmed_cases e incidencia_x10000
    respetando la dimensión espacial (por comuna).

    Genera: cases_lag1 a cases_lag4
            incidencia_lag1 a incidencia_lag4
    """
    logger.info("--- PASO 3: Creando lags de casos por comuna ---")

    df = df.sort_values(["comuna_id", "year", "epi_week"]).copy()

    for lag in range(1, N_LAGS + 1):
        # Lag de casos absolutos
        df[f"cases_lag{lag}"] = (
            df.groupby("comuna_id")["confirmed_cases"]
            .shift(lag)
        )
        # Lag de incidencia per cápita
        df[f"incidencia_lag{lag}"] = (
            df.groupby("comuna_id")["incidencia_x10000"]
            .shift(lag)
        )

    n_lags_creados = N_LAGS * 2
    logger.info("  Features de lags de casos creadas: %d", n_lags_creados)
    logger.info("  Ejemplos: cases_lag1, cases_lag2, incidencia_lag1, incidencia_lag2")

    return df


# =============================================================================
# PASO 4: LAGS TEMPORALES DE VARIABLES CLIMÁTICAS
# El clima de hace N semanas predice los casos actuales.
# Del EDA: temperatura óptima lag 4w (r=+0.45), precipitación lag 1w (r=+0.25).
# Decisión técnica DT-12.
# =============================================================================

def crear_lags_clima(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea lags temporales de 1 a 4 semanas para cada variable climática.
    Como el clima es el mismo para todas las comunas en cada semana,
    los lags se calculan sobre el dataset ordenado por semana (no por comuna).

    Genera: temp_mean_lag1 a temp_mean_lag4, etc.
    """
    logger.info("--- PASO 4: Creando lags de variables climáticas ---")

    # Primero obtenemos los valores climáticos únicos por semana
    # (son iguales para todas las comunas)
    cols_clave = ["year", "epi_week"]
    df_clima_unico = (
        df[cols_clave + VARS_CLIMA]
        .drop_duplicates(subset=cols_clave)
        .sort_values(cols_clave)
        .copy()
    )

    # Creamos los lags sobre el dataset de clima único
    for var in VARS_CLIMA:
        for lag in range(1, N_LAGS + 1):
            df_clima_unico[f"{var}_lag{lag}"] = df_clima_unico[var].shift(lag)

    # Columnas de lags generadas
    cols_lags_clima = [
        f"{var}_lag{lag}"
        for var in VARS_CLIMA
        for lag in range(1, N_LAGS + 1)
    ]

    # Unimos los lags climáticos al dataset principal
    df = df.merge(
        df_clima_unico[cols_clave + cols_lags_clima],
        on=cols_clave,
        how="left"
    )

    logger.info("  Features de lags climáticos creadas: %d", len(cols_lags_clima))
    logger.info("  Variables: %s", ", ".join(VARS_CLIMA))
    logger.info("  Lags: 1 a %d semanas", N_LAGS)

    return df


# =============================================================================
# PASO 5: VARIABLES DE ESTACIONALIDAD
# La semana del año es el predictor más importante según el EDA.
# Codificación cíclica para que SE52 y SE1 sean semanas contiguas.
# Decisión técnica DT-09 y DT-10.
# =============================================================================

def crear_features_estacionalidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea tres variables de estacionalidad:

    1. semana_sin: seno de la semana epidemiológica (componente cíclica)
    2. semana_cos: coseno de la semana epidemiológica (componente cíclica)
    3. is_epidemic_season: 1 si estamos en temporada alta (SE 1-17 o SE 48-52)

    La codificación seno + coseno permite que el modelo entienda que
    la SE52 y la SE1 son semanas consecutivas (inicio del verano austral).
    """
    logger.info("--- PASO 5: Creando features de estacionalidad ---")

    # Codificación cíclica
    df["semana_sin"] = np.sin(2 * np.pi * df["epi_week"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["epi_week"] / 52)

    # Variable binaria de temporada alta
    # SE 1-17: enero a abril (brote principal)
    # SE 48-52: diciembre (inicio de temporada)
    df["is_epidemic_season"] = (
        (df["epi_week"] <= 17) | (df["epi_week"] >= 48)
    ).astype(int)

    # Mes aproximado (útil para interpretabilidad)
    df["mes_aprox"] = ((df["epi_week"] - 1) // 4 + 1).clip(1, 12)

    logger.info(
        "  Semanas en temporada alta: %d (%.1f%%)",
        df["is_epidemic_season"].sum(),
        df["is_epidemic_season"].mean() * 100
    )

    return df


# =============================================================================
# PASO 6: CODIFICACIÓN DE COMUNA_ID
# Generamos DOS representaciones de la comuna para máxima flexibilidad:
#
# A) comuna_id como entero (1-15) → para modelos de árboles (XGBoost, RF)
#    Los árboles hacen cortes binarios y no asumen orden entre valores.
#    No hay problema en usar el entero directamente.
#
# B) One-Hot Encoding (comuna_1 ... comuna_15) → para modelos lineales
#    Los modelos lineales SÍ interpretan los números como magnitudes con
#    orden implícito. Sin One-Hot el modelo asumiría que Comuna 15 > Comuna 1,
#    lo cual no tiene ningún sentido geográfico ni epidemiológico.
#    Con dummies cada comuna es una variable binaria independiente.
#
# C) es_comuna_1 → flag binario para la Comuna 1 dado su comportamiento
#    atípico documentado en el EDA comparativo (39.9% de todos los casos,
#    57.1% de semanas con cero vs 69.8% del resto).
# =============================================================================

def codificar_comuna(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera tres representaciones de la variable comuna:

    1. comuna_id (entero 1-15)     → para XGBoost y Random Forest
    2. comuna_1 ... comuna_15      → One-Hot Encoding para modelos lineales
    3. es_comuna_1 (binaria 0/1)   → flag por comportamiento atípico

    El One-Hot Encoding evita que los modelos lineales interpreten
    el ID numérico como una jerarquía de importancia entre comunas.
    """
    logger.info("--- PASO 6: Codificando variables de comuna ---")

    # A) comuna_id como entero ya está en el dataset — no se modifica

    # B) One-Hot Encoding — genera una columna binaria por cada comuna
    # drop_first=False: conservamos las 15 dummies para máxima información
    # dtype=int: 0/1 en lugar de True/False
    dummies = pd.get_dummies(
        df["comuna_id"],
        prefix="comuna",
        dtype=int
    )
    # Aseguramos que las 15 comunas estén representadas
    # (por si alguna no aparece en un split)
    for c in range(1, 16):
        col = f"comuna_{c}"
        if col not in dummies.columns:
            dummies[col] = 0

    # Ordenamos las columnas numéricamente
    dummies = dummies[[f"comuna_{c}" for c in range(1, 16)]]
    df = pd.concat([df, dummies], axis=1)

    # C) Flag específico para la Comuna 1
    df["es_comuna_1"] = (df["comuna_id"] == 1).astype(int)

    logger.info("  One-Hot Encoding: columnas comuna_1 a comuna_15 generadas")
    logger.info(
        "  Filas Comuna 1: %d (%.1f%%) — flag es_comuna_1 activo",
        df["es_comuna_1"].sum(),
        df["es_comuna_1"].mean() * 100
    )
    logger.info(
        "  Nota: usar comuna_id (entero) para XGBoost/RF, "
        "dummies comuna_1..15 para modelos lineales"
    )

    return df


# =============================================================================
# PASO 7: NORMALIZACIÓN
# Estandarizamos las variables numéricas continuas (media=0, std=1).
# El scaler se ajusta SOLO con train para evitar data leakage.
# En este paso solo definimos qué variables normalizar 
# el scaler se aplica después del split.
# =============================================================================

VARS_A_NORMALIZAR = (
    ["temp_mean", "temp_max_mean", "temp_min_mean",
     "precipitation", "humidity_mean", "heat_index_mean",
     "temp_mean_anomaly", "precipitation_anomaly", "humidity_mean_anomaly",
     "incidencia_x10000"] +
    [f"cases_lag{i}" for i in range(1, N_LAGS + 1)] +
    [f"incidencia_lag{i}" for i in range(1, N_LAGS + 1)] +
    [f"{var}_lag{lag}" for var in VARS_CLIMA for lag in range(1, N_LAGS + 1)]
)


# =============================================================================
# PASO 8: SPLIT TEMPORAL TRAIN / VALIDATION / TEST
# Respeta el orden cronológico — nunca entrenamos con datos del futuro.
#
# Train:      2023 completo
# Validation: 2024 SE 1-26 (primer semestre — incluye el brote)
# Test:       2024 SE 27-52 + 2025 (segundo semestre y año siguiente)
#
# Esta división garantiza que el modelo se evalúe con datos
# genuinamente futuros respecto al período de entrenamiento.
# =============================================================================

def split_temporal(df: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Divide el dataset en train, validation y test respetando el orden temporal.

    Returns:
        df_train, df_val, df_test
    """
    logger.info("--- PASO 8: Split temporal train / validation / test ---")

    mask_train = df["year"] == 2023

    mask_val = (
        (df["year"] == 2024) &
        (df["epi_week"] <= WEEK_VAL_END)
    )

    mask_test = (
        ((df["year"] == 2024) & (df["epi_week"] > WEEK_VAL_END)) |
        (df["year"] == 2025)
    )

    df_train = df[mask_train].copy()
    df_val   = df[mask_val].copy()
    df_test  = df[mask_test].copy()

    logger.info(
        "  Train:      %d filas | %d comunas | año 2023",
        len(df_train), df_train["comuna_id"].nunique()
    )
    logger.info(
        "  Validation: %d filas | %d comunas | 2024 SE 1-26",
        len(df_val), df_val["comuna_id"].nunique()
    )
    logger.info(
        "  Test:       %d filas | %d comunas | 2024 SE 27-52 + 2025",
        len(df_test), df_test["comuna_id"].nunique()
    )

    # Verificar que no haya solapamiento
    assert len(df_train) + len(df_val) + len(df_test) == len(df), \
        "Error: las particiones no cubren todo el dataset"

    return df_train, df_val, df_test


def aplicar_normalizacion(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Ajusta el StandardScaler SOLO con train y lo aplica a los tres splits.
    Esto evita data leakage — el modelo no ve las estadísticas de val y test
    durante el entrenamiento.
    """
    logger.info("  Aplicando normalización (fit solo en train)...")

    # Solo normalizamos las columnas que existen en el dataset
    vars_existentes = [
        v for v in VARS_A_NORMALIZAR
        if v in df_train.columns
    ]

    scaler = StandardScaler()

    # Fit SOLO con train
    scaler.fit(df_train[vars_existentes].fillna(0))

    # Transform en los tres splits
    sufijo = "_norm"
    for df_split, nombre in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        valores_norm = scaler.transform(df_split[vars_existentes].fillna(0))
        for i, var in enumerate(vars_existentes):
            df_split[f"{var}{sufijo}"] = valores_norm[:, i]

    logger.info(
        "  Variables normalizadas: %d (sufijo '_norm')",
        len(vars_existentes)
    )

    return df_train, df_val, df_test, scaler


# =============================================================================
# PASO 9: REPORTE DE FEATURES
# =============================================================================

def reporte_features(
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame
):
    """Imprime un resumen de todas las features generadas."""

    print("\n" + "=" * 65)
    print("  REPORTE DE FEATURES — Sprint 4 / HU5")
    print("=" * 65)

    # Categorías de features
    features_casos = (
        ["confirmed_cases", "incidencia_x10000"] +
        [f"cases_lag{i}" for i in range(1, N_LAGS + 1)] +
        [f"incidencia_lag{i}" for i in range(1, N_LAGS + 1)]
    )

    features_clima_base = VARS_CLIMA

    features_clima_lags = [
        f"{var}_lag{lag}"
        for var in VARS_CLIMA
        for lag in range(1, N_LAGS + 1)
    ]

    features_estacionalidad = [
        "semana_sin", "semana_cos",
        "is_epidemic_season", "mes_aprox"
    ]

    features_espaciales = (
        ["comuna_id", "es_comuna_1", "poblacion"] +
        [f"comuna_{c}" for c in range(1, 16)]
    )

    todas_las_features = (
        features_casos +
        features_clima_base +
        features_clima_lags +
        features_estacionalidad +
        features_espaciales
    )

    existentes = [f for f in todas_las_features if f in df.columns]

    print(f"\n  FEATURES POR CATEGORÍA")
    print(f"  Epidemiológicas (casos + lags):  {len([f for f in features_casos if f in df.columns])}")
    print(f"  Climáticas base:                 {len([f for f in features_clima_base if f in df.columns])}")
    print(f"  Climáticas con lags:             {len([f for f in features_clima_lags if f in df.columns])}")
    print(f"  Estacionalidad:                  {len([f for f in features_estacionalidad if f in df.columns])}")
    print(f"  Espaciales:                      {len([f for f in features_espaciales if f in df.columns])}")
    print(f"  ─────────────────────────────────────────")
    print(f"  TOTAL FEATURES:                  {len(existentes)}")

    print(f"\n  SPLIT TEMPORAL")
    print(f"  Train:      {len(df_train):>5} filas | año 2023 completo")
    print(f"  Validation: {len(df_val):>5} filas | 2024 SE 1-{WEEK_VAL_END}")
    print(f"  Test:       {len(df_test):>5} filas | 2024 SE {WEEK_VAL_END+1}-52 + 2025")

    print(f"\n  VARIABLE OBJETIVO")
    print(f"  confirmed_cases (regresión)")
    print(f"  incidencia_x10000 (regresión normalizada por población)")

    print(f"\n  VALORES FALTANTES TRAS FEATURE ENGINEERING")
    nans = df[existentes].isnull().sum()
    cols_con_nans = nans[nans > 0]
    if len(cols_con_nans) == 0:
        print("  No hay valores faltantes.")
    else:
        print(f"  Columnas con NaN: {len(cols_con_nans)}")
        print("  (esperado: primeras semanas de cada comuna por los lags)")
        for col in cols_con_nans.index[:5]:
            print(f"    {col}: {nans[col]} NaN")
        if len(cols_con_nans) > 5:
            print(f"    ... y {len(cols_con_nans)-5} columnas más")

    print(f"\n  CRITERIOS DE ACEPTACIÓN HU5:")
    print(f"  ✓ Features de rezago temporal (cases_lag1-4, clima_lag1-4)")
    print(f"  ✓ Variables de estacionalidad (semana_sin, semana_cos, is_epidemic_season)")
    print(f"  ✓ Tasa de incidencia per cápita (incidencia_x10000)")
    print(f"  ✓ Variables espaciales: comuna_id (entero) + One-Hot (comuna_1..15) + es_comuna_1")
    print(f"  ✓ Normalización aplicada (sufijo _norm)")
    print(f"  ✓ Split temporal train/validation/test sin data leakage")
    print("=" * 65 + "\n")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_feature_engineering(save: bool = True) -> dict:
    """
    Pipeline completo de feature engineering.

    Returns:
        dict con df_train, df_val, df_test, df_full, scaler
    """
    print("\n" + "=" * 65)
    print("  SPRINT 4 — Feature Engineering")
    print("  HU5: Creación de features para el modelado")
    print("=" * 65 + "\n")

    # Pasos 1-2: carga e incidencia
    df, df_pob = cargar_datos()
    df = calcular_incidencia(df, df_pob)

    # Pasos 3-6: features
    df = crear_lags_casos(df)
    df = crear_lags_clima(df)
    df = crear_features_estacionalidad(df)
    df = codificar_comuna(df)

    # Paso 8: split temporal
    df_train, df_val, df_test = split_temporal(df)

    # Normalización (fit solo en train)
    df_train, df_val, df_test, scaler = aplicar_normalizacion(
        df_train, df_val, df_test
    )

    # Reconstruimos el dataset completo con todas las features
    df_full = pd.concat([df_train, df_val, df_test]).sort_values(
        ["comuna_id", "year", "epi_week"]
    ).reset_index(drop=True)

    # Paso 9: reporte
    reporte_features(df_full, df_train, df_val, df_test)

    # Paso 10: guardar
    if save:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)

        df_full.to_parquet(FEATURES_FILE, index=False)
        logger.info("Dataset features guardado en: %s", FEATURES_FILE)

        df_full.to_csv(FEATURES_CSV, index=False)
        logger.info("CSV de inspección: %s", FEATURES_CSV)

        # Guardamos los splits por separado
        df_train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
        df_val.to_parquet(PROCESSED_DIR / "validation.parquet", index=False)
        df_test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
        logger.info("Splits guardados: train.parquet, validation.parquet, test.parquet")

        # Guardamos el scaler para usarlo en producción
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Scaler guardado en: %s", SCALER_FILE)

    return {
        "df_full":  df_full,
        "df_train": df_train,
        "df_val":   df_val,
        "df_test":  df_test,
        "scaler":   scaler
    }


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    resultado = run_feature_engineering(save=True)

    df_train = resultado["df_train"]
    df_test  = resultado["df_test"]

    print("Muestra de features — primeras filas del train:")
    cols_muestra = [
        "year", "epi_week", "comuna_id",
        "confirmed_cases", "incidencia_x10000",
        "cases_lag1", "cases_lag2",
        "temp_mean_lag1", "temp_mean_lag4",
        "semana_sin", "semana_cos", "is_epidemic_season",
        "es_comuna_1", "comuna_1", "comuna_2", "comuna_9"
    ]
    cols_existentes = [c for c in cols_muestra if c in df_train.columns]
    print(df_train[cols_existentes].head(10).to_string(index=False))

    print(f"\nDataset features listo para el modelado (Sprint 5)")
