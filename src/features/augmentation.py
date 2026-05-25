"""
================================================================================
Data Augmentation — Series temporales de dengue en CABA
================================================================================

¿QUÉ ES DATA AUGMENTATION?
----------------------------
Data augmentation es una técnica que consiste en generar datos sintéticos
(artificiales pero realistas) a partir de los datos reales disponibles.

En imágenes es común rotar, reflejar o cambiar el brillo de una foto para
que el modelo vea más ejemplos. En series temporales se aplican técnicas
equivalentes: escalar los valores, agregar ruido pequeño o combinar series.

¿POR QUÉ LO NECESITAMOS?
--------------------------
El dataset de entrenamiento (2023) tiene un problema severo de desbalanceo:
  - 60% de semanas con 0 casos (temporada baja)
  - 2.1% de semanas con más de 200 casos (brote severo)
  - Solo 61 semanas con más de 50 casos en todo el año 2023

El brote de 2024 (validación) llegó a 1.391 casos por semana — más del doble
del máximo visto en entrenamiento (649 casos en la Comuna 1). El modelo nunca
vio un brote de esa magnitud y no puede predecirlo.

SOLUCIÓN: generamos versiones artificiales de las semanas de brote con
intensidades más altas (2x y 3x los valores originales), para que el modelo
"vea" brotes similares en magnitud al de 2024 durante el entrenamiento.

¿ES ESTO VÁLIDO?
----------------------------------
Sí. El data augmentation en series temporales epidemiológicas está documentado
en la literatura y es especialmente útil cuando:
  - Los eventos extremos (brotes) son raros en el dataset
  - El modelo necesita generalizar a intensidades no vistas durante el training
  - No es posible conseguir más datos históricos reales

Lo que NO hacemos: no inventamos patrones climáticos o geográficos nuevos.
Solo amplificamos la magnitud de brotes reales ya observados, manteniendo
la estructura temporal y las relaciones entre variables.

¿QUÉ GENERA ESTE SCRIPT?
--------------------------
Para cada semana de brote severo (>50 casos) en el set de entrenamiento:
  1. Escala x2: duplica los valores de casos y variables relacionadas
               + agrega ruido gaussiano pequeño (5% de variabilidad)
  2. Escala x3: triplica los valores
               + agrega ruido gaussiano pequeño (5% de variabilidad)

El ruido evita que las copias sean idénticas a los originales — el modelo
aprende patrones ligeramente diferentes, lo que mejora la generalización.

RESULTADO:
  Train original:   780 filas (61 semanas de brote severo)
  Train aumentado:  ~900 filas (61 + ~61x2 semanas de brote adicionales)

PREREQUISITO:
  Ejecutar src/features/lags.py antes de este script.

PIPELINE:
  Paso 1 → Cargar el train set original
  Paso 2 → Identificar semanas de brote severo (>50 casos)
  Paso 3 → Generar copias escaladas con ruido (x2 y x3)
  Paso 4 → Combinar con el train original
  Paso 5 → Guardar train_augmented.parquet
  Paso 6 → Generar reporte de la distribución antes/después
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Semilla para reproducibilidad del ruido gaussiano
np.random.seed(42)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

PROCESSED_DIR = Path("data/processed")
TRAIN_FILE    = PROCESSED_DIR / "train.parquet"
OUTPUT_FILE   = PROCESSED_DIR / "train_augmented.parquet"

# Umbral para considerar una semana como "brote severo"
# Elegimos 50 casos porque por debajo de ese valor el brote es leve
# y amplificarlo artificialmente podría generar valores poco realistas.
UMBRAL_BROTE = 50

# Factores de escala para la amplificación
# x2: simula un brote moderado-alto (como las semanas 10-20 de 2024)
# x3: simula un brote severo (como el pico de 2024 con ~1.391 casos)
FACTORES = [2.0, 3.0]

# Nivel de ruido gaussiano — 5% de variabilidad sobre el valor escalado
# Suficiente para que las copias no sean idénticas pero no tanto como
# para distorsionar los patrones epidemiológicos reales.
RUIDO_STD = 0.05

# Columnas que representan CASOS y deben escalarse con el factor
# Las columnas climáticas NO se escalan — el clima es independiente
# del número de casos y no tiene sentido amplificarlo.
COLS_CASOS = [
    "confirmed_cases",
    "incidencia_x10000",
    "cases_lag1",    "cases_lag2",    "cases_lag3",    "cases_lag4",
    "incidencia_lag1","incidencia_lag2","incidencia_lag3","incidencia_lag4",
    "casos_vecinas_lag1",
    "incidencia_vecinas_lag1",
    # Targets multi-horizonte — también deben escalarse
    "target_h1", "target_h2", "target_h3", "target_h4",
]

# Columnas normalizadas equivalentes (sufijo _norm)
# También se escalan para mantener consistencia con las features del LSTM/GRU
COLS_CASOS_NORM = [
    "cases_lag1_norm",    "cases_lag2_norm",
    "cases_lag3_norm",    "cases_lag4_norm",
    "incidencia_lag1_norm","incidencia_lag2_norm",
    "incidencia_lag3_norm","incidencia_lag4_norm",
]


# =============================================================================
# PASO 1: CARGA DE DATOS
# =============================================================================

def cargar_train() -> pd.DataFrame:
    """
    Carga el conjunto de entrenamiento original (2023 completo).
    Verifica que el archivo exista antes de proceder.
    """
    logger.info("--- PASO 1: Cargando train set original ---")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {TRAIN_FILE}\n"
            "Ejecutá src/features/lags.py primero."
        )

    df = pd.read_parquet(TRAIN_FILE)
    logger.info("  Train original: %d filas × %d columnas", len(df), len(df.columns))
    logger.info("  Período: año %d | Comunas: %d",
                df["year"].iloc[0], df["comuna_id"].nunique())

    return df


# =============================================================================
# PASO 2: IDENTIFICAR SEMANAS DE BROTE SEVERO
# =============================================================================

def identificar_semanas_brote(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica las filas del train donde los casos superan el umbral.

    ¿Por qué umbral de 50 casos?
    Por debajo de 50 casos el brote es leve — amplificarlo x3 daría 150 casos,
    que es un valor moderado-alto pero no representa la dinámica de un brote
    masivo. Con umbral de 50, amplificar x3 da hasta 1947 casos, comparable
    al brote real de 2024 (máximo 1391 casos).

    Retorna solo las filas donde confirmed_cases > UMBRAL_BROTE.
    """
    logger.info("--- PASO 2: Identificando semanas de brote severo ---")

    mask_brote = df["confirmed_cases"] > UMBRAL_BROTE
    df_brote   = df[mask_brote].copy()

    logger.info(
        "  Semanas con brote (>%d casos): %d de %d (%.1f%%)",
        UMBRAL_BROTE, len(df_brote), len(df),
        len(df_brote) / len(df) * 100
    )
    logger.info(
        "  Distribución de brote — media: %.1f | máx: %.0f | comunas: %s",
        df_brote["confirmed_cases"].mean(),
        df_brote["confirmed_cases"].max(),
        str(df_brote.groupby("comuna_id")["confirmed_cases"].max()
            .sort_values(ascending=False).head(3).to_dict())
    )

    return df_brote


# =============================================================================
# PASO 3: GENERAR COPIAS ESCALADAS CON RUIDO
# =============================================================================

def escalar_con_ruido(
    df_brote: pd.DataFrame,
    factor: float,
    ruido_std: float = RUIDO_STD
) -> pd.DataFrame:
    """
    Genera una copia de las semanas de brote escalada por un factor,
    con ruido gaussiano pequeño para evitar copias exactas.

    ¿Qué se escala?
      - Columnas de casos: confirmed_cases, incidencia, lags de casos,
        vecindad espacial, targets futuros y sus versiones normalizadas.
      - El ruido se agrega DESPUÉS de escalar para mantener proporcionalidad.

    ¿Qué NO se escala?
      - Variables climáticas: la temperatura, lluvia y humedad son
        independientes del número de casos — no tiene sentido amplificarlas.
      - Estacionalidad: semana_sin, semana_cos, is_epidemic_season.
      - Variables espaciales fijas: comuna_id, población.
      - Año y semana epidemiológica: siguen siendo de 2023.

    El ruido gaussiano tiene media=0 y desvío=ruido_std × valor_escalado.
    Con ruido_std=0.05 la variabilidad es del 5% — suficiente para
    diversificar sin distorsionar.

    Retorna un DataFrame con las filas aumentadas listas para concatenar.
    """
    df_aug = df_brote.copy()

    # Identificar columnas a escalar que existen en el dataset
    cols_escalar = [c for c in COLS_CASOS + COLS_CASOS_NORM if c in df_aug.columns]

    for col in cols_escalar:
        valores_originales = df_aug[col].fillna(0).values

        # Escalar por el factor
        valores_escalados = valores_originales * factor

        # Agregar ruido gaussiano proporcional al valor escalado
        # El ruido tiene desvío = ruido_std × |valor_escalado|
        # Así el ruido es pequeño donde los valores son pequeños y
        # proporcional donde los valores son grandes
        ruido = np.random.normal(
            loc=0,
            scale=ruido_std * np.abs(valores_escalados) + 1e-6,
            size=len(valores_escalados)
        )

        # Aplicar ruido y garantizar que los casos no sean negativos
        valores_finales = np.maximum(0, valores_escalados + ruido)

        df_aug[col] = valores_finales

    # Marcar las filas como sintéticas para trazabilidad
    df_aug["es_sintetico"] = True
    df_aug["factor_escala"] = factor

    logger.info(
        "  Factor x%.1f: %d filas generadas | media casos: %.1f | máx: %.0f",
        factor,
        len(df_aug),
        df_aug["confirmed_cases"].mean(),
        df_aug["confirmed_cases"].max()
    )

    return df_aug


# =============================================================================
# PASO 4: COMBINAR DATOS ORIGINALES Y AUMENTADOS
# =============================================================================

def combinar_datasets(
    df_original: pd.DataFrame,
    lista_aumentados: list[pd.DataFrame]
) -> pd.DataFrame:
    """
    Combina el dataset original con todas las versiones aumentadas.

    El dataset original recibe la marca es_sintetico=False para distinguirlo
    de los datos generados artificialmente.

    Orden de concatenación:
      1. Train original (2023 completo, 780 filas)
      2. Versiones x2 de semanas de brote (~61 filas)
      3. Versiones x3 de semanas de brote (~61 filas)

    El orden no importa para el entrenamiento del LSTM/GRU porque las
    secuencias se construyen por comuna en orden cronológico — pero
    mantenerlo ordenado facilita la inspección del dataset.
    """
    df_original = df_original.copy()
    df_original["es_sintetico"]  = False
    df_original["factor_escala"] = 1.0

    df_combinado = pd.concat(
        [df_original] + lista_aumentados,
        ignore_index=True
    )

    return df_combinado


# =============================================================================
# PASO 5: GUARDAR Y REPORTAR
# =============================================================================

def guardar_y_reportar(
    df_original: pd.DataFrame,
    df_aumentado: pd.DataFrame
):
    """
    Guarda el dataset aumentado y muestra un reporte comparativo
    de la distribución antes y después del augmentation.

    El reporte permite verificar que:
    1. El augmentation no eliminó datos originales
    2. La distribución de brotes mejoró (más ejemplos en rangos altos)
    3. Las columnas climáticas no fueron modificadas
    """
    logger.info("--- PASO 5: Guardando y reportando ---")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_aumentado.to_parquet(OUTPUT_FILE, index=False)
    logger.info("  Dataset aumentado guardado: %s", OUTPUT_FILE)

    print("\n" + "=" * 65)
    print("  REPORTE DE DATA AUGMENTATION")
    print("=" * 65)

    print(f"\n  Dataset original:")
    print(f"    Total filas:           {len(df_original):>6}")
    print(f"    Filas reales:          {len(df_original):>6}")
    print(f"    Semanas con 0 casos:   {(df_original['confirmed_cases']==0).sum():>6} "
          f"({(df_original['confirmed_cases']==0).mean()*100:.1f}%)")
    print(f"    Semanas con >50 casos: {(df_original['confirmed_cases']>50).sum():>6} "
          f"({(df_original['confirmed_cases']>50).mean()*100:.1f}%)")
    print(f"    Máximo de casos:       {df_original['confirmed_cases'].max():>6.0f}")

    df_reales = df_aumentado[~df_aumentado["es_sintetico"]]
    df_sint   = df_aumentado[df_aumentado["es_sintetico"]]

    print(f"\n  Dataset aumentado:")
    print(f"    Total filas:           {len(df_aumentado):>6}")
    print(f"    Filas reales:          {len(df_reales):>6}")
    print(f"    Filas sintéticas:      {len(df_sint):>6} "
          f"({len(df_sint)/len(df_aumentado)*100:.1f}%)")
    print(f"    Semanas con 0 casos:   {(df_reales['confirmed_cases']==0).sum():>6} "
          f"({(df_reales['confirmed_cases']==0).mean()*100:.1f}% de reales)")
    print(f"    Semanas con >50 casos (real+sint): "
          f"{(df_aumentado['confirmed_cases']>50).sum():>4} "
          f"({(df_aumentado['confirmed_cases']>50).mean()*100:.1f}%)")
    print(f"    Máximo de casos (sint):{df_sint['confirmed_cases'].max():>6.0f}")

    print(f"\n  Distribución de casos por rango (dataset aumentado):")
    rangos = [(0, 0), (1, 50), (51, 200), (201, 500), (501, 9999)]
    labels = ["= 0", "1-50", "51-200", "201-500", "> 500"]
    for (lo, hi), label in zip(rangos, labels):
        mask = (df_aumentado["confirmed_cases"] >= lo) & (df_aumentado["confirmed_cases"] <= hi)
        n    = mask.sum()
        pct  = n / len(df_aumentado) * 100
        barra = "█" * int(pct / 2)
        print(f"    {label:>8}: {n:>4} filas ({pct:>5.1f}%) {barra}")

    print(f"\n  ✓ Variables climáticas NO modificadas (solo casos y lags de casos)")
    print(f"  ✓ Ruido gaussiano aplicado (std={RUIDO_STD*100:.0f}% del valor escalado)")
    print(f"  ✓ Archivo guardado: {OUTPUT_FILE}")
    print("=" * 65 + "\n")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_augmentation() -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de data augmentation.

    Genera train_augmented.parquet que contiene:
      - Todas las filas originales del train (2023)
      - Copias x2 de las semanas con brote severo (>50 casos)
      - Copias x3 de las semanas con brote severo (>50 casos)

    Este archivo es leído por lstm_model.py y gru_model.py en lugar
    del train.parquet original para que los modelos vean más ejemplos
    de brotes intensos durante el entrenamiento.
    """
    print("\n" + "=" * 65)
    print("  DATA AUGMENTATION — Series temporales de dengue en CABA")
    print(f"  Umbral de brote: >{UMBRAL_BROTE} casos")
    print(f"  Factores de escala: {FACTORES}")
    print(f"  Ruido gaussiano: {RUIDO_STD*100:.0f}%")
    print("=" * 65 + "\n")

    # Paso 1: cargar datos originales
    df_original = cargar_train()

    # Paso 2: identificar semanas de brote
    df_brote = identificar_semanas_brote(df_original)

    # Paso 3: generar copias escaladas para cada factor
    logger.info("--- PASO 3: Generando versiones escaladas ---")
    aumentados = []
    for factor in FACTORES:
        df_aug = escalar_con_ruido(df_brote, factor)
        aumentados.append(df_aug)

    # Paso 4: combinar
    logger.info("--- PASO 4: Combinando datasets ---")
    df_final = combinar_datasets(df_original, aumentados)
    logger.info(
        "  Dataset final: %d filas (%d originales + %d sintéticas)",
        len(df_final),
        len(df_original),
        len(df_final) - len(df_original)
    )

    # Paso 5: guardar y reportar
    guardar_y_reportar(df_original, df_final)

    return df_final


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    df_aumentado = run_augmentation()
    print("Data augmentation completado.")
    print(f"Próximo paso: ejecutar lstm_model.py y gru_model.py")
    print(f"Ambos scripts leerán train_augmented.parquet automáticamente.")
